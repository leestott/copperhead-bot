#!/usr/bin/env python3
"""
bot.py - CopperHead Tournament Bot v2 Entry Point

Improvements over v1:
- Tracks when opponents ate food (tail didn't move)
- More robust message handling for edge-case server messages
- Game-state history for post-mortem analysis
"""

import os
import json
import asyncio
import argparse
import logging
import random
import signal
import socket
import time
from collections import deque
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidHandshake,
    InvalidURI,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    logging.getLogger(__name__).debug("python-dotenv not installed; .env files will not be loaded")
except Exception:
    logging.getLogger(__name__).exception("Unexpected error importing/loading dotenv")

from strategy import StrategyEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

_MAX_RECONNECT_DELAY = 60
_BASE_RECONNECT_DELAY = 2


class CopperheadBot:
    """
    Main bot class handling WebSocket communication and game loop.
    """

    def __init__(
        self,
        server_url: str,
        name: str = "CopperheadChampion",
        difficulty: int = 10,
        quiet: bool = False,
    ):
        self.server_url = server_url
        self.name = name
        self.difficulty = difficulty
        self.quiet = quiet

        # Connection state
        self.ws: Optional[Any] = None
        self.connected: bool = False
        self.player_id: Optional[int] = None
        self.room_id: Optional[str] = None
        self._last_ready_sent_at = 0.0

        # Game state
        self.game_state: Optional[dict[str, Any]] = None
        self.game_running = False
        self.grid_width = 30
        self.grid_height = 20
        self._round_state_history: deque[dict[str, Any]] = deque(maxlen=20)
        self._last_round_summary: Optional[dict[str, Any]] = None
        self._pending_named_opponent_summary: Optional[dict[str, Any]] = None

        # Strategy engine
        self.strategy = StrategyEngine(difficulty=difficulty)

        # Match tracking
        self.current_wins = 0
        self.points_to_win = 3
        self._match_redeploys = 0
        self._observed_match_score: dict[str, int] = {}
        self._round_redeploys = 0
        self.redeploy_after_round_loss = os.environ.get("BOT_REDEPLOY_AFTER_ROUND_LOSS", "true").lower() == "true"
        self.redeploy_ladder_only = os.environ.get("BOT_REDEPLOY_LADDER_ONLY", "true").lower() == "true"

        # Shutdown flag
        self.shutdown_requested = False

        # Reconnection tracking
        self._consecutive_failures = 0

        if quiet:
            logger.setLevel(logging.WARNING)

    _LOG_METHODS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def log(self, msg: str, level: str = "info") -> None:
        if not self.quiet:
            numeric_level = self._LOG_METHODS.get(level, logging.INFO)
            logger.log(numeric_level, msg)

    def _is_us(self, value: Any) -> bool:
        return self.player_id is not None and str(value) == str(self.player_id)

    def _reset_match_context(self) -> None:
        self.room_id = None
        self.current_wins = 0
        self.points_to_win = 3
        self._observed_match_score.clear()
        self.game_running = False
        self.game_state = None
        self._round_state_history.clear()
        self._last_round_summary = None
        try:
            if hasattr(self.strategy, "set_current_opponent"):
                self.strategy.set_current_opponent("")
        except Exception:
            logger.exception("Error clearing opponent context")

    @staticmethod
    def _normalize_scoreboard(scoreboard: Any) -> dict[str, int]:
        if not isinstance(scoreboard, dict):
            return {}
        normalized: dict[str, int] = {}
        for key, value in scoreboard.items():
            if isinstance(value, int):
                normalized[str(key)] = value
        return normalized

    def _determine_match_outcome(self, winner: Any, final_score: Any) -> Optional[bool]:
        """Return True if we won the match, False if we lost, else None if unknown."""
        if self.player_id is None:
            return None

        observed_score = dict(self._observed_match_score)
        reported_score = self._normalize_scoreboard(final_score)
        score_source = observed_score or reported_score

        if observed_score and reported_score and observed_score != reported_score:
            self.log(
                f"Score mismatch: observed={observed_score} reported={reported_score}; "
                "trusting observed round stream",
                "warning",
            )

        if score_source:
            my_score = score_source.get(str(self.player_id))
            if isinstance(my_score, int):
                best_score = max(score_source.values())
                leaders = [sid for sid, score in score_source.items() if score == best_score]
                if str(self.player_id) in leaders and len(leaders) == 1:
                    return True
                if str(self.player_id) not in leaders:
                    return False

        if self._is_us(winner):
            return True
        if winner is not None:
            return False
        return None

    async def _redeploy_after_loss(self, reason: str) -> None:
        """Recycle the websocket session after a match loss to rejoin fresh."""
        if "round" in reason:
            self._round_redeploys += 1
            redeploy_count = self._round_redeploys
        else:
            self._match_redeploys += 1
            redeploy_count = self._match_redeploys
        self.log(
            f"Redeploying bot session after {reason} (rejoin #{redeploy_count})",
            "warning",
        )
        self._reset_match_context()
        if self.ws:
            await self.disconnect()

    def _should_redeploy_after_round_loss(self) -> bool:
        if not self.redeploy_after_round_loss:
            return False
        if not self.redeploy_ladder_only:
            return True
        try:
            if hasattr(self.strategy, "is_ladder_opponent"):
                return bool(self.strategy.is_ladder_opponent())
        except Exception:
            logger.exception("Error checking ladder-opponent state")
        return False

    def _attribute_last_round_to_named_opponent(self, opponent_name: Any) -> None:
        if not isinstance(opponent_name, str):
            return
        opponent_name = opponent_name.strip()
        if not opponent_name:
            return
        summary = self._last_round_summary or self._pending_named_opponent_summary
        if not summary:
            return
        try:
            if hasattr(self.strategy, "set_current_opponent"):
                self.strategy.set_current_opponent(opponent_name)
            if hasattr(self.strategy, "learn_from_round"):
                self.strategy.learn_from_round(summary)
            self._pending_named_opponent_summary = None
        except Exception:
            logger.exception("Error attributing last round to named opponent")

    @staticmethod
    def _build_connect_url(server_url: str) -> str:
        parsed = urlparse(server_url)
        path = parsed.path.rstrip("/")
        if not path.endswith("/join"):
            path = path + "/join"
        return urlunparse(parsed._replace(path=path))

    async def connect(self) -> bool:
        url = self._build_connect_url(self.server_url)
        self.log(f"Connecting to {url}...")
        try:
            self.ws = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            )
            self.connected = True
            self._consecutive_failures = 0
            self.log("Connected successfully!")
            await self.send_join()
            return True
        except asyncio.CancelledError:
            raise
        except socket.gaierror as e:
            self.log(f"DNS resolution failed for {url}: {e}", "error")
            return False
        except ConnectionRefusedError as e:
            self.log(f"Connection refused at {url}: {e}", "error")
            return False
        except (OSError, asyncio.TimeoutError) as e:
            self.log(f"Connection failed (network): {e}", "error")
            return False
        except (InvalidURI, InvalidHandshake) as e:
            self.log(f"Connection failed (handshake/URI): {e}", "error")
            return False
        except Exception:
            logger.exception("Unexpected error during connect to %s", url)
            return False

    async def disconnect(self) -> None:
        if self.ws:
            try:
                await self.ws.close()
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, asyncio.TimeoutError):
                logger.debug("WebSocket already closed or timed out during disconnect")
            except OSError as e:
                logger.warning("OS error closing websocket: %s", e)
            except Exception:
                logger.exception("Unexpected error closing websocket")
            finally:
                self.ws = None
        self.connected = False
        self.log("Disconnected from server")

    async def send_message(self, message: dict[str, Any]) -> None:
        if not self.ws:
            return
        try:
            msg_str = json.dumps(message)
        except (TypeError, ValueError) as e:
            self.log(f"Message serialization failed: {e}", "error")
            return
        try:
            await self.ws.send(msg_str)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as e:
            self.log(f"Send failed (connection closed): {e}", "warning")
            self.connected = False
            self.ws = None
        except OSError as e:
            self.log(f"Send failed (OS error): {e}", "error")
            self.connected = False
            self.ws = None
        except Exception:
            logger.exception("Unexpected error sending message")

    async def send_join(self) -> None:
        await self.send_message({"action": "join", "name": self.name})
        self.log(f"Sent join request as '{self.name}'")

    async def send_ready(self) -> None:
        self._last_ready_sent_at = time.monotonic()
        await self.send_message({"action": "ready", "name": self.name})
        self.log(f"Sent ready signal as '{self.name}'")

    async def send_move(self, direction: str) -> None:
        if direction not in ("up", "down", "left", "right"):
            self.log(f"Invalid direction: {direction}", "warning")
            return
        await self.send_message({"action": "move", "direction": direction})

    async def handle_message(self, data: dict[str, Any]) -> None:
        msg_type = data.get("type", "")

        if msg_type == "lobby_joined":
            self.log(f"Joined lobby as '{data.get('name', self.name)}'")

        elif msg_type == "lobby_update":
            self.log("Lobby updated", "debug")

        elif msg_type == "lobby_kicked":
            self.log("Kicked from lobby by admin", "warning")
            self.shutdown_requested = True

        elif msg_type == "lobby_left":
            self.log("Left lobby")
            self.connected = False
            if self.ws:
                await self.disconnect()

        elif msg_type == "joined":
            self.player_id = data.get("player_id")
            self.room_id = data.get("room_id")
            self.log(f"Joined room {self.room_id} as player {self.player_id}")
            await self.send_ready()

        elif msg_type == "waiting":
            self.log("Waiting for opponent...")
            if time.monotonic() - self._last_ready_sent_at > 15:
                await self.send_ready()

        elif msg_type == "start":
            mode = data.get("mode", "two_player")
            self.log(f"Game starting! Mode: {mode}")
            self.game_running = True
            self._round_state_history.clear()
            try:
                self.strategy.reset_game_state()
            except Exception:
                logger.exception("Error resetting strategy state")

        elif msg_type == "state":
            game = data.get("game")
            if not isinstance(game, dict):
                self.log("Received 'state' message with invalid game payload", "warning")
                return
            await self.handle_game_state(game)

        elif msg_type == "gameover":
            winner = data.get("winner")
            wins = data.get("wins", {})
            normalized_wins = self._normalize_scoreboard(wins)
            if self.player_id is not None and normalized_wins:
                self._observed_match_score = normalized_wins
                self.current_wins = normalized_wins.get(str(self.player_id), 0)
            else:
                self.log("Cannot read wins: player_id not set or wins invalid", "warning")
            self.points_to_win = data.get("points_to_win", 3)

            we_won = self._is_us(winner)
            try:
                round_summary = self._build_round_summary(we_won)
                self._last_round_summary = round_summary
                if hasattr(self.strategy, "learn_from_round"):
                    self.strategy.learn_from_round(round_summary)
                self.strategy.record_game_result(we_won)
            except Exception:
                logger.exception("Error recording game result")

            result = "WON" if we_won else "LOST"
            self.log(f"Round {result}! Score: {self.current_wins}/{self.points_to_win}")
            if hasattr(self.strategy, "get_adaptation_summary"):
                self.log(f"Adaptive profile: {self.strategy.get_adaptation_summary()}", "info")
            if self._last_round_summary:
                self.log(
                    "Round analysis: "
                    f"wall={self._last_round_summary.get('last_state', {}).get('wall_distance')} "
                    f"free={self._last_round_summary.get('last_state', {}).get('free_ratio', 0):.2f} "
                    f"opp_dist={self._last_round_summary.get('last_state', {}).get('nearest_opponent_distance')}",
                    "debug",
                )

            self.game_running = False
            if not we_won and self._should_redeploy_after_round_loss():
                await self._redeploy_after_loss("round loss")
                return
            await self.send_ready()

        elif msg_type == "match_complete":
            winner = data.get("winner")
            final_score = self._normalize_scoreboard(data.get("final_score", {}))
            did_win = self._determine_match_outcome(winner, final_score)
            if did_win is True:
                self.log("MATCH WON!")
            elif did_win is False:
                self.log("Match lost.")
            else:
                self.log("Match outcome unclear from server payload.", "warning")
            self.log(f"Final score: {final_score}")
            self.game_running = False
            if did_win is False:
                self._pending_named_opponent_summary = self._last_round_summary
                await self._redeploy_after_loss("match loss")
            else:
                self._reset_match_context()

        elif msg_type == "match_assigned":
            self.room_id = data.get("room_id")
            self.player_id = data.get("player_id")
            opponent = data.get("opponent", "Unknown")
            try:
                if hasattr(self.strategy, "set_current_opponent"):
                    self.strategy.set_current_opponent(str(opponent))
            except Exception:
                logger.exception("Error setting opponent context")
            self.log(f"New match assigned! vs {opponent} in room {self.room_id}")
            self.current_wins = 0
            await self.send_ready()

        elif msg_type == "competition_complete":
            champion = data.get("champion", "Unknown")
            champion_name = champion.get("name") if isinstance(champion, dict) else champion
            if champion_name == self.name:
                self.log("TOURNAMENT CHAMPION!")
            else:
                self.log(f"Tournament complete. Champion: {champion}")
                self._attribute_last_round_to_named_opponent(champion_name)
            self.shutdown_requested = True

        elif msg_type == "error":
            error_msg = data.get("message", "Unknown error")
            self.log(f"Server error: {error_msg}", "error")

        elif msg_type == "ready_required":
            self.log("Server requires ready signal")
            await self.send_ready()

        else:
            if msg_type:
                self.log(f"Unknown message type: {msg_type}", "debug")

    async def handle_game_state(self, game: dict[str, Any]) -> None:
        if not game.get("running", False):
            return
        if self.player_id is None:
            self.log("Received game state but player_id is not set, skipping", "warning")
            return

        self.game_state = game
        self._record_round_state(game)

        # Update grid dimensions with validation
        grid = game.get("grid", {})
        if isinstance(grid, dict):
            raw_w = grid.get("width", 30)
            raw_h = grid.get("height", 20)
            if isinstance(raw_w, int) and raw_w > 0:
                self.grid_width = raw_w
            if isinstance(raw_h, int) and raw_h > 0:
                self.grid_height = raw_h
        try:
            self.strategy.update_grid_size(self.grid_width, self.grid_height)
        except Exception:
            logger.exception("Error updating grid size")

        snakes = game.get("snakes")
        if not isinstance(snakes, dict):
            self.log("Invalid snakes payload in game state", "warning")
            return

        my_snake = snakes.get(str(self.player_id))
        if not my_snake or not my_snake.get("alive", True):
            return

        direction = self._safe_calculate_move(game, self.player_id, my_snake)
        try:
            if hasattr(self.strategy, "get_last_decision_trace"):
                trace = self.strategy.get_last_decision_trace()
                if trace:
                    logger.info("Opening trace: %s", trace)
        except Exception:
            logger.exception("Error reading strategy decision trace")

        # Per-tick diagnostic logging
        body = my_snake.get("body", [])
        head_pos = body[0] if body else [0, 0]
        opp_info = ""
        for sid, sd in snakes.items():
            if sid != str(self.player_id) and sd.get("alive") and sd.get("body"):
                oh = sd["body"][0]
                opp_info = f" opp=({oh[0]},{oh[1]}) olen={len(sd['body'])}"
                break
        foods = game.get("foods", [])
        food_info = f" food={len(foods)}"
        if foods:
            fp = foods[0]
            food_info += f" f0=({fp.get('x','?')},{fp.get('y','?')})"
        logger.info(
            f"T{self.strategy.tick} pos=({head_pos[0]},{head_pos[1]}) "
            f"len={len(body)} -> {direction}{opp_info}{food_info}"
        )

        if direction:
            await self.send_move(direction)
        else:
            current_dir = my_snake.get("direction", "right")
            self.log(f"No valid move found, maintaining {current_dir}", "warning")
            await self.send_move(current_dir)

    def _safe_calculate_move(
        self, game: dict[str, Any], player_id: int, my_snake: dict[str, Any]
    ) -> Optional[str]:
        try:
            return self.strategy.calculate_move(game, player_id)
        except Exception:
            logger.exception("Strategy engine error; falling back to safe move")
            return self._fallback_direction(my_snake)

    def _fallback_direction(self, my_snake: dict[str, Any]) -> Optional[str]:
        current_dir = my_snake.get("direction", "right")
        try:
            body = my_snake.get("body", [])
            if not body:
                return current_dir
            seg = body[0]
            head = (int(seg[0]), int(seg[1]))
            body_set: set[tuple[int, int]] = set()
            for s in body:
                body_set.add((int(s[0]), int(s[1])))
        except (TypeError, IndexError, ValueError) as e:
            logger.warning("Malformed snake body in fallback: %s", e)
            return current_dir

        opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
        best_dir = None
        best_space = -1
        for d in ("up", "down", "left", "right"):
            if d == opposites.get(current_dir):
                continue
            dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
            nx2, ny2 = head[0] + dx, head[1] + dy
            if 0 <= nx2 < self.grid_width and 0 <= ny2 < self.grid_height and (nx2, ny2) not in body_set:
                # Pick direction with most space
                from pathfinding import flood_fill_count
                space = flood_fill_count((nx2, ny2), self.grid_width, self.grid_height, body_set, max_depth=10)
                if space > best_space:
                    best_space = space
                    best_dir = d
        return best_dir if best_dir else current_dir

    def _record_round_state(self, game: dict[str, Any]) -> None:
        if self.player_id is None:
            return
        snakes = game.get("snakes")
        if not isinstance(snakes, dict):
            return
        my_snake = snakes.get(str(self.player_id))
        if not isinstance(my_snake, dict):
            return
        body = my_snake.get("body", [])
        if not body:
            return

        head = body[0]
        opponents = []
        nearest_opponent_distance = None
        nearest_opponent_length = None
        for sid, snake_data in snakes.items():
            if sid == str(self.player_id) or not snake_data.get("alive", True):
                continue
            opp_body = snake_data.get("body", [])
            if not opp_body:
                continue
            opp_head = opp_body[0]
            dist = abs(head[0] - opp_head[0]) + abs(head[1] - opp_head[1])
            opponents.append(dist)
            if nearest_opponent_distance is None or dist < nearest_opponent_distance:
                nearest_opponent_distance = dist
                nearest_opponent_length = len(opp_body)

        grid = game.get("grid", {}) if isinstance(game.get("grid"), dict) else {}
        width = grid.get("width", self.grid_width)
        height = grid.get("height", self.grid_height)
        total_cells = max(1, width * height)
        occupied = 0
        for snake_data in snakes.values():
            occupied += len(snake_data.get("body", []))

        foods = game.get("foods", [])
        food_positions = {(f.get("x"), f.get("y")) for f in foods if isinstance(f, dict)}

        self._round_state_history.append(
            {
                "tick": getattr(self.strategy, "tick", len(self._round_state_history)),
                "head": (head[0], head[1]),
                "my_length": len(body),
                "wall_distance": min(head[0], head[1], width - 1 - head[0], height - 1 - head[1]),
                "free_ratio": max(0.0, (total_cells - occupied) / total_cells),
                "nearest_opponent_distance": nearest_opponent_distance,
                "nearest_opponent_length": nearest_opponent_length,
                "ate_recently": (head[0], head[1]) in food_positions,
                "aggressive_posture": nearest_opponent_distance is not None and nearest_opponent_distance <= 3,
            }
        )

    def _build_round_summary(self, we_won: bool) -> dict[str, Any]:
        last_state = self._round_state_history[-1] if self._round_state_history else {}
        return {
            "won": we_won,
            "ticks_seen": len(self._round_state_history),
            "last_state": last_state,
        }

    async def play(self) -> None:
        if not self.ws:
            self.log("Not connected!", "error")
            return

        self.log("Starting game loop...")

        try:
            async for message in self.ws:
                if self.shutdown_requested:
                    break

                try:
                    data = json.loads(message)
                    if not isinstance(data, dict):
                        self.log(f"Non-dict JSON message: {str(message)[:100]}", "warning")
                        continue
                    await self.handle_message(data)
                except json.JSONDecodeError:
                    self.log(f"Invalid JSON: {message[:100]}", "warning")
                except Exception:
                    logger.exception("Error handling message")

        except asyncio.CancelledError:
            raise
        except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError) as e:
            self.log(f"Connection closed: {e}", "warning")
        except Exception:
            logger.exception("Unexpected game loop error")

        self.log("Game loop ended")

    async def run(self) -> None:
        while not self.shutdown_requested:
            if await self.connect():
                self._consecutive_failures = 0
                await self.play()
                await self.disconnect()
            else:
                self._consecutive_failures += 1

            if not self.shutdown_requested:
                delay = min(
                    _BASE_RECONNECT_DELAY * (2 ** self._consecutive_failures),
                    _MAX_RECONNECT_DELAY,
                )
                delay += random.uniform(0, delay * 0.25)
                if self._consecutive_failures >= 5:
                    self.log(
                        f"Reconnect attempt {self._consecutive_failures} failed; "
                        f"retrying in {delay:.1f}s",
                        "warning",
                    )
                else:
                    self.log(f"Reconnecting in {delay:.1f}s...")
                await asyncio.sleep(delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CopperHead Tournament Bot v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bot.py --server ws://localhost:8765/ws/ --name "MyBotV2"
  python bot.py -s wss://server.example.com/ws/ -n "Champion" -d 10

Environment variables:
  COPPERHEAD_SERVER_URL  Server WebSocket URL
  BOT_NAME               Bot display name
  BOT_DIFFICULTY         Difficulty level (1-10)
  BOT_QUIET              Suppress output (true/false)
""",
    )

    parser.add_argument(
        "--server", "-s",
        default=os.environ.get("COPPERHEAD_SERVER_URL", "ws://localhost:8765/ws/"),
        help="Server WebSocket URL",
    )
    parser.add_argument(
        "--name", "-n",
        default=os.environ.get("BOT_NAME", "CopperheadChampion"),
        help="Bot display name",
    )
    _raw_difficulty = os.environ.get("BOT_DIFFICULTY", "10")
    try:
        _default_difficulty = int(_raw_difficulty)
    except ValueError:
        logger.warning("Invalid BOT_DIFFICULTY env var %r, defaulting to 10", _raw_difficulty)
        _default_difficulty = 10

    parser.add_argument(
        "--difficulty", "-d",
        type=int,
        default=_default_difficulty,
        help="Difficulty level 1-10 (10 = optimal play)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=os.environ.get("BOT_QUIET", "").lower() == "true",
        help="Suppress console output",
    )

    return parser.parse_args()


def _validate_server_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("Server URL must not be empty")
    parsed = urlparse(url)
    if parsed.scheme not in ("ws", "wss"):
        raise ValueError(
            f"Server URL must use ws:// or wss:// scheme, got {parsed.scheme!r}"
        )
    return url


async def main() -> None:
    args = parse_args()

    try:
        server_url = _validate_server_url(args.server)
    except ValueError as e:
        logger.error("Invalid --server value: %s", e)
        raise SystemExit(1) from e

    name = args.name.strip() if args.name else ""
    if not name:
        logger.error("Bot name (--name) must be a non-empty string")
        raise SystemExit(1)

    difficulty = max(1, min(10, args.difficulty))

    bot = CopperheadBot(
        server_url=server_url,
        name=name,
        difficulty=difficulty,
        quiet=args.quiet,
    )

    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        bot.shutdown_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: signal_handler())

    logger.info(f"CopperHead Bot v2 starting: {bot.name}")
    logger.info(f"Server: {bot.server_url}")
    logger.info(f"Difficulty: {difficulty}/10")

    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Fatal error in main loop")
    finally:
        await bot.disconnect()

    logger.info("Bot shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
