#!/usr/bin/env python3
"""
bot.py - CopperHead Tournament Bot Entry Point

A high-performance competitive Snake AI for CopperHead tournaments.
Connects via WebSocket and autonomously plays using multi-layer strategy.

Usage:
    python bot.py --server ws://localhost:8765/ws/ --name "MyBot"
    
Or with environment variables:
    COPPERHEAD_SERVER_URL=ws://localhost:8765/ws/ python bot.py
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import random
import signal
import socket
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import websockets
import websockets.exceptions
from websockets.client import WebSocketClientProtocol

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present
except ImportError:
    pass  # dotenv is optional

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
    
    Connects to CopperHead server, processes game messages, and sends
    move commands based on strategy engine decisions.
    """
    
    def __init__(
        self,
        server_url: str,
        name: str = "CopperheadChampion",
        difficulty: int = 10,
        quiet: bool = False
    ):
        """
        Initialize the bot.
        
        Args:
            server_url: WebSocket URL of CopperHead server
            name: Bot display name
            difficulty: Strategy difficulty (1-10)
            quiet: Suppress log output
        """
        self.server_url = server_url
        self.name = name
        self.difficulty = difficulty
        self.quiet = quiet
        
        # Connection state
        self.ws: Optional[WebSocketClientProtocol] = None
        self.connected: bool = False
        self.player_id: Optional[int] = None
        self.room_id: Optional[str] = None
        
        # Game state
        self.game_state: Optional[dict[str, Any]] = None
        self.game_running = False
        self.grid_width = 30
        self.grid_height = 20
        
        # Strategy engine
        self.strategy = StrategyEngine(difficulty=difficulty)
        
        # Match tracking
        self.current_wins = 0
        self.points_to_win = 3
        
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
        """Log message if not in quiet mode."""
        if not self.quiet:
            numeric_level = self._LOG_METHODS.get(level, logging.INFO)
            logger.log(numeric_level, msg)
    
    @staticmethod
    def _build_connect_url(server_url: str) -> str:
        """Build the websocket connect URL, appending /join if needed."""
        parsed = urlparse(server_url)
        path = parsed.path.rstrip("/")
        if not path.endswith("/join"):
            path = path + "/join"
        return urlunparse(parsed._replace(path=path))

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to server.
        
        Returns:
            True if connection successful
        """
        url = self._build_connect_url(self.server_url)
        self.log(f"Connecting to {url}...")
        try:
            self.ws = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10
            )
            self.connected = True
            self._consecutive_failures = 0
            self.log("Connected successfully!")
            return True
        except socket.gaierror as e:
            self.log(f"DNS resolution failed for {url}: {e}", "error")
            return False
        except ConnectionRefusedError as e:
            self.log(f"Connection refused at {url}: {e}", "error")
            return False
        except (OSError, asyncio.TimeoutError) as e:
            self.log(f"Connection failed (network): {e}", "error")
            return False
        except (
            websockets.exceptions.InvalidURI,
            websockets.exceptions.InvalidHandshake,
        ) as e:
            self.log(f"Connection failed (handshake/URI): {e}", "error")
            return False
        except Exception:
            logger.exception("Unexpected error during connect to %s", url)
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        if self.ws:
            try:
                await self.ws.close()
            except websockets.exceptions.ConnectionClosed:
                logger.debug("WebSocket already closed during disconnect")
            except Exception:
                logger.exception("Unexpected error closing websocket")
            finally:
                self.ws = None
        self.connected = False
        self.log("Disconnected from server")
    
    async def send_message(self, message: dict[str, Any]) -> None:
        """
        Send JSON message to server.
        
        Args:
            message: Dict to send as JSON
        """
        if not self.ws:
            return
        
        try:
            msg_str = json.dumps(message)
        except (TypeError, ValueError) as e:
            self.log(f"Message serialization failed: {e}", "error")
            return
        
        try:
            await self.ws.send(msg_str)
        except websockets.exceptions.ConnectionClosed as e:
            self.log(f"Send failed (connection closed): {e}", "warning")
            self.connected = False
            self.ws = None
        except Exception:
            logger.exception("Unexpected error sending message")
    
    async def send_ready(self) -> None:
        """Send ready signal to server."""
        await self.send_message({
            "action": "ready",
            "mode": "two_player",
            "name": self.name
        })
        self.log(f"Sent ready signal as '{self.name}'")
    
    async def send_move(self, direction: str) -> None:
        """
        Send move command to server.
        
        Args:
            direction: One of "up", "down", "left", "right"
        """
        if direction not in ("up", "down", "left", "right"):
            self.log(f"Invalid direction: {direction}", "warning")
            return
        
        await self.send_message({
            "action": "move",
            "direction": direction
        })
    
    async def handle_message(self, data: dict[str, Any]) -> None:
        """
        Process incoming server message.
        
        Routes messages to appropriate handlers based on type.
        
        Args:
            data: Parsed JSON message from server
        """
        msg_type = data.get("type", "")
        
        if msg_type == "joined":
            # We've been assigned to a room
            self.player_id = data.get("player_id")
            self.room_id = data.get("room_id")
            self.log(f"Joined room {self.room_id} as player {self.player_id}")
            # Send ready signal
            await self.send_ready()
        
        elif msg_type == "waiting":
            # Waiting for opponent
            self.log("Waiting for opponent...")
        
        elif msg_type == "start":
            # Game is starting
            mode = data.get("mode", "two_player")
            self.log(f"Game starting! Mode: {mode}")
            self.game_running = True
            try:
                self.strategy.reset_game_state()
            except Exception:
                logger.exception("Error resetting strategy state")
        
        elif msg_type == "state":
            # Game tick - make a move
            game = data.get("game")
            if not isinstance(game, dict):
                self.log("Received 'state' message with invalid game payload", "warning")
                return
            await self.handle_game_state(game)
        
        elif msg_type == "gameover":
            # Round ended
            winner = data.get("winner")
            wins = data.get("wins", {})
            if self.player_id is not None and isinstance(wins, dict):
                self.current_wins = wins.get(str(self.player_id), 0)
            else:
                self.log("Cannot read wins: player_id not set or wins invalid", "warning")
            self.points_to_win = data.get("points_to_win", 3)
            
            we_won = winner == self.player_id
            try:
                self.strategy.record_game_result(we_won)
            except Exception:
                logger.exception("Error recording game result")
            
            result = "WON" if we_won else "LOST"
            self.log(f"Round {result}! Score: {self.current_wins}/{self.points_to_win}")
            
            self.game_running = False
            # Send ready for next round
            await self.send_ready()
        
        elif msg_type == "match_complete":
            # Match ended
            winner = data.get("winner")
            final_score = data.get("final_score", {})
            
            if winner == self.player_id:
                self.log("MATCH WON!")
            else:
                self.log("Match lost.")
            
            self.log(f"Final score: {final_score}")
            self.game_running = False
        
        elif msg_type == "match_assigned":
            # New match in tournament
            self.room_id = data.get("room_id")
            self.player_id = data.get("player_id")
            opponent = data.get("opponent", "Unknown")
            self.log(f"New match assigned! vs {opponent} in room {self.room_id}")
            self.current_wins = 0
            await self.send_ready()
        
        elif msg_type == "competition_complete":
            # Tournament ended
            champion = data.get("champion", "Unknown")
            if champion == self.name:
                self.log("TOURNAMENT CHAMPION!")
            else:
                self.log(f"Tournament complete. Champion: {champion}")
            self.shutdown_requested = True
        
        elif msg_type == "error":
            # Server error
            error_msg = data.get("message", "Unknown error")
            self.log(f"Server error: {error_msg}", "error")
        
        elif msg_type == "ready_required":
            # Server requires ready signal
            self.log("Server requires ready signal")
            await self.send_ready()
        
        else:
            # Unknown message type - log for debugging
            if msg_type:
                self.log(f"Unknown message type: {msg_type}", "debug")
    
    async def handle_game_state(self, game: dict[str, Any]) -> None:
        """
        Process game state and make move decision.
        
        This is called every game tick. It:
        1. Updates internal state
        2. Calls strategy engine
        3. Sends move command
        
        Args:
            game: Game state object from server
        """
        if not game.get("running", False):
            return

        if self.player_id is None:
            self.log("Received game state but player_id is not set, skipping", "warning")
            return
        
        self.game_state = game
        
        # Update grid dimensions
        grid = game.get("grid", {})
        if isinstance(grid, dict):
            self.grid_width = grid.get("width", 30)
            self.grid_height = grid.get("height", 20)
        try:
            self.strategy.update_grid_size(self.grid_width, self.grid_height)
        except Exception:
            logger.exception("Error updating grid size")
        
        # Verify we're still alive
        snakes = game.get("snakes")
        if not isinstance(snakes, dict):
            self.log("Invalid snakes payload in game state", "warning")
            return

        my_snake = snakes.get(str(self.player_id))
        if not my_snake or not my_snake.get("alive", True):
            return
        
        # Calculate best move with fallback on strategy failure
        direction = self._safe_calculate_move(game, self.player_id, my_snake)
        
        if direction:
            await self.send_move(direction)
        else:
            # Fallback - keep current direction
            current_dir = my_snake.get("direction", "right")
            self.log(f"No valid move found, maintaining {current_dir}", "warning")
            await self.send_move(current_dir)

    def _safe_calculate_move(
        self,
        game: dict[str, Any],
        player_id: int,
        my_snake: dict[str, Any],
    ) -> Optional[str]:
        """Call strategy engine with fallback on unexpected errors."""
        try:
            return self.strategy.calculate_move(game, player_id)
        except Exception:
            logger.exception("Strategy engine error; falling back to safe move")
            # Lightweight fallback: pick the first non-reversing, in-bounds,
            # non-body direction so the bot doesn't crash.
            return self._fallback_direction(my_snake)

    def _fallback_direction(self, my_snake: dict[str, Any]) -> Optional[str]:
        """Best-effort safe direction when the strategy engine fails."""
        body = my_snake.get("body", [])
        if not body:
            return None
        head = (body[0][0], body[0][1])
        current_dir = my_snake.get("direction", "right")
        opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
        body_set = {(s[0], s[1]) for s in body}
        for d in ("up", "down", "left", "right"):
            if d == opposites.get(current_dir):
                continue
            dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
            nx, ny = head[0] + dx, head[1] + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and (nx, ny) not in body_set:
                return d
        return current_dir
    
    async def play(self) -> None:
        """
        Main game loop.
        
        Continuously receives and processes messages until
        disconnected or shutdown requested.
        """
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
        
        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.ConnectionClosedOK,
            websockets.exceptions.ConnectionClosedError,
        ) as e:
            self.log(f"Connection closed: {e}", "warning")
        except Exception:
            logger.exception("Unexpected game loop error")
        
        self.log("Game loop ended")
    
    async def run(self) -> None:
        """
        Main entry point - connect and play.
        
        Handles connection, reconnection, and graceful shutdown.
        Uses exponential backoff with jitter for reconnection.
        """
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
                delay += random.uniform(0, delay * 0.25)  # jitter
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CopperHead Tournament Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bot.py --server ws://localhost:8765/ws/ --name "MyBot"
  python bot.py -s wss://server.example.com/ws/ -n "Champion" -d 10
  
Environment variables:
  COPPERHEAD_SERVER_URL  Server WebSocket URL
  BOT_NAME               Bot display name
  BOT_DIFFICULTY         Difficulty level (1-10)
  BOT_QUIET              Suppress output (true/false)
"""
    )
    
    parser.add_argument(
        "--server", "-s",
        default=os.environ.get("COPPERHEAD_SERVER_URL", "ws://localhost:8765/ws/"),
        help="Server WebSocket URL"
    )
    parser.add_argument(
        "--name", "-n",
        default=os.environ.get("BOT_NAME", "CopperheadChampion"),
        help="Bot display name"
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
        help="Difficulty level 1-10 (10 = optimal play)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=os.environ.get("BOT_QUIET", "").lower() == "true",
        help="Suppress console output"
    )
    
    return parser.parse_args()


async def main() -> None:
    """Main async entry point."""
    args = parse_args()
    
    # Validate difficulty
    difficulty = max(1, min(10, args.difficulty))
    
    # Create bot
    bot = CopperheadBot(
        server_url=args.server,
        name=args.name,
        difficulty=difficulty,
        quiet=args.quiet
    )
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        bot.shutdown_requested = True
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: signal_handler())
    
    # Run bot
    logger.info(f"CopperHead Bot starting: {bot.name}")
    logger.info(f"Server: {bot.server_url}")
    logger.info(f"Difficulty: {difficulty}/10")
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await bot.disconnect()
    
    logger.info("Bot shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
