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
import signal
from typing import Any, Optional

try:
    import websockets
    import websockets.exceptions
    from websockets.client import WebSocketClientProtocol
except ImportError:
    print("ERROR: websockets package not installed. Run: pip install websockets")
    sys.exit(1)

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
logger = logging.getLogger("CopperheadBot")


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
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to server.
        
        Returns:
            True if connection successful
        """
        url = self.server_url.rstrip("/") + "/join"
        self.log(f"Connecting to {url}...")
        try:
            self.ws = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10
            )
            self.connected = True
            self.log("Connected successfully!")
            return True
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
            self.strategy.reset_game_state()
        
        elif msg_type == "state":
            # Game tick - make a move
            game = data.get("game", {})
            await self.handle_game_state(game)
        
        elif msg_type == "gameover":
            # Round ended
            winner = data.get("winner")
            self.current_wins = data.get("wins", {}).get(str(self.player_id), 0)
            self.points_to_win = data.get("points_to_win", 3)
            
            we_won = winner == self.player_id
            self.strategy.record_game_result(we_won)
            
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
        
        self.game_state = game
        
        # Update grid dimensions
        grid = game.get("grid", {})
        self.grid_width = grid.get("width", 30)
        self.grid_height = grid.get("height", 20)
        self.strategy.update_grid_size(self.grid_width, self.grid_height)
        
        # Verify we're still alive
        snakes = game.get("snakes", {})
        my_snake = snakes.get(str(self.player_id))
        if not my_snake or not my_snake.get("alive", True):
            return
        
        # Calculate best move
        direction = self.strategy.calculate_move(game, self.player_id)
        
        if direction:
            await self.send_move(direction)
        else:
            # Fallback - keep current direction
            current_dir = my_snake.get("direction", "right")
            self.log(f"No valid move found, maintaining {current_dir}", "warning")
            await self.send_move(current_dir)
    
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
        """
        while not self.shutdown_requested:
            if await self.connect():
                await self.play()
                await self.disconnect()
            
            if not self.shutdown_requested:
                self.log("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)


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


async def main():
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
