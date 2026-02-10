# 🐍 CopperHead Tournament Bot

A high-performance, competitive Snake AI bot designed for CopperHead Bot Hack Tournaments.

## Features

- **Multi-Layer Strategy Engine**: Safety-first survival, A* pathfinding, area control, and adaptive aggression
- **Tournament-Ready**: Handles multiple consecutive games with clean state resets
- **Protocol Compliant**: Full WebSocket protocol compliance with proper message handling
- **Dynamic Awareness**: Reads grid dimensions from server, tracks all game objects
- **Deterministic Behavior**: Predictable, explainable decision-making

## Quick Start

### Prerequisites

- Python 3.10+
- A running CopperHead server

### Installation

```bash
# Clone this repository
git clone https://github.com/leestott/copperhead-bot.git
cd copperhead-bot

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the environment template and configure:

```bash
cp .env.example .env
```

Edit `.env` with your server URL:

```
COPPERHEAD_SERVER_URL=ws://localhost:8765/ws/
BOT_NAME=CopperheadChampion
```

### Running the Bot

```bash
# Using environment variables (recommended)
python bot.py

# Or with command-line arguments
python bot.py --server ws://localhost:8765/ws/ --name "MyBot"

# Against a specific difficulty opponent
python bot.py --server ws://localhost:8765/ws/ --difficulty 5
```

## Architecture

```
copperhead-bot/
├── bot.py          # Main entry point, WebSocket connection, game loop
├── strategy.py     # Multi-layer decision engine
├── pathfinding.py  # A*, BFS, flood-fill algorithms
├── utils.py        # Grid helpers, coordinate utilities
├── requirements.txt
├── .env.example
└── README.md
```

## Strategy Overview

### 1. Safety-First Survival
- Never collide with walls, self, or opponent (unless intentional)
- Maintain fallback safe moves at all times
- Validate all moves before execution

### 2. Short-Horizon Pathfinding
- A* pathfinding to food with safety constraints
- BFS for distance calculations
- Penalty for paths that reduce future mobility

### 3. Area Control & Space Maximization
- Flood-fill to count accessible cells
- Prefer moves that maximize reachable space
- Avoid corridors and dead-ends

### 4. Adaptive Aggression (Two-Player Mode)
- **When Ahead**: Play defensively, deny space to opponent
- **When Behind**: Apply pressure, cut off escape routes, force risky opponent moves

### 5. Tournament Awareness
- Clean state reset between games
- No persistent assumptions across rounds
- Robust reconnection handling

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COPPERHEAD_SERVER_URL` | WebSocket URL of the CopperHead server | `ws://localhost:8765/ws/` |
| `BOT_NAME` | Display name for the bot | `CopperheadChampion` |
| `BOT_DIFFICULTY` | Internal difficulty setting (1-10) | `10` |
| `BOT_QUIET` | Suppress console output | `false` |

## Command-Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--server` | `-s` | Server WebSocket URL |
| `--name` | `-n` | Bot display name |
| `--difficulty` | `-d` | Difficulty level (1-10) |
| `--quiet` | `-q` | Suppress console output |

## Game Protocol

### Messages Received

| Type | Description |
|------|-------------|
| `joined` | Player assigned to arena with `player_id` and `room_id` |
| `waiting` | Waiting for opponent to join |
| `start` | Game beginning with `mode` and `room_id` |
| `state` | Game tick with full `game` state object |
| `gameover` | Round ended with `winner`, `wins`, `points_to_win` |
| `match_complete` | Match finished with `winner` and `final_score` |
| `match_assigned` | New tournament round with `room_id`, `player_id`, `opponent` |
| `competition_complete` | Tournament ended with `champion` |
| `error` | Server error with `message` description |

### Messages Sent

```json
{"action": "ready", "mode": "two_player", "name": "BotName"}
{"action": "move", "direction": "up|down|left|right"}
```

## Performance Tuning

The strategy engine uses weighted scoring for move selection:

- **Safety**: Immediate collision avoidance (critical)
- **Food Distance**: Weighted by snake length comparison
- **Reachable Space**: Flood-fill cell count
- **Opponent Pressure**: Distance to opponent head
- **Corridor Avoidance**: Penalize moves with few neighbors

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

Key areas for enhancement:

- Enhanced opponent prediction algorithms
- Multi-food path planning
- Buff-aware strategies
- Machine learning integration

## Security

See [SECURITY.md](SECURITY.md) for security policy and vulnerability reporting.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Resources

- [CopperHead Server](https://github.com/revodavid/copperhead-server) - The game server
- [Building Your Own Bot](https://github.com/revodavid/copperhead-server/blob/main/Building-Your-Own-Bot.md) - Official bot guide
- [Game Rules](https://github.com/revodavid/copperhead-server/blob/main/game-rules.md) - Complete game rules
- [Tournament Guide](https://github.com/revodavid/copperhead-server/blob/main/How-To-Host-A-Bot-Hack-Tournament.md) - Tournament hosting

## Acknowledgments

Built for the [CopperHead Bot Hack Tournament](https://github.com/revodavid/copperhead-server)
