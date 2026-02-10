# Contributing to CopperHead Bot

Thank you for your interest in contributing to this CopperHead tournament bot! We welcome contributions from the community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/copperhead-bot.git
   cd copperhead-bot
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   # or: source .venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose

### Project Structure

```
copperhead-bot/
├── bot.py          # Entry point, WebSocket handling
├── strategy.py     # AI decision engine
├── pathfinding.py  # Algorithms (A*, BFS, flood-fill, Voronoi)
├── utils.py        # Grid utilities and helpers
├── requirements.txt
└── README.md
```

### Testing Your Changes

1. **Syntax check**:
   ```bash
   python -m py_compile bot.py strategy.py pathfinding.py utils.py
   ```

2. **Test against a CopperHead server**:
   ```bash
   # Start the CopperHead server (see link below)
   python bot.py --server ws://localhost:8765/ws/ --name "TestBot"
   ```

3. **Test against the default CopperBot**:
   ```bash
   curl -X POST "http://localhost:8765/add_bot?difficulty=5"
   python bot.py --server ws://localhost:8765/ws/
   ```

## Areas for Contribution

### Strategy Improvements

- Enhanced opponent prediction algorithms
- Machine learning integration
- Multi-food path planning
- Buff-aware tactics

### Code Quality

- Additional unit tests
- Performance optimizations
- Documentation improvements
- Bug fixes

### New Features

- Replay analysis tools
- Strategy visualization
- Configuration profiles for different playstyles

## Submitting Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit with clear messages:
   ```bash
   git commit -m "Add feature: description of change"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Any related issue numbers
   - Test results or screenshots if applicable

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and improve

## Resources

- [CopperHead Server](https://github.com/revodavid/copperhead-server) - The game server
- [Building Your Own Bot](https://github.com/revodavid/copperhead-server/blob/main/Building-Your-Own-Bot.md) - Official bot guide
- [Game Rules](https://github.com/revodavid/copperhead-server/blob/main/game-rules.md) - Complete game rules
- [Tournament Guide](https://github.com/revodavid/copperhead-server/blob/main/How-To-Host-A-Bot-Hack-Tournament.md) - Tournament hosting

## Questions?

Open an issue on GitHub or refer to the CopperHead server documentation.

Happy hacking! 🐍
