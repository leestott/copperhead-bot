# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this CopperHead bot, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include detailed information about the vulnerability:
   - Description of the issue
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- Acknowledgment of your report within 48 hours
- Regular updates on the progress of addressing the vulnerability
- Credit in the security advisory (if desired) once the issue is resolved

## Security Considerations

This bot is designed for tournament play and connects to CopperHead game servers via WebSocket. Key security considerations:

### Network Security

- Only connect to trusted CopperHead servers
- Use `wss://` (WebSocket Secure) when available
- The bot does not store or transmit sensitive data

### Environment Variables

- Store server URLs and configuration in `.env` files
- Never commit `.env` files to version control (included in `.gitignore`)
- Use environment variables for any sensitive configuration

### Dependencies

- Keep dependencies updated (`pip install --upgrade -r requirements.txt`)
- Only two dependencies: `websockets` and `python-dotenv`
- Both are well-maintained, widely-used packages

## Related Resources

- [CopperHead Server Repository](https://github.com/revodavid/copperhead-server)
- [Building Your Own Bot Guide](https://github.com/revodavid/copperhead-server/blob/main/Building-Your-Own-Bot.md)
