# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.x     | :white_check_mark: |
| 1.x     | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Apex Trading System, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers directly
3. Include a detailed description of the vulnerability
4. Provide steps to reproduce the issue if possible
5. Allow up to 72 hours for an initial response

### What to Expect

- **Acknowledgment** within 72 hours of your report
- **Assessment** of severity and impact within 1 week
- **Fix timeline** communicated based on severity:
  - Critical: Patch within 48 hours
  - High: Patch within 1 week
  - Medium: Patch within 2 weeks
  - Low: Included in next release

## Security Standards

### Authentication
- JWT tokens with configurable expiration (default: 60 minutes)
- Refresh token rotation (7-day expiry)
- API key authentication for service-to-service communication
- bcrypt password hashing with automatic salt generation
- Rate limiting on authentication endpoints

### Data Protection
- All sensitive configuration via environment variables
- No secrets committed to version control
- `.gitignore` covers credentials, API keys, and `.env` files
- Database credentials isolated per environment

### Network Security
- CORS restricted to configured origins
- HTTPS enforced in production
- WebSocket connections authenticated via bearer tokens
- Rate limiting to prevent abuse and DDoS

### Infrastructure
- Docker containers run as non-root users (production)
- PostgreSQL with parameterized queries (SQLAlchemy ORM)
- Redis for session management with configurable TTL
- Health check endpoints for monitoring

### Compliance
- Pre-trade compliance checks
- Immutable audit trail logging
- Position limit enforcement
- Sector exposure monitoring

## Security Checklist for Deployment

- [ ] Set strong `APEX_SECRET_KEY` (min 64 characters)
- [ ] Set unique `POSTGRES_PASSWORD` (not the default)
- [ ] Set `APEX_ADMIN_API_KEY` to a secure random value
- [ ] Enable HTTPS/TLS termination at load balancer
- [ ] Configure CORS `allowed_origins` for your domain
- [ ] Enable `APEX_AUTH_ENABLED=true`
- [ ] Review and restrict API rate limits
- [ ] Set up log aggregation (do not log to local disk in production)
- [ ] Enable database connection encryption (SSL mode)
- [ ] Configure firewall rules for PostgreSQL and Redis ports
