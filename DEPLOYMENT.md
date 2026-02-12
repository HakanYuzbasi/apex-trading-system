# Deployment Guide

## Prerequisites

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| Node.js | 20+ (frontend) |
| Docker & Compose | Latest |
| IBKR TWS/Gateway | Latest (live trading only) |

---

## Local Development

```bash
# 1. Clone and create virtualenv
git clone <repo-url> && cd apex-trading
python -m venv venv && source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — at minimum set APEX_SECRET_KEY and APEX_ADMIN_API_KEY

# 4. Run the trading engine
python main.py

# 5. Run the API server (separate terminal)
uvicorn api.server:app --reload --port 8000

# 6. Run the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

---

## Docker Compose (Full Stack)

```bash
# Required environment variables
export POSTGRES_PASSWORD=$(openssl rand -hex 16)
export APEX_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export APEX_ADMIN_API_KEY=$(python -c "import secrets; print(secrets.token_hex(16))")

# Launch all services
docker compose up -d

# Verify
curl http://localhost:8000/health/
curl http://localhost:3000  # Frontend
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| `apex` | host network | Trading engine (connects to IBKR) |
| `apex-api` | 8000 | FastAPI REST + WebSocket |
| `frontend` | 3000 | Next.js dashboard |
| `postgres` | 5432 | PostgreSQL 16 |
| `redis` | 6379 | Redis 7 (caching) |

### Scaling Notes

- The trading engine (`apex`) runs as a single process — it is stateful and must not be replicated.
- The API server (`apex-api`) is stateless and can be scaled behind a load balancer.
- The frontend is a static Next.js build served by Node.

---

## Environment Variables

See [.env.example](.env.example) for the full list. Critical variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `APEX_SECRET_KEY` | Yes | JWT signing key (min 32 hex chars) |
| `APEX_ADMIN_API_KEY` | Recommended | Persistent admin API key |
| `POSTGRES_PASSWORD` | Docker only | PostgreSQL password |
| `APEX_AUTH_ENABLED` | No | Enable auth (default: `true`) |
| `APEX_LIVE_TRADING` | No | Enable live execution (default: `true`) |
| `APEX_IBKR_PORT` | No | 7497 = paper, 7496 = live |

---

## Database Migrations

```bash
# Run Alembic migrations (requires DATABASE_URL)
alembic upgrade head
```

---

## Health Checks

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health/` | None | Basic liveness (from health router) |
| `GET /health` | API key | Authenticated staleness check |
| `GET /status` | API key | Full trading status |

---

## Security Checklist

- [ ] `APEX_SECRET_KEY` is set to a unique, random value
- [ ] `APEX_ADMIN_API_KEY` is set (not auto-generated)
- [ ] `POSTGRES_PASSWORD` uses a strong password
- [ ] IBKR port is `7497` (paper) for testing, `7496` (live) for production
- [ ] CORS origins are restricted to your domain
- [ ] TLS termination is configured (nginx/Cloudflare) in front of the API
- [ ] Rate limiting is active on `/auth/*` endpoints
- [ ] `bcrypt` is installed for password hashing

---

## Monitoring

- **Logs**: Written to `logs/apex.log` with rotation (5 MB, 5 backups)
- **State file**: `data/trading_state.json` updated every cycle
- **WebSocket**: Real-time state push to connected dashboards
- **Heartbeat**: `data/heartbeat.json` for external monitoring
