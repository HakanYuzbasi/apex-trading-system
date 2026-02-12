# APEX Trading System

**Institutional-Grade Algorithmic Trading Platform with Regime-Adaptive ML, Multi-Asset Execution, and Enterprise Risk Management**

---

## Overview

APEX is a fully automated trading system that combines advanced machine learning with institutional-grade risk management to generate alpha across equities, FX, and crypto markets. The system adapts to market regimes in real-time, scaling positions and strategies based on current conditions.

### Key Metrics

| Metric | Value |
|--------|-------|
| **ML Features** | 67 engineered features across 10 categories |
| **Market Regimes** | 4 (Bull, Bear, Neutral, Volatile) |
| **Directional Accuracy** | 54-58% (regime-weighted) |
| **Risk Modules** | 16+ independent safety systems |
| **ML Models** | 8-model ensemble (RF, GBM, XGBoost, LightGBM, ElasticNet, BayesianRidge, SVR, CatBoost) |
| **Asset Classes** | Equities, FX, Crypto |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    APEX Trading System                    │
├──────────────┬───────────────┬───────────────────────────┤
│  ML Engine   │  Risk Engine  │     Execution Engine      │
│              │               │                           │
│ 67 Features  │ BlackSwanGuard│ IBKR Connector            │
│ 4 Regimes    │ DrawdownBreak │ Smart Order Routing       │
│ 8 Models     │ CorrelCascade │ VWAP/TWAP/Iceberg         │
│ Walk-Forward │ ProfitRatchet │ Slippage Optimization     │
│ Drift Monitor│ LiquidityGuard│ Options Trading           │
├──────────────┴───────────────┴───────────────────────────┤
│                    API Layer (FastAPI)                     │
│  REST + WebSocket  │  JWT/API Key Auth  │  Health Checks  │
├──────────────────────────────────────────────────────────┤
│                 Frontend (Next.js + React)                 │
│  Live Dashboard  │  P&L Tracking  │  Position Monitor     │
└──────────────────────────────────────────────────────────┘
```

---

## Features

### ML Signal Generation
- **8-model ensemble** with regime-specific weighting and automatic retraining
- **67 engineered features** across volatility dynamics, microstructure, cross-asset, momentum, and more
- **Walk-forward validation** with PurgedTimeSeriesSplit to prevent look-ahead bias
- **Adaptive regime detection** with EMA-smoothed probability estimates
- **Drift monitoring** with automatic retrain triggers when accuracy degrades

### Risk Management (16+ modules)
- **BlackSwanGuard** — Flash crash detection with 4 escalating threat levels
- **CorrelationCascadeBreaker** — Detects risk-off herding across portfolio
- **DrawdownCascadeBreaker** — Tiered position reduction on equity drawdowns
- **ProfitRatchet** — Locks in gains with progressive stop tightening
- **LiquidityGuard** — Monitors market depth and adjusts order sizing
- **OvernightRiskGuard** — Manages gap risk across market sessions
- **MacroEventShield** — Reduces exposure ahead of macro releases
- **VIX Regime Manager** — Adapts strategy to implied volatility environment

### Execution
- **IBKR integration** with automatic reconnection and request throttling
- **Smart Order Routing** with multi-venue price optimization
- **Advanced algorithms** — VWAP, TWAP, Iceberg, POV
- **Options trading** with Black-Scholes pricing and Greeks monitoring
- **Transaction cost analysis** and slippage modeling

### Infrastructure
- **FastAPI** backend with WebSocket live streaming
- **Next.js/React** dashboard with real-time P&L and position monitoring
- **PostgreSQL + Redis** for persistence and caching
- **Docker Compose** for one-command deployment
- **Alembic** database migrations
- **JWT + API key** authentication with rate limiting

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+ (for frontend)
- Docker & Docker Compose (for full stack)
- Interactive Brokers TWS or Gateway (for live trading)

### Local Development

```bash
# Clone and set up
git clone <repository-url>
cd apex-trading

# Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run trading engine
python main.py

# Run API server (separate terminal)
uvicorn api.server:app --reload --port 8000

# Run frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### Docker Deployment

```bash
# Set required environment variables
export POSTGRES_PASSWORD=<your-secure-password>
export APEX_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Launch full stack
docker compose up -d

# Check health
curl http://localhost:8000/health/
```

---

## Configuration

All settings configurable via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `APEX_LIVE_TRADING` | Enable live order execution | `false` |
| `APEX_INITIAL_CAPITAL` | Starting capital | `100000` |
| `APEX_MAX_POSITIONS` | Maximum concurrent positions | `40` |
| `APEX_IBKR_HOST` | IBKR TWS/Gateway host | `127.0.0.1` |
| `APEX_IBKR_PORT` | IBKR port (7497=paper, 7496=live) | `7497` |
| `APEX_AUTH_ENABLED` | Enable API authentication | `true` |
| `APEX_SECRET_KEY` | JWT signing key | *required* |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_api_auth_health.py -v    # Auth tests
python -m pytest tests/test_signal_fortress.py -v     # Risk system tests
python -m pytest tests/benchmarks/ -v                 # Performance benchmarks
```

---

## Project Structure

```
apex-trading/
├── api/              # FastAPI server, auth, health checks, middleware
├── models/           # ML signal generators, regime detection, model registry
├── risk/             # 16+ risk management modules
├── execution/        # IBKR connector, smart order routing, options
├── core/             # Trading engine core, symbols, logging
├── data/             # Data fetching, feature store, sentiment
├── portfolio/        # Portfolio optimization and rebalancing
├── backtesting/      # Backtesting engine with lookahead prevention
├── monitoring/       # Signal decay, compliance, data watchdog
├── services/         # SaaS features (auth, billing, validators)
├── frontend/         # Next.js React dashboard
├── tests/            # Test suite (400+ tests)
├── migrations/       # Alembic database migrations
├── config.py         # Central configuration
├── main.py           # Trading engine entry point
└── docker-compose.yml
```

---

## Documentation

- [Enhanced Features](ENHANCED_FEATURES.md) — Detailed ML feature engineering documentation
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) — Performance improvements and financial impact
- [Features Reference](FEATURES_REFERENCE.md) — Quick feature cheat sheet with signal combinations
- [Model Improvements](docs/MODEL_IMPROVEMENTS_V2.md) — Regime-specific hyperparameter tuning
- [Security Policy](SECURITY.md) — Vulnerability reporting and security standards

---

## License

See [LICENSE](LICENSE) for details.
