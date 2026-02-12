# Architecture

## System Overview

APEX is a modular algorithmic trading platform with four main subsystems:

```
                         ┌─────────────────────┐
                         │   Next.js Frontend   │
                         │   (Dashboard, WS)    │
                         └──────────┬───────────┘
                                    │ WebSocket + REST
                         ┌──────────▼───────────┐
                         │   FastAPI API Server  │
                         │   (api/server.py)     │
                         └──────────┬───────────┘
                          ┌─────────┼─────────┐
                          │         │         │
                ┌─────────▼──┐ ┌────▼────┐ ┌──▼──────────┐
                │  ML Engine │ │  Risk   │ │  Execution  │
                │  (models/) │ │ (risk/) │ │ (execution/)│
                └────────────┘ └─────────┘ └─────────────┘
                                                │
                                    ┌───────────▼──────────┐
                                    │  IBKR TWS / Gateway  │
                                    └──────────────────────┘
```

---

## Module Map

### ML Engine (`models/`)

| Component | File | Purpose |
|-----------|------|---------|
| Signal Generator | `institutional_signal_generator.py` | 8-model ensemble, regime-specific training |
| Regime Detector | `adaptive_regime_detector.py` | EMA-smoothed probability regime detection |
| Feature Engineering | `feature_engineering.py` | 67 features across 10 categories |
| ML Methods | `ml_methods/` | ElasticNet, BayesianRidge, SVR, CatBoost, etc. |

**Data flow**: Raw prices -> Feature extraction (67 features) -> Regime detection -> Regime-specific model ensemble -> Weighted signal [-1, +1]

**Training**: Walk-forward with PurgedTimeSeriesSplit. Separate model per regime (bull/bear/neutral/volatile). Drift monitoring triggers automatic retraining.

### Risk Engine (`risk/`)

16+ independent risk modules that form a layered defense:

| Module | Trigger | Action |
|--------|---------|--------|
| BlackSwanGuard | SPY velocity, VIX spike | 4-tier escalation, emergency liquidation |
| DrawdownCascadeBreaker | Portfolio drawdown | 5-tier position reduction |
| CorrelationCascadeBreaker | Portfolio herding | Block correlated entries |
| ProfitRatchet | Position gains | Progressive trailing stops |
| LiquidityGuard | Spread widening | Reduce order sizes |
| OvernightRiskGuard | Market close | Reduce overnight exposure |
| MacroEventShield | Economic calendar | Blackout periods |

### Execution Engine (`execution/`)

| Component | Purpose |
|-----------|---------|
| `ibkr_connector.py` | IBKR TWS API connection, reconnection, throttling |
| `smart_order_router.py` | Algo selection (VWAP, TWAP, Iceberg, POV) |
| `options_trader.py` | Black-Scholes pricing, Greeks, hedging |
| `transaction_cost_optimizer.py` | Slippage modeling, cost analysis |

### API Layer (`api/`)

| Component | Purpose |
|-----------|---------|
| `server.py` | FastAPI app, REST endpoints, WebSocket streaming |
| `auth.py` | JWT + API key auth, rate limiting, token blacklist |
| `health.py` | Health check endpoints |

### Frontend (`frontend/`)

Next.js 14 + React dashboard:
- Real-time P&L and position monitoring via WebSocket
- Sector exposure charts
- Authentication with JWT tokens
- Responsive design

---

## Configuration

All configuration is centralized in `config.py` -> `ApexConfig` class. Every parameter can be overridden via environment variables prefixed with `APEX_`.

Key configuration groups:
- **Trading**: Capital, position sizing, signal thresholds
- **Risk**: Circuit breakers, drawdown tiers, VIX thresholds
- **Execution**: IBKR connection, order routing parameters
- **ML**: Regime detection, walk-forward splits, drift thresholds

---

## Data Flow

```
1. Price Data (yfinance / IBKR)
       │
2. Feature Engineering (67 features)
       │
3. Regime Detection (bull/bear/neutral/volatile)
       │
4. ML Signal Generation (8-model ensemble per regime)
       │
5. Signal Consensus Engine (multi-generator agreement)
       │
6. Risk Filter Chain (16+ independent checks)
       │
7. Position Sizing (Kelly + ATR-based)
       │
8. Order Execution (IBKR: VWAP/TWAP/Market)
       │
9. State Update -> WebSocket -> Dashboard
```

---

## Authentication

Two auth paths (automatic fallback):

1. **PostgreSQL-backed** (SaaS): `services/auth/` with bcrypt passwords, Stripe billing, MFA
2. **In-memory** (standalone): `api/auth.py` with JWT tokens and API keys

The system automatically falls back to in-memory auth when PostgreSQL is unavailable.

---

## Testing

```bash
python -m pytest tests/ -v --tb=short
```

Test categories:
- `test_api_auth_health.py` — Auth and health endpoints
- `test_signal_fortress.py` — Risk module chain
- `test_websocket.py` — WebSocket and REST integration
- `test_backtest_validator.py` — Backtest validation service
- `test_execution_simulator.py` — Execution simulation
- `benchmarks/` — Performance benchmarks
