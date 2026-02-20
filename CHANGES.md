# CHANGES

## Item 1 - Fail-safe trading defaults
- Updated `.env.example`:
  - Set `APEX_LIVE_TRADING=false`
  - Added `APEX_LIVE_TRADING_CONFIRMED=false`
  - Added clear two-flag warning comments.
- Updated `config.py`:
  - Changed fallback for `LIVE_TRADING` to `false`
  - Added `LIVE_TRADING_CONFIRMED`
  - Added `assert_live_trading_confirmation()`
  - Added hard validation error when live trading is enabled without confirmation.
- Updated `main.py`:
  - Startup guard now calls `assert_live_trading_confirmation()`.

## Item 2 - Harden auth for production
- Updated `api/auth.py`:
  - Replaced file-backed/in-memory user lookup with `DatabaseUserStore` using async DB calls.
  - Replaced in-memory token blacklist with Redis-backed `TokenBlacklist`.
  - Added strict runtime guard: Redis required in `staging|production|prod`.
  - Added mock-token gating: mock token create/verify now raises `RuntimeError` outside development.
  - Added `verify_auth_runtime_prerequisites()`.
- Updated `services/auth/service.py`:
  - Mock token create/decode is now development-only.
- Updated `services/auth/middleware.py`:
  - Removed legacy in-memory fallback path.
- Updated `services/auth/router.py`:
  - Removed legacy in-memory fallback flows for register/login/refresh/me/subscription.
  - DB unavailable now returns 503.
- Updated `api/server.py`:
  - Startup now calls `verify_auth_runtime_prerequisites()`.
- Added tests:
  - `tests/test_mock_token_gating.py` for non-development mock-path denial.

## Item 3 - Tighten CI quality gates
- Updated `pytest.ini`:
  - Coverage threshold raised to 45.
  - Added note for next sprint 60% target.
- Updated `.github/workflows/ci.yml`:
  - Coverage gate raised to 45.
  - Removed Ruff `--exit-zero`.
  - Removed TS `|| true` bypass.
  - Added `pip check` step(s).
  - Added Python import smoke test step.
  - Added smoke execution of `scripts/check_secrets.py` logic.
  - Added exception policy check step.

## Item 4 - Real observability and alerting
- Replaced `monitoring/alert_aggregator.py` TODO stubs with real delivery logic:
  - Slack webhook delivery (`APEX_SLACK_WEBHOOK_URL`).
  - PagerDuty Events v2 delivery (`APEX_PAGERDUTY_ROUTING_KEY`) for critical alerts.
  - Severity policy implemented:
    - INFO -> Slack
    - WARNING -> Slack + warning log
    - CRITICAL -> Slack + PagerDuty + structured critical log
  - Delivery failures log and re-raise.
- Added `tests/test_monitoring_alert_aggregator_channels.py`:
  - Verifies routing and failure behavior.

## Item 5 - Remove single-tenant assumptions in API streaming
- Updated `api/ws_manager.py`:
  - Added tenant-aware connection tracking.
  - Added `broadcast_to_tenant()` and tenant listing.
- Updated `api/server.py` streaming loop:
  - Removed direct `data/users` iteration.
  - Uses service-layer tenant queries (`broker_service.list_tenant_ids()`).
  - Uses `asyncio.gather(..., return_exceptions=True)` for isolated tenant fetch failures.
  - Broadcasts tenant-scoped equity updates.
- Updated websocket endpoint:
  - Attaches tenant context from authenticated user.
  - Sends tenant-scoped initial equity payload.
- Updated `api/routers/public.py` for new manager connect signature.
- Added `tests/test_ws_tenant_isolation.py`.

## Item 6 - IBKR broker parity
- Updated `services/broker/service.py`:
  - Added normalized `BrokerEquitySnapshot` payload.
  - Implemented IBKR equity retrieval (`NetLiquidation`) with blocking adapter wrapped async-safely.
  - Implemented per-connection cache with stale fallback behavior.
  - Added tenant-level equity aggregation with per-source breakdown.
- Updated `services/broker/router.py`:
  - `/portfolio/balance` now uses tenant equity snapshot service and returns normalized breakdown rows including `value`, `broker`, `stale`, `as_of`.
- Added `tests/test_ibkr_equity_pooling.py`:
  - Covers happy path, connection failure, stale fallback, and no-cache failure.
- Added `tests/test_portfolio_balance_shape.py`:
  - Verifies dashboard equity response shape.

## Item 7 - Break down oversized modules
- Created `core/execution_loop.py` and moved trading runtime logic there.
- Replaced `main.py` with startup orchestrator (~100 lines) that delegates to `core.execution_loop`.
- Added extracted support modules with typed interfaces/docstrings:
  - `core/risk_orchestration.py`
  - `core/state_sync.py`
  - `core/broker_dispatch.py`
- Integrated extracted helpers into `core/execution_loop.py` initialization/state export flow.
- Frontend decomposition:
  - Added `frontend/components/dashboard/AlertsFeed.tsx`
  - Added `frontend/components/dashboard/EquityPanel.tsx`
  - Added `frontend/components/dashboard/PositionsTable.tsx`
  - Added `frontend/components/dashboard/ControlsPanel.tsx`
  - Updated `frontend/components/Dashboard.tsx` to use these typed slice components.

## Item 8 - Fix dependency/runtime drift
- Updated `requirements.txt`:
  - Added `aiosqlite` and additional missing runtime dependencies used in codepaths.
- Updated `core/database.py`:
  - Added explicit `db_path must not be None` guard in `SQLiteBackend`.
  - Updated `Database` to pass resolved default DB path.
- CI import smoke test now imports `core.database` and `aiosqlite`.

## Item 9 - Standardize exception policy
- Added requested standardized exception hierarchy to `core/exceptions.py`:
  - `ApexBaseException`
  - `ApexTradingError`
  - `ApexRiskError`
  - `ApexBrokerError`
  - `ApexDataError`
  - `ApexAuthError`
- Updated critical broker/auth paths to use structured broker exception wrapping.
- Added explicit swallow annotations and stack logging on intentional swallow paths.
- Added `scripts/check_exception_policy.py` and wired it into CI.

## Item 10 - Security baseline defaults
- Updated `.env.example` weak defaults:
  - Replaced weak secret placeholders (including Grafana admin password).
- Added `scripts/check_secrets.py`:
  - Parses secret-like keys from `.env.example`.
  - Enforces non-weak values in non-development runtime.
  - Raises hard error on weak/missing required secrets.
- `main.py` startup now runs `validate_secrets()` before initialization.
- CI import smoke step executes `validate_secrets(runtime_env="production")`.

## Validation Run Summary
- Python compile checks passed for all modified Python files.
- TypeScript strict check passed: `npx tsc --noEmit`.
- Targeted tests added/updated passed:
  - `tests/test_monitoring_alert_aggregator_channels.py`
  - `tests/test_mock_token_gating.py`
  - `tests/test_ws_tenant_isolation.py`
  - `tests/test_ibkr_equity_pooling.py`
  - `tests/test_portfolio_balance_shape.py`
- Next.js production build could not complete in this sandbox due blocked external font fetch (network-limited environment).
