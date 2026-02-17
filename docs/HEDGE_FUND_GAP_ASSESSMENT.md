# APEX vs Hedge-Fund Platform Gap Assessment

This document captures practical gaps between APEX and institutional hedge-fund trading stacks.

## Implemented now (high impact)

1. Centralized pre-trade hard-limit gateway (`risk/pretrade_risk_gateway.py`)
- Added deterministic, centralized checks before order routing:
  - max order notional
  - max order shares (fat-finger control)
  - price-band validation (bps vs reference)
  - ADV participation cap
  - projected gross exposure cap
- Integrated directly into live entry flow in `main.py` (fail-closed optional).

2. Tamper-evident pre-trade decision trail
- Every pre-trade allow/block decision is appended to:
  - `data/audit/pretrade_gateway/pretrade_gateway_YYYYMMDD.jsonl`
- Records are hash-chained via `prev_hash` + `hash` for auditability.

3. Operational observability for pre-trade risk
- Added Prometheus counter:
  - `apex_pretrade_gate_decisions_total{asset_class,result,reason}`
- Emitted from live trading loop for allow/block monitoring in Grafana.

## Still missing vs top-tier hedge funds (next phases)

1. Deterministic replay and event sourcing
- Full event journal for market data, signal state, orders, and fills enabling exact run replay.

2. Venue-aware microstructure controls
- Dynamic participation limits by venue liquidity regime and queue-position-aware execution.

3. Formal model governance in runtime path
- Enforced model manifest verification and runtime model policy checks before live start.

4. Intraday portfolio stress engine in control loop
- Continuous live scenario shocks with automatic risk de-leveraging hooks.

5. Production-grade SLO automation
- Explicit latency/error SLOs with automatic degradation modes and pager integration.
