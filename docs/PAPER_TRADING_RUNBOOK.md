# APEX Paper Trading Runbook

## Prerequisites
```bash
pip install -r requirements_live.txt
```
Set the following in `.env` (already present for paper trading):
```
ALPACA_API_KEY=<your paper key>
ALPACA_SECRET_KEY=<your paper secret>
ALPACA_BASE_URL=https://paper-api.alpaca.markets
LIVE_MODE=paper
MAX_ORDER_USD=2000
DAILY_LOSS_LIMIT_PCT=0.03
SLIPPAGE_HALT_BPS=20
DATA_RETRY_COUNT=3
DATA_RETRY_BACKOFF_S=5
```

---

## How to Start
```bash
python scripts/live_trader.py
```
- Acquires `data/live_trader.pid` — running a second instance exits immediately.
- Runs a health check on startup; halts if overall status is `FAIL`.
- Schedules the trading cycle at **15:55 ET** daily, skipping weekends and US holidays.

---

## How to Check Health
```bash
python scripts/health_check.py
```
Writes `data/health_status.json` with `overall: OK | WARN | FAIL`.

| Status | Meaning |
|--------|---------|
| `OK`   | All checks passed |
| `WARN` | Minor issue (disk, last-run age) — trading continues |
| `FAIL` | Model missing / checksum changed — **live_trader halts** |

---

## How to Read the Execution Log
```bash
python scripts/slippage_monitor.py --report
```
Prints: avg / P50 / P95 slippage (bps), rolling-20 average, and halt status.

To tail the raw JSONL log:
```bash
tail -f data/execution_log.jsonl | python -c "import sys,json; [print(json.dumps(json.loads(l),indent=2)) for l in sys.stdin]"
```

---

## ALERT and HALT Files

| File pattern | Cause | Resolution |
|---|---|---|
| `data/ALERT_data_failure_YYYYMMDD.txt` | yfinance failed all 3 retries for a symbol | Check network / yfinance; file clears itself next successful run |
| `data/ALERT_slippage_critical_*.txt` | Rolling-20 slippage avg > 10 bps | Review broker fills; increase limit awareness |
| `data/HALT_slippage_exceeded.txt` | Rolling-20 slippage avg ≥ 20 bps | Investigate execution quality; **delete this file** to re-enable trading |
| `data/health_status.json` → `"overall":"FAIL"` | Model missing or checksum mismatch | Re-train or restore model; re-run `health_check.py` |

---

## Promote from Paper to Live

> ⚠️ Live trading uses real capital. Double-check all risk limits before promoting.

1. Update `.env`:
   ```
   ALPACA_API_KEY=<live key>
   ALPACA_SECRET_KEY=<live secret>
   ALPACA_BASE_URL=https://api.alpaca.markets
   LIVE_MODE=live
   ```
2. Delete `data/live_trader.pid` if stale.
3. Re-run health check: `python scripts/health_check.py`
4. Start: `python scripts/live_trader.py`
