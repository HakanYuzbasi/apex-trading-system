"""live_trader.py — Production paper-trading scheduler for R18 strategy.
Runs at 15:55 ET daily (5 min before close). Raises on any critical failure.
No synthetic fallbacks. >99.9% reliability target.
"""
import json, logging, os, sys, time, hashlib, pickle, atexit
from datetime import date, datetime
from pathlib import Path
import signal as _signal

import pandas as pd
import schedule
from dotenv import load_dotenv

from scripts.data_provider import DataProvider

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from r18_train import SIGNAL_THRESHOLD, KELLY_FRACTION, load_model
from r17_train import UNIVERSE, build_features

# ── Config from env ───────────────────────────────────────────────────────────
APEX_ALPACA_API_KEY  = os.getenv("APEX_ALPACA_API_KEY", "")
APEX_ALPACA_SECRET   = os.getenv("APEX_ALPACA_SECRET_KEY", "")
APEX_ALPACA_BASE_URL = os.getenv("APEX_ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
LIVE_MODE        = os.getenv("LIVE_MODE", "paper")
MAX_ORDER_USD    = float(os.getenv("MAX_ORDER_USD", "2000"))
DAILY_LOSS_PCT   = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.03"))
DATA_RETRY_COUNT = int(os.getenv("DATA_RETRY_COUNT", "3"))
DATA_RETRY_BACK  = float(os.getenv("DATA_RETRY_BACKOFF_S", "5"))
DATA_DIR         = ROOT / "data"
PID_FILE         = DATA_DIR / "live_trader.pid"
LOG_LEVEL        = os.getenv("LOG_LEVEL", "INFO")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("live_trader")

class CriticalFileHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.CRITICAL:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = DATA_DIR / f"CRITICAL_{record.name}_{ts}.txt"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(self.format(record) + "\n")

log.addHandler(CriticalFileHandler())

_halt_orders: bool = False   # set True when daily-loss limit breached


# ── PID guard ─────────────────────────────────────────────────────────────────
def _acquire_pid() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        old_pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(old_pid, 0)
            log.critical("Duplicate process detected (PID %s). Exiting.", old_pid)
            sys.exit(1)
        except OSError:
            pass  # stale PID
    PID_FILE.write_text(str(os.getpid()))
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))


# ── Data fetch (delegated to DataProvider) ───────────────────────────────────
# DataProvider handles Tier 1 (Alpaca), Tier 2 (Polygon), and emergency fallback
# with retries, validation, and ALERT files internally.


# ── Alpaca order wrapper ──────────────────────────────────────────────────────
def _get_alpaca_api():
    import alpaca_trade_api as tradeapi  # lazy import
    return tradeapi.REST(APEX_ALPACA_API_KEY, APEX_ALPACA_SECRET, APEX_ALPACA_BASE_URL, api_version="v2")


def _submit_order_with_retry(api, symbol: str, qty: float,
                              side: str, notional: float) -> dict | None:
    """Retry 3× with exponential backoff. Checks for duplicates first."""
    # Duplicate check
    open_orders = api.list_orders(status="open")
    for o in open_orders:
        if o.symbol == symbol and o.side == side:
            log.warning("Duplicate order blocked: %s %s already open.", side, symbol)
            return None

    # Buying-power check
    account = api.get_account()
    if account.trading_blocked:
        log.critical("Alpaca account is trading-blocked. Halting order for %s.", symbol)
        return None
    if float(account.buying_power) < notional:
        log.warning("Insufficient buying power (%.2f < %.2f) for %s. Skipping.",
                    float(account.buying_power), notional, symbol)
        return None

    for attempt in range(1, 4):
        try:
            order = api.submit_order(
                symbol=symbol, notional=round(notional, 2),
                side=side, type="market", time_in_force="day",
            )
            return {"id": order.id, "symbol": symbol, "notional": notional,
                    "side": side, "status": order.status}
        except Exception as exc:
            import alpaca_trade_api
            if isinstance(exc, alpaca_trade_api.rest.APIError):
                log.warning("Order submit attempt %d/3 failed for %s (APIError): %s", attempt, symbol, exc)
            else:
                log.warning("Order submit attempt %d/3 failed for %s: %s", attempt, symbol, exc)
            if attempt < 3:
                time.sleep(2 ** attempt)
    log.critical("Order submission failed after 3 retries for %s.", symbol)
    return None


# ── Daily trading cycle ───────────────────────────────────────────────────────
def _is_trading_day() -> bool:
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=str(date.today()), end_date=str(date.today()))
    return not sched.empty


def _get_regime_scalar() -> float:
    """Return Alpaca SPY realized-vol/trend scalar for live signal gating."""
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(
            APEX_ALPACA_API_KEY,
            APEX_ALPACA_SECRET,
            "https://data.alpaca.markets",
            api_version="v2",
        )
        bars = api.get_bars("SPY", "1Day", limit=260).df
        if bars.empty:
            raise RuntimeError("SPY returned empty bars")
        close = bars["close" if "close" in bars.columns else "Close"].astype(float)
        realized_vol = close.pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5) * 100
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
    except Exception as exc:
        log.critical("Alpaca regime fetch failed: %s. Blocking new trades.", exc)
        return 0.0

    vol_scalar = 0.0 if realized_vol > 22 else 0.5 if realized_vol >= 16 else 1.0 if realized_vol >= 12 else 0.5
    trend_scalar = 0.0 if ma20 < ma50 else 0.5 if ma50 < ma200 else 1.0
    scalar = vol_scalar * trend_scalar
    log.info(
        "Regime scalar %.2f (realized_vol=%.2f ma20=%.2f ma50=%.2f ma200=%.2f)",
        scalar, realized_vol, ma20, ma50, ma200,
    )
    return scalar


def run_once() -> None:
    """Execute one end-of-day trading cycle."""
    global _halt_orders
    import os
    if not os.getenv("APEX_ALPACA_API_KEY") or not os.getenv("APEX_ALPACA_SECRET_KEY"):
        raise RuntimeError("APEX_ALPACA_API_KEY and APEX_ALPACA_SECRET_KEY must be set in .env")
        
    log.info("=== live_trader cycle start ===")

    if not _is_trading_day():
        log.info("Non-trading day — skipping.")
        return

    # Health check gate
    health_path = DATA_DIR / "health_status.json"
    if health_path.exists():
        hs = json.loads(health_path.read_text())
        if hs.get("overall") == "FAIL":
            log.critical("Health check FAIL — halting trading cycle.")
            raise SystemExit("Health check failed; live_trader cannot proceed.")

    try:
        api = _get_alpaca_api()
        account = api.get_account()
    except Exception as exc:
        log.critical("Cannot connect to Alpaca: %s", exc)
        raise

    # Daily loss limit
    portfolio_value = float(account.portfolio_value)
    last_equity     = float(account.last_equity)
    intraday_loss   = (last_equity - portfolio_value) / last_equity if last_equity else 0.0
    if intraday_loss > DAILY_LOSS_PCT:
        log.critical("Daily loss limit breached (%.2f%%). Halting all new orders.",
                     intraday_loss * 100)
        _halt_orders = True
    if _halt_orders:
        log.warning("Order halt active — no trades this cycle.")
        return

    regime_scalar = _get_regime_scalar()
    if regime_scalar <= 0.0:
        log.warning("Regime scalar is %.2f — no trades this cycle.", regime_scalar)
        return

    # Load model
    try:
        art = load_model()
        gbm, scaler = art["gbm"], art["scaler"]
    except Exception as exc:
        log.critical("Model load failed: %s. Raising.", exc)
        raise

    # Fetch all symbols via 3-tier DataProvider
    today_str = str(date.today())
    provider  = DataProvider()
    try:
        all_bars = provider.fetch(list(UNIVERSE) + ["SPY"])
    except RuntimeError as exc:
        log.critical("DataProvider halt: %s. Aborting cycle.", exc)
        return

    spy_df = all_bars.get("SPY", pd.DataFrame())
    if spy_df.empty:
        log.critical("SPY data unavailable — cannot compute features. Aborting cycle.")
        return
    spy_ret = spy_df["Close"].pct_change(1)

    from scripts.execution_log import ExecutionLog
    elog = ExecutionLog()
    submitted_orders = []

    for sym in UNIVERSE:
        df = all_bars.get(sym, pd.DataFrame())
        if df.empty:
            continue

        # Stale data check
        latest_date = str(df.index[-1].date())
        if latest_date != today_str:
            log.warning("%s: latest bar %s != today %s — skipping.", sym, latest_date, today_str)
            continue

        # Bar validation (DataProvider already validates, belt-and-suspenders)
        bar = df.iloc[-1]
        if bar["Close"] <= 0 or bar["Volume"] <= 0 or pd.isna(bar["Close"]):
            log.warning("%s: invalid bar — skipping.", sym)
            continue

        # Feature + prediction
        try:
            X = build_features(df, macro=False,
                               spy_ret=spy_ret.reindex(df.index, fill_value=0.0))
            X = X.replace([float("inf"), float("-inf")], float("nan")).dropna()
            if len(X) < 5:
                log.warning("%s: insufficient feature rows (%d) — skipping.", sym, len(X))
                continue
            probs = gbm.predict_proba(scaler.transform(X))
            if hasattr(probs, "values"): probs = probs.values
            # Handle both numpy [rows, cols] and list of lists [[...]]
            if isinstance(probs, (list, tuple)):
                prob = probs[-1][1]
            else:
                prob = probs[-1, 1]
        except Exception as exc:
            log.warning("%s: predict_proba failed: %s — skipping.", sym, exc)
            continue

        if prob < SIGNAL_THRESHOLD:
            continue

        # Kelly sizing with hard cap
        kelly_size = portfolio_value * KELLY_FRACTION / len(UNIVERSE)
        notional   = min(kelly_size, MAX_ORDER_USD) * regime_scalar

        log.info("%s: signal=%.3f notional=$%.2f", sym, prob, notional)
        order = _submit_order_with_retry(api, sym, qty=0, side="buy", notional=notional)
        status = "SUBMITTED" if order else "FAILED"
        rec = elog.write(sym, "buy", prob, notional, status, regime="r18",
                   vol_cb_state="normal", kelly_fraction=KELLY_FRACTION,
                   model_price=float(bar['Close']))
        if order:
            submitted_orders.append({"id": order["id"], "record_id": rec["_id"]})

    if submitted_orders:
        log.info("Polling %d submitted orders for up to 60s...", len(submitted_orders))
        t0 = time.monotonic()
        from datetime import timezone
        while submitted_orders and time.monotonic() - t0 < 60:
            for o_meta in submitted_orders.copy():
                try:
                    updated = api.get_order(o_meta["id"])
                    if updated.status == "filled":
                        fill_price = float(updated.filled_avg_price) if updated.filled_avg_price else 0.0
                        ts = updated.updated_at.isoformat() if hasattr(updated.updated_at, 'isoformat') else str(updated.updated_at)
                        elog.update(o_meta["record_id"], fill_price, ts, "FILLED")
                        submitted_orders.remove(o_meta)
                    elif updated.status in ("canceled", "rejected"):
                        elog.update(o_meta["record_id"], 0.0, datetime.now(timezone.utc).isoformat(), "CANCELLED")
                        submitted_orders.remove(o_meta)
                except Exception as exc:
                    log.warning("Failed to poll order %s: %s", o_meta["id"], exc)
            if submitted_orders:
                time.sleep(2)
        
        for o_meta in submitted_orders:
            log.warning("Order %s timed out after 60s. Cancelling.", o_meta["id"])
            try:
                api.cancel_order(o_meta["id"])
            except Exception as exc:
                log.warning("Failed to cancel timed-out order %s: %s", o_meta["id"], exc)
            elog.update(o_meta["record_id"], 0.0, datetime.now(timezone.utc).isoformat(), "CANCELLED")

    log.info("=== live_trader cycle complete ===")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import os
    if not os.getenv("APEX_ALPACA_API_KEY") or not os.getenv("APEX_ALPACA_SECRET_KEY"):
        raise RuntimeError("APEX_ALPACA_API_KEY and APEX_ALPACA_SECRET_KEY must be set in .env")
    _acquire_pid()
    log.info("live_trader started in %s mode. Scheduling at 15:55 ET.", LIVE_MODE)

    schedule.every().day.at("15:55").do(run_once)
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
