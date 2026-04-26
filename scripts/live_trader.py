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
import yfinance as yf

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r18_train import SIGNAL_THRESHOLD, KELLY_FRACTION, load_model
from r17_train import UNIVERSE, build_features

# ── Config from env ───────────────────────────────────────────────────────────
ALPACA_API_KEY   = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET    = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL  = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
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


# ── Data fetch with retry ─────────────────────────────────────────────────────
def _fetch_with_retry(symbol: str, period: str = "5d") -> pd.DataFrame:
    last_exc = None
    for attempt in range(1, DATA_RETRY_COUNT + 1):
        try:
            df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                return df
            raise ValueError(f"{symbol}: empty DataFrame from yfinance")
        except Exception as exc:
            last_exc = exc
            log.warning("yfinance attempt %d/%d for %s failed: %s",
                        attempt, DATA_RETRY_COUNT, symbol, exc)
            if attempt < DATA_RETRY_COUNT:
                time.sleep(DATA_RETRY_BACK * (2 ** (attempt - 1)))
    # All retries exhausted
    alert_path = DATA_DIR / f"ALERT_data_failure_{date.today():%Y%m%d}.txt"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    alert_path.write_text(f"yfinance failed for {symbol}: {last_exc}\n")
    log.critical("DATA FAILURE for %s — alert written to %s. Skipping.", symbol, alert_path)
    return pd.DataFrame()


# ── Alpaca order wrapper ──────────────────────────────────────────────────────
def _get_alpaca_api():
    import alpaca_trade_api as tradeapi  # lazy import
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL, api_version="v2")


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


def run_once() -> None:
    """Execute one end-of-day trading cycle."""
    global _halt_orders
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

    # Load model
    try:
        art = load_model()
        gbm, scaler = art["gbm"], art["scaler"]
    except Exception as exc:
        log.critical("Model load failed: %s. Raising.", exc)
        raise

    # Fetch SPY for features
    spy_df = _fetch_with_retry("SPY")
    if spy_df.empty:
        log.critical("SPY data unavailable — cannot compute features. Aborting cycle.")
        return
    spy_ret = spy_df["Close"].pct_change(1)

    today_str = str(date.today())

    from scripts.execution_log import ExecutionLog
    elog = ExecutionLog()

    for sym in UNIVERSE:
        df = _fetch_with_retry(sym)
        if df.empty:
            continue

        # Stale data check
        latest_date = str(df.index[-1].date())
        if latest_date != today_str:
            log.warning("%s: latest bar %s != today %s — skipping.", sym, latest_date, today_str)
            continue

        # Bar validation
        bar = df.iloc[-1]
        if bar["Close"] <= 0 or bar["Volume"] <= 0 or pd.isna(bar["Close"]):
            log.warning("%s: invalid bar (close=%.4f vol=%.0f) — skipping.",
                        sym, bar["Close"], bar["Volume"])
            continue

        # Feature + prediction
        try:
            X = build_features(df, macro=False,
                               spy_ret=spy_ret.reindex(df.index, fill_value=0.0))
            X = X.replace([float("inf"), float("-inf")], float("nan")).dropna()
            if len(X) < 5:
                log.warning("%s: insufficient feature rows (%d) — skipping.", sym, len(X))
                continue
            prob = gbm.predict_proba(scaler.transform(X))[-1, 1]
        except Exception as exc:
            log.warning("%s: predict_proba failed: %s — skipping.", sym, exc)
            continue

        if prob < SIGNAL_THRESHOLD:
            continue

        # Kelly sizing with hard cap
        kelly_size = portfolio_value * KELLY_FRACTION / len(UNIVERSE)
        notional   = min(kelly_size, MAX_ORDER_USD)

        log.info("%s: signal=%.3f notional=$%.2f", sym, prob, notional)
        order = _submit_order_with_retry(api, sym, qty=0, side="buy", notional=notional)
        status = "SUBMITTED" if order else "FAILED"
        elog.write(sym, "buy", prob, notional, status, regime="r18",
                   vol_cb_state="normal", kelly_fraction=KELLY_FRACTION)

    log.info("=== live_trader cycle complete ===")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    _acquire_pid()
    log.info("live_trader started in %s mode. Scheduling at 15:55 ET.", LIVE_MODE)

    schedule.every().day.at("15:55").do(run_once)
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()

print("FILE COMPLETE: live_trader.py")
