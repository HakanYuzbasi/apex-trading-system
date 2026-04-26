"""data_provider.py — 3-tier market data fetch for live trading.
Tier 1: Alpaca Data API (primary)
Tier 2: Polygon.io REST  (fallback)
Tier 3: yfinance         (emergency)
All tiers: 3× retry, 5s exponential backoff, bar validation.
Per-symbol source logged to data/data_source_log.jsonl.
"""
import json, logging, os, time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger("data_provider")

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_LOG = DATA_DIR / "data_source_log.jsonl"

ALPACA_API_KEY  = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET   = os.getenv("ALPACA_SECRET_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
_RETRIES        = int(os.getenv("DATA_RETRY_COUNT", "3"))
_BACKOFF        = float(os.getenv("DATA_RETRY_BACKOFF_S", "5"))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate(df: pd.DataFrame, symbol: str) -> bool:
    """Return True only if latest bar has price>0, volume>0, no NaN."""
    if df.empty:
        return False
    bar = df.iloc[-1]
    if df[["Close", "Volume"]].isnull().any().any():
        log.warning("%s: NaN in bar — rejected", symbol)
        return False
    if bar.get("Close", 0) <= 0 or bar.get("Volume", 0) <= 0:
        log.warning("%s: non-positive price/volume — rejected", symbol)
        return False
    return True


def _log_source(symbol: str, src: str, latency_ms: float) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {"date": str(date.today()), "symbol": symbol,
              "data_source": src, "latency_ms": round(latency_ms, 1)}
    with SOURCE_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _retry(fn, label: str):
    """Call fn() up to _RETRIES times; return result or None on exhaustion."""
    for attempt in range(1, _RETRIES + 1):
        try:
            result = fn()
            if result is not None:
                return result
        except Exception as exc:
            log.warning("[%s] attempt %d/%d failed: %s", label, attempt, _RETRIES, exc)
        if attempt < _RETRIES:
            time.sleep(_BACKOFF * (2 ** (attempt - 1)))
    return None


# ── Tier 1: Alpaca Data API ───────────────────────────────────────────────────
def _fetch_alpaca(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET)
    req    = StockBarsRequest(symbol_or_symbols=symbol,
                               timeframe=TimeFrame.Day, start=start, end=end)
    bars   = client.get_stock_bars(req)
    df     = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    return df if not df.empty else None


# ── Tier 2: Polygon.io REST ───────────────────────────────────────────────────
def _fetch_polygon(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    import urllib.request, urllib.error
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day"
           f"/{start}/{end}?adjusted=true&sort=asc&limit=10"
           f"&apiKey={POLYGON_API_KEY}")
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    results = data.get("results", [])
    if not results:
        return None
    df = pd.DataFrame(results)
    df["Date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    df = df.set_index("Date").rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ── Tier 3: yfinance (emergency) ──────────────────────────────────────────────
def _fetch_yfinance(symbol: str) -> Optional[pd.DataFrame]:
    import yfinance as yf
    df = yf.download(symbol, period="5d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df if not df.empty else None


# ── DataProvider ──────────────────────────────────────────────────────────────
class DataProvider:
    """Fetch OHLCV data through a 3-tier fallback chain."""

    def fetch(self, symbols: list[str],
              fetch_date: Optional[date] = None) -> dict[str, pd.DataFrame]:
        """Return {symbol: DataFrame} for all symbols using best available tier."""
        fetch_date = fetch_date or date.today()
        start      = str(fetch_date - timedelta(days=10))
        end        = str(fetch_date)
        results: dict[str, pd.DataFrame] = {}

        for sym in symbols:
            df = self._fetch_symbol(sym, start, end)
            if df is not None:
                results[sym] = df
            else:
                log.warning("%s: all tiers exhausted — skipping symbol.", sym)

        if not results:
            alert = DATA_DIR / f"ALERT_data_failure_{fetch_date:%Y%m%d}.txt"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            alert.write_text(f"All symbols failed all data tiers on {fetch_date}\n")
            log.critical("DATA TOTAL FAILURE — ALERT written: %s. Halting.", alert)
            raise RuntimeError(f"DataProvider: all {len(symbols)} symbols failed all tiers.")

        return results

    def _fetch_symbol(self, sym: str, start: str, end: str) -> Optional[pd.DataFrame]:
        tiers = [
            ("alpaca",  lambda: _fetch_alpaca(sym, start, end)),
            ("polygon", lambda: _fetch_polygon(sym, start, end)),
            ("yfinance", lambda: _fetch_yfinance(sym)),
        ]
        for tier_name, fn in tiers:
            t0  = time.monotonic()
            df  = _retry(fn, f"{tier_name}/{sym}")
            ms  = (time.monotonic() - t0) * 1000
            if df is not None and _validate(df, sym):
                _log_source(sym, tier_name, ms)
                log.info("%s: fetched from %s (%.0fms)", sym, tier_name, ms)
                if tier_name == "yfinance":
                    warn = DATA_DIR / f"WARN_data_fallback_{date.today():%Y%m%d}.txt"
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    warn.write_text(f"{sym} fell back to yfinance on {date.today()}\n")
                    log.warning("%s: EMERGENCY yfinance fallback used.", sym)
                return df
            log.warning("%s: tier %s failed validation or returned empty.", sym, tier_name)
        return None


print("FILE COMPLETE: data_provider.py")
