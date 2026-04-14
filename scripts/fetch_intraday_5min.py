"""
fetch_intraday_5min.py — Download 5-minute bar data via Alpaca historical API
==============================================================================
Downloads 6 months of 5-minute OHLCV bars for a symbol pair and writes a clean
CSV ready for the walk_forward_backtest.py --csv flag.

Critical design decisions
--------------------------
1. Market-hours filtering (09:30–16:00 ET, inclusive):
   Overnight gaps are NOT pairs-trading signal. Carrying a spread z-score
   across a session boundary contaminates the Kalman state with gap noise.
   Every bar outside regular market hours is dropped before saving.

2. Session-boundary tagging:
   A `session_start` boolean column is written. The backtest engine uses this
   to reset the spread rolling window at each open, preventing the Kalman
   filter from treating the gap between 16:00 one day and 09:30 the next as
   a continuous spread movement.

3. Feed selection (IEX vs SIP):
   IEX feed is free and sufficient for AAPL/MSFT (high-liquidity names with
   tight IEX coverage). If you have an Alpaca paid subscription, set
   ALPACA_STOCK_DATA_FEED=sip in .env for consolidated tape data.

4. Pagination:
   alpaca-py handles pagination automatically via get_stock_bars(). For 6
   months × 2 symbols × ~78 bars/day × 126 days ≈ 19,656 bars per symbol,
   this fits in a single paginated response with no manual chunking needed.

5. Alignment:
   AAPL and MSFT bars are inner-joined on timestamp. Any bar present in one
   feed but not the other (rare data gaps) is dropped to ensure the backtest
   always has synchronised pairs.

Usage
-----
    # Uses keys from environment (loaded from .env automatically)
    python scripts/fetch_intraday_5min.py

    # Explicit overrides
    python scripts/fetch_intraday_5min.py --symbols AAPL MSFT --months 6 --out data/aapl_msft_5min.csv

    # V/MA pair
    python scripts/fetch_intraday_5min.py --symbols V MA --out data/v_ma_5min.csv

Output columns
--------------
    timestamp    — UTC ISO8601 string (bar open time)
    date         — alias of timestamp (keeps backtest engine happy)
    close_a      — close price of symbol A
    close_b      — close price of symbol B
    volume_a     — volume of symbol A
    volume_b     — volume of symbol B
    session_start— True on the first bar of each trading session (09:30 ET)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env before importing anything that reads env vars
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv optional — keys must be in environment directly

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_intraday")

ET = pytz.timezone("America/New_York")
MARKET_OPEN  = (9, 30)   # (hour, minute) in ET
MARKET_CLOSE = (16, 0)


# ---------------------------------------------------------------------------
# Alpaca client
# ---------------------------------------------------------------------------

def build_client() -> StockHistoricalDataClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env or environment."
        )
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def resolve_feed() -> DataFeed:
    """Honour the same env var the live harness uses."""
    raw = os.getenv("ALPACA_STOCK_DATA_FEED", "iex").strip().lower()
    feed_map = {"sip": DataFeed.SIP, "iex": DataFeed.IEX, "otc": DataFeed.OTC}
    feed = feed_map.get(raw, DataFeed.IEX)
    logger.info("Data feed: %s  (set ALPACA_STOCK_DATA_FEED=sip for consolidated tape)", raw.upper())
    return feed


# ---------------------------------------------------------------------------
# Fetch one symbol
# ---------------------------------------------------------------------------

def fetch_bars(client: StockHistoricalDataClient,
               symbol: str,
               start: datetime,
               end: datetime,
               feed: DataFeed) -> pd.DataFrame:
    """
    Fetch 5-minute bars for one symbol and return a DataFrame indexed by UTC timestamp.
    Columns: open, high, low, close, volume, trade_count, vwap
    """
    logger.info("Fetching %s  5-min bars  %s → %s  ...", symbol, start.date(), end.date())

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed,
        adjustment="split",   # adjust for stock splits — critical for multi-year data
    )

    response = client.get_stock_bars(request)

    if symbol not in response or not response[symbol]:
        raise RuntimeError(
            f"No bars returned for {symbol}. "
            "Check your API key has data permissions and the symbol is correct."
        )

    bars = response[symbol]
    records = []
    for bar in bars:
        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        records.append({
            "timestamp": ts,
            "open":   float(bar.open),
            "high":   float(bar.high),
            "low":    float(bar.low),
            "close":  float(bar.close),
            "volume": float(bar.volume),
        })

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    logger.info("  %s: %d raw bars", symbol, len(df))
    return df


# ---------------------------------------------------------------------------
# Session filtering
# ---------------------------------------------------------------------------

def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only bars whose timestamp falls within regular market hours (09:30–16:00 ET).

    WHY THIS MATTERS:
    The Kalman filter tracks a dynamic hedge ratio. If you feed it a bar from
    09:30 immediately after a bar from 16:00 the previous day, the filter sees
    a price discontinuity that is purely a gap event, not a cointegration
    signal. This would make the z-score spike at every open, generating false
    entry signals. Filtering to single-session bars eliminates this entirely.
    """
    et_index = df.index.tz_convert(ET)
    open_h,  open_m  = MARKET_OPEN
    close_h, close_m = MARKET_CLOSE

    in_session = (
        (et_index.hour > open_h)  |
        ((et_index.hour == open_h)  & (et_index.minute >= open_m))
    ) & (
        (et_index.hour < close_h) |
        ((et_index.hour == close_h) & (et_index.minute <= close_m))
    )

    filtered = df[in_session]
    dropped = len(df) - len(filtered)
    if dropped:
        logger.info("  Dropped %d outside-market-hours bars", dropped)
    return filtered


def tag_session_starts(df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series that is True on the first bar of each trading session.
    Used by the backtest to reset the Kalman spread window at each open.
    """
    et_index = df.index.tz_convert(ET)
    is_open = (et_index.hour == MARKET_OPEN[0]) & (et_index.minute == MARKET_OPEN[1])
    return is_open.rename("session_start")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_pair_csv(symbol_a: str, symbol_b: str,
                   months: int, out_path: Path) -> None:
    """
    Full pipeline:
      1. Fetch both symbols
      2. Filter to market hours
      3. Inner-join on timestamp (drops any bar present in only one feed)
      4. Tag session starts
      5. Write CSV
    """
    client = build_client()
    feed = resolve_feed()

    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=int(months * 30.44))  # approximate

    df_a = fetch_bars(client, symbol_a, start, end, feed)
    df_b = fetch_bars(client, symbol_b, start, end, feed)

    df_a = filter_market_hours(df_a)
    df_b = filter_market_hours(df_b)

    # Inner join — both symbols must have a bar at the same timestamp
    df_a = df_a[["close", "volume"]].rename(columns={"close": "close_a", "volume": "volume_a"})
    df_b = df_b[["close", "volume"]].rename(columns={"close": "close_b", "volume": "volume_b"})

    df = df_a.join(df_b, how="inner")

    pre_align = len(df_a)
    dropped_align = pre_align - len(df)
    if dropped_align:
        logger.warning(
            "Dropped %d bars during timestamp alignment (present in one symbol only). "
            "This is normal for occasional data gaps.",
            dropped_align,
        )

    # Tag session starts
    df["session_start"] = tag_session_starts(df)

    # Add human-readable date column (backtest engine expects 'date')
    df.index.name = "timestamp"
    df = df.reset_index()
    df["date"] = df["timestamp"]

    # Final column order
    df = df[["timestamp", "date", "close_a", "close_b", "volume_a", "volume_b", "session_start"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_sessions = df["session_start"].sum()
    logger.info(
        "Saved %d bars  |  %d trading sessions  |  %s → %s",
        len(df), n_sessions,
        df["timestamp"].iloc[0], df["timestamp"].iloc[-1],
    )
    logger.info("CSV → %s", out_path)
    logger.info("")
    logger.info("Bars per session: %.1f  (expect ~78 for full 6.5h session at 5-min)", len(df) / max(n_sessions, 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Alpaca 5-min bar data for a symbol pair"
    )
    parser.add_argument("--symbols", nargs=2, default=["AAPL", "MSFT"],
                        metavar=("A", "B"))
    parser.add_argument("--months", type=int, default=6,
                        help="Months of history to fetch (default: 6). "
                             "IEX free tier: up to ~15 months. SIP: unlimited.")
    parser.add_argument("--out", type=str,
                        default="data/aapl_msft_5min.csv",
                        help="Output CSV path (default: data/aapl_msft_5min.csv)")
    args = parser.parse_args()

    # Auto-name output if using non-default symbols
    out_path = Path(args.out)
    if args.symbols != ["AAPL", "MSFT"] and args.out == "data/aapl_msft_5min.csv":
        a, b = args.symbols
        out_path = Path(f"data/{a.lower()}_{b.lower()}_5min.csv")

    build_pair_csv(
        symbol_a=args.symbols[0],
        symbol_b=args.symbols[1],
        months=args.months,
        out_path=PROJECT_ROOT / out_path,
    )


if __name__ == "__main__":
    main()
