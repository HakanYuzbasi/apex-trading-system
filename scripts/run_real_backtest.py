#!/usr/bin/env python3
"""
run_real_backtest.py — Walk-forward backtest on REAL historical OHLCV data.

Uses actual yfinance data (2020-present) and the live ApexConfig thresholds
so results reflect the actual system edge, not synthetic approximations.

Run:
    python3 scripts/run_real_backtest.py
    python3 scripts/run_real_backtest.py --years 3 --symbols 20
    python3 scripts/run_real_backtest.py --session apex --symbols 30 --years 5
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from core.logging_config import setup_logging
setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)
logger = logging.getLogger(__name__)

from config import ApexConfig
from scripts.god_level_backtest import GodLevelBacktester, print_results, go_no_go_assessment


# ── Symbol universe ──────────────────────────────────────────────────────────
EQUITY_CORE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "JPM", "GS", "BAC",
    "XOM", "CVX",
    "SPY", "QQQ", "IWM",
    "NFLX", "HD", "UNH", "LLY", "AVGO",
]

CRYPTO_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD",
]

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY"]


def fetch_real_ohlcv(symbols: list, years: int = 3) -> dict:
    """
    Download daily OHLCV from yfinance for each symbol.
    Returns {symbol: pd.DataFrame} with Open/High/Low/Close/Volume columns.
    Rate-limited to avoid bans.
    """
    import yfinance as yf
    import time

    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25) + 60)   # extra 60 for warmup
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    data = {}
    for i, sym in enumerate(symbols):
        try:
            df = yf.download(sym, start=start_str, end=end_str,
                             progress=False, auto_adjust=True)
            if df is not None and len(df) >= 120:
                df.index = pd.to_datetime(df.index)
                # yfinance multi-level columns when downloading single ticker
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                # Keep only OHLCV
                needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                data[sym] = df[needed].dropna()
                print(f"  ✓ {sym}: {len(data[sym])} days")
            else:
                print(f"  ✗ {sym}: insufficient data ({len(df) if df is not None else 0} days)")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")

        # Polite rate-limiting (yfinance bans on rapid-fire requests)
        if (i + 1) % 5 == 0:
            time.sleep(1.0)

    return data


def run(session: str = "apex", n_symbols: int = 20, years: int = 5,
        walk_forward: bool = True, capital: float = 100_000) -> None:

    print("\n" + "=" * 70)
    print(f"REAL HISTORICAL BACKTEST  |  session={session}  |  {years}yr  |  {n_symbols} symbols")
    print("=" * 70)

    # Build symbol list
    if session == "crypto":
        symbols = CRYPTO_SYMBOLS[:n_symbols]
    elif session in ("core", "apex"):
        symbols = (EQUITY_CORE + SECTOR_ETFS)[:n_symbols]
    else:
        symbols = (EQUITY_CORE + CRYPTO_SYMBOLS)[:n_symbols]

    print(f"\nDownloading {len(symbols)} symbols from yfinance ({years} years)…")
    historical_data = fetch_real_ohlcv(symbols, years=years)

    if len(historical_data) < 5:
        print("\n⛔  Too few symbols downloaded. Check internet / yfinance.")
        return

    print(f"\n✓ Downloaded {len(historical_data)} / {len(symbols)} symbols")
    print(f"  Date range: {min(df.index[0] for df in historical_data.values()).date()} "
          f"→ {max(df.index[-1] for df in historical_data.values()).date()}")

    # ── Run walk-forward backtest ───────────────────────────────────────────
    print(f"\nRunning {'walk-forward' if walk_forward else 'simple'} backtest…")
    backtester = GodLevelBacktester(initial_capital=capital)
    backtester._session_type = session

    result = backtester.run_backtest(
        symbols=list(historical_data.keys()),
        walk_forward=walk_forward,
        historical_data_override=historical_data,
    )

    if result is None:
        print("\n⛔  Backtest returned no result.")
        return

    # ── Stress tests ───────────────────────────────────────────────────────
    mc_results = backtester.run_monte_carlo(n_simulations=500)
    result.monte_carlo = mc_results

    stress_results = backtester.run_all_stress_tests(
        sharpe_ratio=result.sharpe_ratio,
        n_trades=result.total_trades,
    )
    result.deflated_sharpe = stress_results.get("deflated_sharpe")
    result.stress_tests = stress_results

    # ── Print ───────────────────────────────────────────────────────────────
    print_results(result, stress_results=stress_results)
    go_no_go_assessment(result, stress_results=stress_results)

    # ── IC per feature (if available) ──────────────────────────────────────
    _print_ic_summary(result)

    return result


def _print_ic_summary(result) -> None:
    """Print IC (signal→5-day return correlation) for each feature if trades exist."""
    if not result.trades:
        return
    try:
        from monitoring.ic_tracker import ICTracker
        tracker = ICTracker(persist=False)
        print("\n" + "=" * 70)
        print("SIGNAL IC SUMMARY  (correlation of signal → 5-day fwd return)")
        print("=" * 70)
        summary = tracker.get_summary()
        if summary:
            for feat, ic in sorted(summary.items(), key=lambda x: -abs(x[1]))[:15]:
                bar = "█" * int(abs(ic) * 20)
                sign = "+" if ic >= 0 else "-"
                print(f"  {feat:<30} IC={sign}{abs(ic):.4f}  {bar}")
        else:
            print("  (IC data builds over live trading — re-run after 30+ trades)")
    except Exception:
        pass


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real historical backtest")
    parser.add_argument("--session", default="apex",
                        choices=["apex", "core", "crypto", "unified"],
                        help="Session type (default: apex — uses live config thresholds)")
    parser.add_argument("--symbols", type=int, default=20,
                        help="Number of symbols to test (default: 20)")
    parser.add_argument("--years", type=int, default=5,
                        help="Years of history (default: 5 — includes 2022 bear market)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Starting capital (default: 100000)")
    parser.add_argument("--no-walk-forward", dest="walk_forward",
                        action="store_false", default=True,
                        help="Skip walk-forward, use simple single-period")
    args = parser.parse_args()

    run(
        session=args.session,
        n_symbols=args.symbols,
        years=args.years,
        walk_forward=args.walk_forward,
        capital=args.capital,
    )
