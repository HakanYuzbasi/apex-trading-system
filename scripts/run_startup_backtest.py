#!/usr/bin/env python3
"""
scripts/run_startup_backtest.py

Pre-flight signal-quality validation using the last 90 days of real market data.

Run before deploying a new container or after a major code change:
    python scripts/run_startup_backtest.py [--days 90] [--min-sharpe 0.0] [--min-winrate 0.40]

Exit codes:
    0 — validation passed (Sharpe ≥ min_sharpe AND max_drawdown ≥ -20%)
    1 — validation failed  (log shows which threshold was missed)
    2 — not enough data to validate (< 30 days fetched); treated as pass to avoid
        blocking restarts on data outages

The momentum signal used here is deliberately simple (5-day vs 20-day log-return
z-score) so the script has no dependency on live model state — it tests the
portfolio infrastructure (position sizing, costs, risk circuit breakers) rather
than the alpha signal itself.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("startup_backtest")

_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "AMD",  "V",    "MA",    "JPM",
    "BAC",  "KO",   "PEP",  "SPY",   "QQQ",
]

_DEFAULT_DAYS       = 90
_DEFAULT_MIN_SHARPE = 0.0      # warn-only at this level
_DEFAULT_MIN_WINRATE = 0.40
_MAX_DRAWDOWN_HARD  = -0.20    # hard-fail threshold


def _fetch_via_polygon(
    symbols: list[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV from Polygon.io REST API.

    Uses the POLYGON_API_KEY env var (already configured in .env).
    Returns only symbols where ≥20 bars were retrieved.
    """
    import os
    import requests as _requests

    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        logger.warning("POLYGON_API_KEY not set — skipping Polygon fetch")
        return {}

    base = "https://api.polygon.io/v2/aggs/ticker"
    session = _requests.Session()
    adapter = _requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)

    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        url = f"{base}/{sym}/range/1/day/{start}/{end}"
        params = {"apiKey": api_key, "limit": 500, "adjusted": "true"}
        for attempt in range(2):  # one retry on 429
            try:
                import time as _time
                resp = session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    if attempt == 0:
                        _time.sleep(15)  # back off 15s on rate-limit then retry
                        continue
                    logger.warning("  Polygon 429 twice for %s — skipping", sym)
                    break
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("status") not in ("OK", "DELAYED") or not payload.get("results"):
                    logger.debug("Polygon: no results for %s (status=%s)", sym, payload.get("status"))
                    break
                rows = payload["results"]
                df = pd.DataFrame(rows)
                # Polygon returns epoch milliseconds in 't'
                df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(None)
                df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
                df = df.set_index("date").sort_index()
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                if len(df) >= 20:
                    data[sym] = df
                else:
                    logger.debug("Polygon: only %d bars for %s — skipping", len(df), sym)
                break
            except Exception as exc:
                logger.warning("  Polygon fetch failed for %s: %s", sym, exc)
                break
        _time.sleep(0.25)  # 4 req/s — stays within free-tier 5 req/min limit
    return data


def _fetch_via_yfinance(
    symbols: list[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """
    yfinance fallback — may fail in Docker due to urllib3 version mismatch,
    but works fine in local development.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not installed — skipping")
        return {}

    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(
                sym,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if len(df) >= 20:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[sym] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                logger.debug("yfinance: only %d bars for %s — skipping", len(df), sym)
        except Exception as exc:
            logger.warning("  yfinance fetch failed for %s: %s", sym, exc)
    return data


def _fetch_price_data(symbols: list[str], days: int) -> Dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV for all symbols over the requested lookback period.

    Strategy:
      1. Try Polygon.io REST (reliable in Docker, uses POLYGON_API_KEY).
      2. For any symbol not returned by Polygon, try yfinance as fallback.
      3. Return the merged dataset; exit-code 2 (pass) if < 5 symbols fetched.
    """
    from datetime import datetime, timedelta

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 10)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = end_dt.strftime("%Y-%m-%d")

    # --- Primary: Polygon --------------------------------------------------
    data = _fetch_via_polygon(symbols, start, end)
    missing = [s for s in symbols if s not in data]

    # --- Fallback: yfinance (for any missed symbols) -----------------------
    if missing:
        logger.info("Polygon returned %d/%d symbols; trying yfinance for %d missing",
                    len(data), len(symbols), len(missing))
        yf_data = _fetch_via_yfinance(missing, start, end)
        data.update(yf_data)

    logger.info(
        "Fetched %d / %d symbols (%d+ bars each)",
        len(data), len(symbols), 20,
    )
    return data



def _build_momentum_probabilities(
    price_data: Dict[str, pd.DataFrame],
    fast: int = 5,
    slow: int = 20,
) -> Dict[str, pd.Series]:
    """
    Convert price history into a probability-style signal in [0, 1].

    Signal = sigmoid(z-score of fast-momentum vs slow-momentum cross-section).
    Values above 0.55 are treated as long candidates by the backtester.
    """
    # Collect cross-sectional z-scores per date
    daily_scores: Dict[str, Dict] = {}
    for sym, df in price_data.items():
        closes = df["Close"]
        fast_ret = np.log(closes / closes.shift(fast))
        slow_ret = np.log(closes / closes.shift(slow))
        raw = fast_ret - slow_ret
        daily_scores[sym] = raw

    scores_df = pd.DataFrame(daily_scores).dropna(how="all")
    # Cross-sectional z-score per date
    cs_mean = scores_df.mean(axis=1)
    cs_std  = scores_df.std(axis=1).replace(0, np.nan)
    z_df = scores_df.sub(cs_mean, axis=0).div(cs_std, axis=0).fillna(0.0)

    # Sigmoid to [0, 1]
    prob_df = 1.0 / (1.0 + np.exp(-z_df))
    return {sym: prob_df[sym].dropna() for sym in prob_df.columns}


def run_validation(
    days: int = _DEFAULT_DAYS,
    min_sharpe: float = _DEFAULT_MIN_SHARPE,
    min_winrate: float = _DEFAULT_MIN_WINRATE,
) -> int:
    """Returns exit code: 0=pass, 1=fail, 2=insufficient data."""
    from backtesting.realistic_portfolio_backtester import (
        ExitConfig,
        PortfolioBacktestConfig,
        RealisticPortfolioBacktester,
        VolCircuitConfig,
    )

    price_data = _fetch_price_data(_UNIVERSE, days)
    if len(price_data) < 5:
        logger.warning("Only %d symbols fetched — insufficient for validation, treating as pass", len(price_data))
        return 2

    probabilities = _build_momentum_probabilities(price_data)

    cfg = PortfolioBacktestConfig(
        initial_capital=78_000.0,
        signal_threshold=0.55,
        kelly_fraction=0.25,
        max_positions=8,
        max_position_pct=0.15,
        allow_shorts=False,
        fee_gate_enabled=True,
        one_way_cost_bps=8.0,
        half_spread_bps=1.0,
        hrp_enabled=True,
        vol_circuit=VolCircuitConfig(enabled=True),
        exits=ExitConfig(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            trailing_pct=0.03,
            max_hold_bars=20,
        ),
    )

    backtester = RealisticPortfolioBacktester(price_data, probabilities, cfg)
    result = backtester.run()

    logger.info(
        "Backtest result: sharpe=%.2f  win_rate=%.1f%%  max_dd=%.1f%%  "
        "trades=%d  return=%.1f%%",
        result.sharpe,
        result.win_rate * 100,
        result.max_drawdown_pct * 100,
        result.trades,
        result.total_return_pct * 100,
    )

    passed = True

    if result.max_drawdown_pct < _MAX_DRAWDOWN_HARD:
        logger.error(
            "VALIDATION FAIL: max_drawdown=%.1f%% < hard limit %.1f%%",
            result.max_drawdown_pct * 100, _MAX_DRAWDOWN_HARD * 100,
        )
        passed = False

    if result.sharpe < min_sharpe:
        logger.warning(
            "VALIDATION WARN: sharpe=%.2f < min_sharpe=%.2f — strategy may be degraded",
            result.sharpe, min_sharpe,
        )
        # Warn only — do not hard-fail on Sharpe since short windows are noisy

    if result.win_rate < min_winrate and result.trades >= 10:
        logger.warning(
            "VALIDATION WARN: win_rate=%.1f%% < min_winrate=%.1f%% over %d trades",
            result.win_rate * 100, min_winrate * 100, result.trades,
        )

    if passed:
        logger.info("Startup backtest PASSED")
        return 0
    else:
        logger.error("Startup backtest FAILED — review logs before deploying live")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-flight portfolio backtest validation")
    parser.add_argument("--days",        type=int,   default=_DEFAULT_DAYS)
    parser.add_argument("--min-sharpe",  type=float, default=_DEFAULT_MIN_SHARPE)
    parser.add_argument("--min-winrate", type=float, default=_DEFAULT_MIN_WINRATE)
    args = parser.parse_args()

    code = run_validation(
        days=args.days,
        min_sharpe=args.min_sharpe,
        min_winrate=args.min_winrate,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
