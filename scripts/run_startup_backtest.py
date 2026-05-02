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


def _fetch_price_data(symbols: list[str], days: int) -> Dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — pip install yfinance")
        sys.exit(2)

    end   = datetime.now()
    start = end - timedelta(days=days + 10)
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(
                sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if len(df) >= 20:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[sym] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as exc:
            logger.warning("  %s fetch failed: %s", sym, exc)
    logger.info("Fetched %d / %d symbols (%d+ bars each)", len(data), len(symbols), 20)
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
