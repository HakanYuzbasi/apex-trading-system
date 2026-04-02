#!/usr/bin/env python3
"""
Run the god_level_backtest with synthetic market data.
Used when no internet connection is available (e.g., sandbox environments).

Generates realistic OHLCV data with regime changes for 40 symbols.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_ohlcv(
    symbol: str,
    n_days: int = 504,
    seed: int = None,
) -> pd.DataFrame:
    """Generate realistic OHLCV data with regime changes.

    Uses a regime-switching GBM model:
    - Bull: mu=0.15/yr, sigma=0.18/yr
    - Bear: mu=-0.10/yr, sigma=0.28/yr
    - Sideways: mu=0.03/yr, sigma=0.14/yr
    - High vol: mu=0.0/yr, sigma=0.40/yr
    """
    if seed is None:
        seed = hash(symbol) % (2**31)
    rng = np.random.default_rng(seed)

    # Regime transition matrix (daily probabilities)
    # States: 0=bull, 1=bear, 2=sideways, 3=high_vol
    regime_params = {
        0: (0.15 / 252, 0.18 / np.sqrt(252)),   # bull
        1: (-0.10 / 252, 0.28 / np.sqrt(252)),   # bear
        2: (0.03 / 252, 0.14 / np.sqrt(252)),    # sideways
        3: (0.00 / 252, 0.40 / np.sqrt(252)),    # high vol
    }
    transition = np.array([
        [0.98, 0.008, 0.008, 0.004],  # bull stays bull 98%
        [0.008, 0.975, 0.012, 0.005], # bear stays bear 97.5%
        [0.010, 0.010, 0.975, 0.005], # sideways stays 97.5%
        [0.015, 0.015, 0.020, 0.950], # high vol stays 95%
    ])

    # Symbol-specific starting price
    sym_hash = hash(symbol) % 1000
    if "CRYPTO:" in symbol or "/USD" in symbol:
        base_price = 0.5 + sym_hash * 100  # crypto range
    elif "FX:" in symbol:
        base_price = 0.7 + (sym_hash % 100) / 100  # forex range ~0.7-1.7
    elif symbol in ("SPY", "QQQ", "IWM", "DIA"):
        base_price = 300 + sym_hash % 200
    else:
        base_price = 20 + sym_hash % 400  # equity range

    # Generate regime sequence
    regimes = np.zeros(n_days, dtype=int)
    regimes[0] = rng.choice(4, p=[0.4, 0.15, 0.35, 0.10])  # initial regime
    for i in range(1, n_days):
        regimes[i] = rng.choice(4, p=transition[regimes[i - 1]])

    # Generate returns
    closes = np.zeros(n_days)
    closes[0] = base_price
    for i in range(1, n_days):
        mu, sigma = regime_params[regimes[i]]
        ret = mu + sigma * rng.standard_normal()
        # Add mean-reversion pressure to prevent drift to 0 or infinity
        if closes[i - 1] < base_price * 0.5:
            ret += 0.005
        elif closes[i - 1] > base_price * 2.0:
            ret -= 0.005
        closes[i] = closes[i - 1] * (1 + ret)

    # Generate OHLV from close
    daily_vol = np.array([regime_params[r][1] for r in regimes])
    opens = closes * (1 + rng.normal(0, daily_vol * 0.3, n_days))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, daily_vol * 0.5, n_days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, daily_vol * 0.5, n_days)))

    # Volume: higher in high-vol regimes
    base_vol = 1_000_000 + rng.integers(0, 5_000_000)
    vol_mult = np.where(regimes == 3, 2.5, np.where(regimes == 1, 1.5, 1.0))
    volumes = (base_vol * vol_mult * (1 + rng.normal(0, 0.3, n_days))).astype(int)
    volumes = np.maximum(volumes, 10000)

    # Build date index (business days)
    end_date = datetime(2026, 3, 13)
    dates = pd.bdate_range(end=end_date, periods=n_days)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)

    # Ensure OHLC consistency
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    return df


def patch_market_data():
    """Monkey-patch MarketDataFetcher to return synthetic data."""
    from data.market_data import MarketDataFetcher

    _cache = {}

    def fetch_historical_data(self, symbol, days=252, use_cache=True):
        cache_key = f"{symbol}_{days}"
        if cache_key in _cache:
            return _cache[cache_key].copy()
        df = generate_synthetic_ohlcv(symbol, n_days=max(days, 504))
        if days < len(df):
            df = df.iloc[-days:]
        _cache[cache_key] = df
        return df.copy()

    MarketDataFetcher.fetch_historical_data = fetch_historical_data
    print("[SYNTHETIC] MarketDataFetcher patched — generating synthetic OHLCV data")


if __name__ == "__main__":
    # Patch before importing the backtest
    patch_market_data()

    # Now run the actual backtest
    from scripts.god_level_backtest import GodLevelBacktester, print_results, go_no_go_assessment
    from config import ApexConfig

    print("\n" + "=" * 70)
    print("STARTING GOD LEVEL BACKTEST (SYNTHETIC DATA)")
    print("=" * 70)

    backtester = GodLevelBacktester(initial_capital=100000)

    # Use "core" session to activate the relaxed thresholds designed for the
    # split-strategy architecture.  The "unified" mode uses legacy thresholds
    # (signal >= 0.34-0.55) that filter out nearly all signals and produce
    # too few trades (<50) for any statistical significance.
    result = backtester.run_backtest(
        symbols=ApexConfig.SYMBOLS[:40],
        walk_forward=True,
        session_type="core",
    )

    if result:
        # Run Monte Carlo
        mc_results = backtester.run_monte_carlo(n_simulations=1000)
        result.monte_carlo = mc_results

        # Run Phase 3 stress tests
        stress_results = backtester.run_all_stress_tests(
            sharpe_ratio=result.sharpe_ratio,
            n_trades=result.total_trades,
        )
        result.deflated_sharpe = stress_results.get("deflated_sharpe")
        result.stress_tests = stress_results

        # Print results
        print_results(result, stress_results=stress_results)

        # Phase 4: GO/NO-GO
        go_result = go_no_go_assessment(result, stress_results=stress_results)

        # ── Extra diagnostics the user requested ──

        # Walk-forward per-fold info
        print("\n" + "=" * 70)
        print("WALK-FORWARD FOLD DETAILS")
        print("=" * 70)
        if hasattr(backtester, '_fold_results') and backtester._fold_results:
            print(f"{'Fold':>4} | {'Train Period':>25} | {'Test Period':>25} | {'OOS Sharpe':>10} | {'MaxDD':>8} | {'#Trades':>7}")
            print("-" * 95)
            for f in backtester._fold_results:
                print(f"{f['fold']:4d} | {f['train_period']:>25} | {f['test_period']:>25} | {f['oos_sharpe']:10.2f} | {f['max_dd']:7.1f}% | {f['n_trades']:7d}")
        else:
            print("  Per-fold tracking not available in current implementation.")
            print("  Aggregate walk-forward results used (all folds combined).")

        # Regime breakdown
        print("\n" + "=" * 70)
        print("REGIME BREAKDOWN")
        print("=" * 70)
        if result.regime_performance:
            print(f"{'Regime':<20} | {'Win Rate':>8} | {'Avg Trade':>10} | {'Total P&L':>12} | {'#Trades':>7}")
            print("-" * 70)
            for regime, perf in sorted(result.regime_performance.items()):
                print(f"{regime:<20} | {perf['win_rate']:7.1f}% | ${perf['avg_pnl']:9.2f} | ${perf['total_pnl']:11,.0f} | {perf['trades']:7d}")

        # Cost impact
        print("\n" + "=" * 70)
        print("COST IMPACT ANALYSIS")
        print("=" * 70)
        if result.trades:
            total_commission = sum(getattr(t, 'entry_commission', 0) for t in result.trades)
            # Approximate: commission is typically tracked per-trade
            # Gross PnL = sum of all trade PnLs + total costs
            total_pnl = sum(t.pnl for t in result.trades)
            avg_cost = total_commission / max(len(result.trades), 1)
            print(f"  Net Sharpe (reported):    {result.sharpe_ratio:.2f}")
            print(f"  Total trades:             {len(result.trades)}")
            print(f"  Total P&L:                ${total_pnl:,.2f}")
            print(f"  Avg cost per trade:       ${avg_cost:.2f}")
            print("  (Gross Sharpe not separable — costs embedded in execution prices)")

        # ICIR — cannot compute without forward return correlation data
        print("\n" + "=" * 70)
        print("TOP 10 SIGNALS BY ICIR")
        print("=" * 70)
        print("  ICIR requires: corr(signal_t, return_{t+1}) computed per signal component.")
        print("  The current architecture does not log per-component signal values alongside")
        print("  forward returns during backtest execution. Components are blended before")
        print("  the trade decision point (god_level_signal_generator.py:787-818).")
        print("  DATA MISSING: Per-component signal logs not captured during walk-forward.")
        print("  RECOMMENDATION: Add signal component logging to backtest loop to enable ICIR.")

    else:
        print("\n  BACKTEST FAILED — no result produced.")
        print("  Check data availability and symbol universe.")
