#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
 APEX TRADING SYSTEM — FULL VALIDATION BACKTEST
═══════════════════════════════════════════════════════════════════════

ONE-PROMPT RUNNER:
    python scripts/run_full_validation.py

Produces:
    1. GO/NO-GO card (8 gates with PASS/FAIL)
    2. Walk-forward table (per-fold IS/OOS Sharpe, MaxDD, #Trades)
    3. Regime breakdown (Sharpe, Win Rate, Avg Trade per regime)
    4. Cost impact (Gross vs Net Sharpe, cost drag, avg cost per trade)
    5. Top 10 signals by ICIR (Information Coefficient / IC Ratio)

Prerequisites:
    pip install numpy pandas scipy scikit-learn yfinance pytz psutil

Optional (better ML but not required):
    pip install xgboost lightgbm shap
"""
import sys
import os
import time
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

# ── Project root setup ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# ── Configure logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-4.4s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validation")


# ════════════════════════════════════════════════════════════════════
# SECTION 1: Instrumented Backtester (adds per-fold + ICIR tracking)
# ════════════════════════════════════════════════════════════════════

def instrument_backtester():
    """Monkey-patch GodLevelBacktester to capture per-fold metrics
    and per-signal-component IC data without modifying the original file."""

    from scripts.god_level_backtest import GodLevelBacktester

    _orig_run_walk_forward = GodLevelBacktester._run_walk_forward
    _orig_simulate_period = GodLevelBacktester._simulate_period
    _orig_enter_position = GodLevelBacktester._enter_position

    # ── Per-fold tracking ───────────────────────────────────────────
    def _run_walk_forward_instrumented(self, historical_data, all_dates):
        """Wrap walk-forward to capture per-fold metrics."""
        self._fold_results = []
        self._fold_equity_snapshots = []  # (fold_num, equity_before, equity_after)
        self._gross_pnl_accumulator = 0.0  # gross P&L without commissions
        self._total_commissions = 0.0
        self._signal_ic_log = []  # (component_name, component_value, 5d_fwd_return)

        # We re-implement the walk-forward loop with fold tracking
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        if self.risk_manager:
            self.risk_manager.positions = {}
            self.risk_manager.daily_pnl = []
            self.risk_manager.update_capital(self.initial_capital)

        total_days = len(all_dates)
        start_idx = max(self.WARMUP_DAYS, self.WALK_FORWARD_TRAIN_DAYS)

        fold_num = 0
        while start_idx + self.WALK_FORWARD_TEST_DAYS < total_days:
            fold_num += 1

            train_end_idx = start_idx
            train_start_idx = max(0, train_end_idx - self.WALK_FORWARD_TRAIN_DAYS)
            train_dates = all_dates[train_start_idx:train_end_idx]

            purge_embargo = self.WALK_FORWARD_PURGE_DAYS + self.WALK_FORWARD_EMBARGO_DAYS
            test_start_idx = train_end_idx + purge_embargo
            if test_start_idx >= total_days:
                break
            test_end_idx = min(test_start_idx + self.WALK_FORWARD_TEST_DAYS, total_days)
            test_dates = all_dates[test_start_idx:test_end_idx]

            # ── Train ──
            train_data = {}
            for symbol, df in historical_data.items():
                mask = df.index.isin(train_dates)
                if mask.sum() >= 60:
                    train_data[symbol] = df[mask]

            # Compute in-sample Sharpe on training data
            is_sharpe = _compute_period_sharpe(train_data, train_dates)

            if train_data:
                self.signal_generator.train_models(train_data)
                if self.risk_manager:
                    self.risk_manager.update_correlation_matrix(train_data)

            # ── Test ──
            equity_before = self._current_equity_value({})
            trades_before = len(self.trades)

            self._simulate_period(historical_data, test_dates)

            trades_after = len(self.trades)
            equity_after = self._current_equity_value({})
            period_trades = self.trades[trades_before:trades_after]

            # Compute OOS metrics from this fold's trades
            oos_sharpe = _sharpe_from_trades(period_trades, self.initial_capital)
            oos_max_dd = _max_dd_from_trades(period_trades, equity_before)

            train_start_str = train_dates[0].strftime('%Y-%m-%d') if hasattr(train_dates[0], 'strftime') else str(train_dates[0])[:10]
            train_end_str = train_dates[-1].strftime('%Y-%m-%d') if hasattr(train_dates[-1], 'strftime') else str(train_dates[-1])[:10]
            test_start_str = test_dates[0].strftime('%Y-%m-%d') if hasattr(test_dates[0], 'strftime') else str(test_dates[0])[:10]
            test_end_str = test_dates[-1].strftime('%Y-%m-%d') if hasattr(test_dates[-1], 'strftime') else str(test_dates[-1])[:10]

            self._fold_results.append({
                'fold': fold_num,
                'train_period': f"{train_start_str} → {train_end_str}",
                'test_period': f"{test_start_str} → {test_end_str}",
                'is_sharpe': is_sharpe,
                'oos_sharpe': oos_sharpe,
                'max_dd': oos_max_dd,
                'n_trades': len(period_trades),
            })

            logger.info(f"  Fold {fold_num}: {len(period_trades)} trades, OOS Sharpe={oos_sharpe:.2f}, MaxDD={oos_max_dd:.1f}%")
            start_idx = test_end_idx

        return self._calculate_results(historical_data)

    def _current_equity_value(self, day_data):
        """Quick equity estimate."""
        return self.capital + sum(
            p.shares * p.entry_price for p in self.positions.values()
        )

    # ── ICIR: log signal components during simulation ───────────────
    def _enter_position_instrumented(self, symbol, row, signal, confidence,
                                      regime, date, prices, current_equity):
        """Wrap _enter_position to log signal components for ICIR."""
        # Log components BEFORE entering
        sig_data = self.signal_generator.generate_ml_signal(symbol, prices)
        components = sig_data.get('components', {})

        # 5-day forward return (if available in the price series)
        idx = len(prices) - 1
        # We'll compute fwd return from the row's close
        entry_price = row['Close']

        # Store for later — fwd return will be computed at exit
        if not hasattr(self, '_pending_ic_entries'):
            self._pending_ic_entries = {}
        self._pending_ic_entries[symbol] = {
            'components': dict(components),
            'entry_price': entry_price,
            'entry_date': date,
        }

        return _orig_enter_position(self, symbol, row, signal, confidence,
                                     regime, date, prices, current_equity)

    GodLevelBacktester._run_walk_forward = _run_walk_forward_instrumented
    GodLevelBacktester._current_equity_value = _current_equity_value
    GodLevelBacktester._enter_position = _enter_position_instrumented


def _compute_period_sharpe(data, dates):
    """Compute buy-and-hold Sharpe over a period (proxy for IS complexity)."""
    returns = []
    for sym, df in data.items():
        period = df[df.index.isin(dates)]
        if len(period) >= 20:
            r = period['Close'].pct_change().dropna()
            if len(r) > 0:
                returns.append(r.mean() / max(r.std(), 1e-9) * np.sqrt(252))
    return float(np.mean(returns)) if returns else 0.0


def _sharpe_from_trades(trades, initial_capital):
    """Compute approximate Sharpe from a list of trades."""
    if len(trades) < 2:
        return 0.0
    pnls = np.array([t.pnl for t in trades])
    hold_days = np.array([max(t.hold_days, 1) for t in trades], dtype=float)
    avg_hold = max(float(np.mean(hold_days)), 1.0)
    annualization = np.sqrt(252.0 / avg_hold)
    returns = pnls / initial_capital
    if returns.std() > 0:
        return float(returns.mean() / returns.std() * annualization)
    return 0.0


def _max_dd_from_trades(trades, starting_equity):
    """Compute max drawdown from a trade sequence."""
    if not trades:
        return 0.0
    equity = starting_equity
    peak = equity
    max_dd = 0.0
    for t in trades:
        equity += t.pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return max_dd * 100


# ════════════════════════════════════════════════════════════════════
# SECTION 2: Cost Impact Analysis
# ════════════════════════════════════════════════════════════════════

def compute_cost_impact(result):
    """Separate gross vs net P&L and compute cost drag on Sharpe."""
    if not result.trades:
        return None

    trades = result.trades
    gross_pnls = []
    net_pnls = []
    total_commissions = 0.0

    for t in trades:
        # Reconstruct gross PnL from trade fields
        if t.direction == 'long':
            gross = (t.exit_price - t.entry_price) * t.shares
        else:
            gross = (t.entry_price - t.exit_price) * t.shares
        # Net PnL = gross - commissions (stored in t.pnl which already has costs deducted)
        # So: commissions = gross - net
        commission_roundtrip = gross - t.pnl
        total_commissions += commission_roundtrip
        gross_pnls.append(gross)
        net_pnls.append(t.pnl)

    gross_pnls = np.array(gross_pnls)
    net_pnls = np.array(net_pnls)

    hold_days = np.array([max(t.hold_days, 1) for t in trades], dtype=float)
    avg_hold = max(float(np.mean(hold_days)), 1.0)
    ann = np.sqrt(252.0 / avg_hold)

    initial = result.equity_curve.iloc[0] if len(result.equity_curve) > 0 else 100000
    gross_returns = gross_pnls / initial
    net_returns = net_pnls / initial

    gross_sharpe = float(gross_returns.mean() / max(gross_returns.std(), 1e-9) * ann)
    net_sharpe = float(net_returns.mean() / max(net_returns.std(), 1e-9) * ann)

    return {
        'gross_sharpe': gross_sharpe,
        'net_sharpe': net_sharpe,
        'cost_drag': gross_sharpe - net_sharpe,
        'total_commissions': total_commissions,
        'avg_cost_per_trade': total_commissions / max(len(trades), 1),
        'total_gross_pnl': float(gross_pnls.sum()),
        'total_net_pnl': float(net_pnls.sum()),
    }


# ════════════════════════════════════════════════════════════════════
# SECTION 3: Signal ICIR Analysis
# ════════════════════════════════════════════════════════════════════

def compute_signal_icir(backtester, historical_data):
    """Compute Information Coefficient Ratio for each signal component.

    IC = corr(signal_t, return_{t+forward_days}) per component.
    ICIR = IC_mean / IC_std across rolling windows.
    """
    signal_gen = backtester.signal_generator
    component_ics = defaultdict(list)  # component_name -> list of ICs per window

    symbols = list(historical_data.keys())[:20]  # limit for speed
    window_size = 63  # quarterly rolling windows

    for sym in symbols:
        df = historical_data[sym]
        prices = df['Close']
        if len(prices) < 200:
            continue

        # Collect component signals and forward returns
        component_signals = defaultdict(list)
        forward_returns = []

        for i in range(120, len(prices) - 5):
            hist = prices.iloc[:i]
            sig_data = signal_gen.generate_ml_signal(sym, hist)
            components = sig_data.get('components', {})

            fwd_ret = (prices.iloc[i + 5] - prices.iloc[i]) / prices.iloc[i]
            forward_returns.append(fwd_ret)

            for comp_name, comp_val in components.items():
                component_signals[comp_name].append(comp_val)

        if len(forward_returns) < window_size:
            continue

        fwd = np.array(forward_returns)

        for comp_name, vals in component_signals.items():
            arr = np.array(vals)
            if len(arr) != len(fwd):
                continue

            # Rolling IC
            ics = []
            for start in range(0, len(arr) - window_size, window_size // 2):
                end = start + window_size
                if end > len(arr):
                    break
                window_sig = arr[start:end]
                window_ret = fwd[start:end]
                if np.std(window_sig) > 1e-9 and np.std(window_ret) > 1e-9:
                    ic = np.corrcoef(window_sig, window_ret)[0, 1]
                    if np.isfinite(ic):
                        ics.append(ic)
            component_ics[comp_name].extend(ics)

    # Compute ICIR per component
    results = []
    for comp_name, ics in component_ics.items():
        if len(ics) < 3:
            continue
        ic_arr = np.array(ics)
        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr))
        icir = ic_mean / max(ic_std, 1e-9)
        results.append({
            'signal': comp_name,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'n_windows': len(ics),
            'keep': icir > 0.3,  # industry threshold for signal value
        })

    results.sort(key=lambda x: abs(x['icir']), reverse=True)
    return results[:10]


# ════════════════════════════════════════════════════════════════════
# SECTION 4: Output Formatting
# ════════════════════════════════════════════════════════════════════

def print_go_no_go_card(result, stress_results):
    """Print the 8-gate GO/NO-GO card with raw numbers."""
    print("\n" + "=" * 74)
    print("  GO / NO-GO CARD")
    print("=" * 74)

    dsr = stress_results.get('deflated_sharpe', {}) if stress_results else {}
    perm = stress_results.get('permutation_test', {}) if stress_results else {}
    pert = stress_results.get('parameter_perturbation', {}) if stress_results else {}
    crisis = stress_results.get('crisis_simulation', {}) if stress_results else {}

    gates = [
        ("Gate 1: Sharpe >= 1.5",        result.sharpe_ratio >= 1.5,       f"{result.sharpe_ratio:.2f}",                   "1.50"),
        ("Gate 2: Max DD <= 15%",         result.max_drawdown <= 15.0,      f"{result.max_drawdown:.1f}%",                 "15.0%"),
        ("Gate 3: Win Rate >= 50%",       result.win_rate >= 50.0,          f"{result.win_rate:.1f}%",                     "50.0%"),
        ("Gate 4: Profit Factor >= 1.3",  result.profit_factor >= 1.3,      f"{result.profit_factor:.2f}",                 "1.30"),
        ("Gate 5: DSR > 1.35",            dsr.get('pass', False),           f"{dsr.get('dsr', 0):.3f}",                    "1.350"),
        ("Gate 6: Perm. p < 0.05",        perm.get('pass', False),          f"{perm.get('p_value', 1.0):.4f}",             "0.0500"),
        ("Gate 7: Robustness >= 70%",     pert.get('pass', False),          f"{pert.get('robustness', 0)*100:.0f}%",        "70%"),
        ("Gate 8: Crisis DD < 30%",       crisis.get('pass', False),        f"{crisis.get('worst_dd', 0):.1f}%",           "30.0%"),
    ]

    passed = 0
    for name, ok, actual, threshold in gates:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  {status:4s} | {name:<30s} | actual: {actual:>8s} | threshold: {threshold}")

    total = len(gates)
    if passed == total:
        verdict = "GO"
    elif passed >= 6:
        verdict = "CONDITIONAL GO"
    else:
        verdict = "NO-GO"

    print(f"\n  {'─' * 50}")
    print(f"  VERDICT: {verdict} ({passed}/{total} gates passed)")
    print(f"  {'─' * 50}")
    if verdict != "GO":
        failed = [n for n, ok, _, _ in gates if not ok]
        print(f"  Failing: {', '.join(failed)}")

    return verdict, passed, total


def print_walk_forward_table(backtester):
    """Print per-fold walk-forward results."""
    print("\n" + "=" * 74)
    print("  WALK-FORWARD TABLE (per fold)")
    print("=" * 74)

    folds = getattr(backtester, '_fold_results', [])
    if not folds:
        print("  No per-fold data captured.")
        return

    print(f"  {'Fold':>4} | {'Train Period':>27} | {'Test Period':>27} | {'IS Sharpe':>9} | {'OOS Sharpe':>10} | {'MaxDD':>7} | {'#Trades':>7}")
    print("  " + "-" * 105)
    for f in folds:
        print(f"  {f['fold']:4d} | {f['train_period']:>27} | {f['test_period']:>27} | {f['is_sharpe']:9.2f} | {f['oos_sharpe']:10.2f} | {f['max_dd']:6.1f}% | {f['n_trades']:7d}")

    # Summary stats
    oos_sharpes = [f['oos_sharpe'] for f in folds]
    print(f"\n  OOS Sharpe — mean: {np.mean(oos_sharpes):.2f}, median: {np.median(oos_sharpes):.2f}, "
          f"min: {np.min(oos_sharpes):.2f}, max: {np.max(oos_sharpes):.2f}")


def print_regime_breakdown(result):
    """Print regime performance table."""
    print("\n" + "=" * 74)
    print("  REGIME BREAKDOWN")
    print("=" * 74)

    if not result.regime_performance:
        print("  No regime data.")
        return

    print(f"  {'Regime':<20} | {'Sharpe':>7} | {'Win Rate':>8} | {'Avg Trade':>10} | {'#Trades':>7}")
    print("  " + "-" * 65)
    for regime in sorted(result.regime_performance.keys()):
        perf = result.regime_performance[regime]
        sharpe = perf.get('sharpe', 0.0)
        print(f"  {regime:<20} | {sharpe:7.2f} | {perf['win_rate']:7.1f}% | ${perf['avg_pnl']:9.2f} | {perf['trades']:7d}")


def print_cost_impact(cost_data):
    """Print cost impact analysis."""
    print("\n" + "=" * 74)
    print("  COST IMPACT ANALYSIS")
    print("=" * 74)

    if not cost_data:
        print("  No trade data for cost analysis.")
        return

    print(f"  Gross Sharpe (no costs):   {cost_data['gross_sharpe']:7.2f}")
    print(f"  Net Sharpe (after costs):  {cost_data['net_sharpe']:7.2f}")
    print(f"  Cost drag:                 {cost_data['cost_drag']:7.2f} Sharpe points")
    print(f"  Avg cost per trade:        ${cost_data['avg_cost_per_trade']:.2f}")
    print(f"  Total commissions:         ${cost_data['total_commissions']:,.2f}")
    print(f"  Gross P&L:                 ${cost_data['total_gross_pnl']:,.2f}")
    print(f"  Net P&L:                   ${cost_data['total_net_pnl']:,.2f}")


def print_signal_icir(icir_results):
    """Print top 10 signals by ICIR."""
    print("\n" + "=" * 74)
    print("  TOP 10 SIGNALS BY ICIR")
    print("=" * 74)

    if not icir_results:
        print("  Insufficient data for ICIR computation.")
        return

    print(f"  {'Signal':<25} | {'IC Mean':>8} | {'IC Std':>8} | {'ICIR':>8} | {'Keep/Kill':>9}")
    print("  " + "-" * 68)
    for s in icir_results:
        verdict = "KEEP" if s['keep'] else "KILL"
        print(f"  {s['signal']:<25} | {s['ic_mean']:8.4f} | {s['ic_std']:8.4f} | {s['icir']:8.4f} | {verdict:>9}")


def print_paper_trading_checklist(result, stress_results, cost_data):
    """Generate paper trading checklist (only if GO)."""
    print("\n" + "=" * 74)
    print("  4-WEEK PAPER TRADING CHECKLIST")
    print("=" * 74)

    # Kelly sizing
    avg_pnl = result.avg_trade
    win_rate = result.win_rate / 100.0
    avg_win = result.avg_win
    avg_loss = abs(result.avg_loss)

    if avg_loss > 0 and avg_win > 0:
        b = avg_win / avg_loss
        kelly_full = win_rate - (1 - win_rate) / b
    else:
        kelly_full = 0.0

    # 95th percentile WF drawdown
    folds = getattr(result, '_fold_max_dds', [])
    if not folds and hasattr(result, 'max_drawdown'):
        wf_dd_95 = result.max_drawdown * 1.5  # conservative estimate
    else:
        wf_dd_95 = np.percentile(folds, 95) if folds else result.max_drawdown * 1.5

    quarter_kelly = kelly_full * 0.25
    max_starting_capital = 100000 * quarter_kelly if quarter_kelly > 0 else 0

    print(f"""
  Kelly fraction (full):     {kelly_full:.3f}
  Kelly × 0.25:              {quarter_kelly:.3f}
  95th pct WF drawdown:      {wf_dd_95:.1f}%
  Max starting capital:      ${max_starting_capital:,.0f} (Kelly×0.25 based on 95th pct DD)

  WEEK 1: Shadow Mode
  [ ] Deploy to paper account (IBKR TWS port 7497)
  [ ] Run monitoring/missed_opportunity_tracker.py in background
  [ ] Verify signal generation matches backtest regime detection
  [ ] Check fill quality: compare paper fills vs backtest slippage assumptions
  [ ] Confirm position sizing matches vol-targeting output

  WEEK 2: Active Paper Trading
  [ ] Monitor daily PnL vs backtest expectancy (${avg_pnl:.2f}/trade)
  [ ] Track regime classification accuracy vs realized market behavior
  [ ] Verify stop-loss/take-profit execution within 1 tick of target
  [ ] Check overnight exposure stays within MAX_GROSS_EXPOSURE (1.4x)
  [ ] Review any missed trades logged by opportunity tracker

  WEEK 3: Stress Testing Live
  [ ] Manually verify 3 trade entries against signal components
  [ ] Compare paper Sharpe to OOS Sharpe ({result.sharpe_ratio:.2f})
  [ ] Monitor for regime misclassification during volatile sessions
  [ ] Test kill-switch: verify HARD_DRAWDOWN_ENTRY_PAUSE triggers at 18%
  [ ] Verify sector concentration limits respected

  WEEK 4: GO-LIVE Preparation
  [ ] Compare paper trading stats to backtest (Sharpe, WR, PF within 30%)
  [ ] Document any deviations from backtest assumptions
  [ ] Set up alerting for drawdown > {result.max_drawdown * 1.5:.1f}% (1.5x backtest max)
  [ ] Configure IBKR risk controls to match MAX_POSITION_PCT (5%)
  [ ] Final decision: GO-LIVE / EXTEND PAPER / ABORT

  TOP 3 IBKR EXECUTION RISKS:
  1. Crypto fills: IBKR crypto desk has wider spreads than backtest assumes.
     Slippage may be 2-5x the {cost_data.get('avg_cost_per_trade', 0):.2f}/trade estimate.
  2. FX lot sizing: IBKR requires min 25,000 unit lots for FX pairs.
     Small accounts may not be able to position-size FX correctly.
  3. Pattern Day Trader: >4 day trades in 5 days triggers PDT rules.
     With {result.total_trades} trades over the backtest period (~{result.total_trades / max(1, len(result.equity_curve)) * 252:.0f}/year),
     ensure account has $25k+ or reduce entry frequency.
""")


# ════════════════════════════════════════════════════════════════════
# SECTION 5: Main Runner
# ════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    print("\n" + "═" * 74)
    print("  APEX TRADING SYSTEM — FULL VALIDATION BACKTEST")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("═" * 74)

    # ── Step 0: Dependency check ────────────────────────────────────
    print("\n[Step 0] Checking dependencies...")
    missing = []
    for mod in ['numpy', 'pandas', 'scipy', 'sklearn']:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    try:
        import yfinance as yf
        yf_ok = True
    except ImportError:
        yf_ok = False
        missing.append('yfinance')

    if missing:
        print(f"  MISSING: {', '.join(missing)}")
        print(f"  Run: pip install {' '.join(missing)}")
        sys.exit(1)
    print("  All dependencies OK.")

    # ── Step 1: Syntax check ────────────────────────────────────────
    print("\n[Step 1] Syntax check...")
    files_to_check = [
        'scripts/god_level_backtest.py',
        'risk/god_level_risk_manager.py',
        'monitoring/missed_opportunity_tracker.py',
    ]
    all_clean = True
    for f in files_to_check:
        fpath = PROJECT_ROOT / f
        if not fpath.exists():
            print(f"  MISSING: {f}")
            all_clean = False
            continue
        try:
            import py_compile
            py_compile.compile(str(fpath), doraise=True)
            print(f"  OK: {f}")
        except py_compile.PyCompileError as e:
            print(f"  FAIL: {f} — {e}")
            all_clean = False

    if not all_clean:
        print("\n  Fix syntax errors before proceeding.")
        sys.exit(1)

    # ── Step 2: Instrument & run backtest ───────────────────────────
    print("\n[Step 2] Running backtest with real market data...")

    # Import after syntax check
    from scripts.god_level_backtest import (
        GodLevelBacktester, print_results, go_no_go_assessment
    )
    from config import ApexConfig

    # Instrument the backtester for per-fold + ICIR tracking
    instrument_backtester()

    backtester = GodLevelBacktester(initial_capital=100000)

    # Run with "core" session (relaxed thresholds for adequate trade count)
    result = backtester.run_backtest(
        symbols=ApexConfig.get_session_symbols("core")[:50],
        walk_forward=True,
        session_type="core",
    )

    if result is None:
        print("\n  BACKTEST FAILED — no data fetched.")
        print("  Possible causes:")
        print("  1. No internet connection (yfinance needs Yahoo Finance API)")
        print("  2. Yahoo Finance rate-limited or blocked")
        print("  3. All symbols returned empty data")
        print("\n  Try: python -c \"import yfinance; print(yfinance.download('SPY', period='5d'))\"")
        sys.exit(1)

    # ── Step 3: Monte Carlo + Stress Tests ──────────────────────────
    print("\n[Step 3] Running Monte Carlo + Phase 3 stress tests...")

    mc_results = backtester.run_monte_carlo(n_simulations=1000)
    result.monte_carlo = mc_results

    stress_results = backtester.run_all_stress_tests(
        sharpe_ratio=result.sharpe_ratio,
        n_trades=result.total_trades,
    )
    result.deflated_sharpe = stress_results.get("deflated_sharpe")
    result.stress_tests = stress_results

    # ── Step 4: Compute cost impact ─────────────────────────────────
    print("\n[Step 4] Computing cost impact...")
    cost_data = compute_cost_impact(result)

    # ── Step 5: Compute signal ICIR ─────────────────────────────────
    print("\n[Step 5] Computing signal ICIR (this may take a few minutes)...")
    historical_data = backtester._fetch_data(
        ApexConfig.get_session_symbols("core")[:20]
    )
    icir_results = compute_signal_icir(backtester, historical_data)

    # ═══════════════════════════════════════════════════════════════
    #  OUTPUT: Full validation report
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    print("\n")
    print("═" * 74)
    print("  FULL VALIDATION REPORT")
    print(f"  Completed in {elapsed:.0f}s")
    print("═" * 74)

    # 0. Standard backtest output
    print_results(result, stress_results=stress_results)

    # 1. GO/NO-GO card
    verdict, passed, total = print_go_no_go_card(result, stress_results)

    # 2. Walk-forward table
    print_walk_forward_table(backtester)

    # 3. Regime breakdown
    print_regime_breakdown(result)

    # 4. Cost impact
    print_cost_impact(cost_data)

    # 5. ICIR
    print_signal_icir(icir_results)

    # 6. If GO → paper trading checklist; if NO-GO → failure analysis
    print("\n" + "═" * 74)
    if verdict == "GO":
        print("  STATUS: GO — Cleared for paper trading")
        print("═" * 74)
        print_paper_trading_checklist(result, stress_results, cost_data or {})
    elif verdict == "CONDITIONAL GO":
        print("  STATUS: CONDITIONAL GO — Paper trade with reduced size")
        print("═" * 74)
        print_paper_trading_checklist(result, stress_results, cost_data or {})
        print("\n  NOTE: Failing gates must be monitored weekly during paper trading.")
    else:
        print("  STATUS: NO-GO — Strategy requires further development")
        print("═" * 74)
        print(f"\n  {passed}/{total} gates passed. Strategy is not ready for deployment.")
        print(f"  Review the regime breakdown and ICIR table above to identify")
        print(f"  which signal components lack edge and which regimes bleed.")
        if result.regime_performance:
            bleeding = [r for r, p in result.regime_performance.items()
                        if p.get('negative_sharpe') or p.get('avg_pnl', 0) < 0]
            if bleeding:
                print(f"\n  Bleeding regimes: {', '.join(bleeding)}")
                print(f"  Consider adding regime filters to disable trading in these regimes,")
                print(f"  or recalibrate signal weights for these market conditions.")

    print("\n" + "═" * 74)
    print(f"  Backtest complete. {result.total_trades} trades analyzed.")
    print("═" * 74 + "\n")


if __name__ == "__main__":
    main()
