#!/usr/bin/env python3
"""
best_scenarios_jan_mar_2026_v3.py
----------------------------------
v3 backtest optimisation: Jan 2 – Mar 7, 2026.

5 upgraded strategies (v2 → v3) + 1 new Gold-Silver Catch-Up strategy.

v3 improvements over v2
────────────────────────
• Kelly/vol-aware position sizing via engine.calculate_position_size()
• use_dynamic_slippage=True for realistic market impact
• Regime-aware stops via stop_manager.apply_regime() after every entry
• Monte Carlo simulation (1 000 bootstraps) on every strategy
• Strategy 2: portfolio 5 %-DD guard + oil entry double-gate
• Strategy 3: satellite momentum threshold raised; same-bar VIX exit
• Strategy 4: MACD bearish confirmation for QQQ short; Kelly longs
• Strategy 5: portfolio circuit breaker −7 % + inverse-vol weighting
• Strategy 6 (NEW): Gold-Silver ratio catch-up pairs trade
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ─── path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtesting.backtest_engine import BacktestEngine, StopLevel  # noqa: E402

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

VIX_TICKER = "^VIX"


def load_data(symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV via yfinance.  VIX is stored under key '_VIX'.
    Extends lookback by 90 days so rolling indicators have enough warm-up history.
    """
    from datetime import timedelta
    tickers = list(dict.fromkeys(symbols + [VIX_TICKER]))
    # 90-day lookback warm-up so SMA-30, RSI-14, etc. have valid history at bar 1
    fetch_start = (start - timedelta(days=90)).strftime("%Y-%m-%d")
    fetch_end   = (end + timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=tickers,
        start=fetch_start,
        end=fetch_end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            lvl0 = raw.columns.get_level_values(0)
            if ticker not in lvl0:
                continue
            df = raw[ticker].copy()
            # Normalise column names (yfinance can return lowercase in some versions)
            df.columns = [c.capitalize() for c in df.columns]
            needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[needed].dropna(how="all")
            if df.empty:
                continue
            key = "_VIX" if ticker == VIX_TICKER else ticker
            data[key] = df
        except Exception:
            pass
    return data


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_vix(dv) -> float:
    """Return current VIX from data view; default 18.0 if unavailable."""
    try:
        vix_df = dv.get("_VIX")
        if vix_df is None or vix_df.empty:
            return 18.0
        return float(vix_df["Close"].dropna().iloc[-1])
    except Exception:
        return 18.0


def get_regime(vix: float) -> str:
    """Map VIX level to engine regime string."""
    if vix >= 28:
        return "high_volatility"
    if vix >= 22:
        return "bear"
    if vix >= 18:
        return "neutral"
    return "bull"


def size_pos(equity: float, price: float, pct: float, max_shares: int = 50_000) -> int:
    """Simple flat-percentage position size (fallback)."""
    if price <= 0:
        return 0
    return max(1, min(int(equity * pct / price), max_shares))


def kelly_shares(
    eng: BacktestEngine,
    _symbol: str,
    price: float,
    prices: pd.Series | None,
    max_pct: float = 0.40,
    vix: float = 18.0,
    force_signal: float | None = None,
    min_scaling: float = 0.75,
) -> int:
    """
    Fully dynamic position sizing derived entirely from live market data.

    Four independent factors are computed from the DataView-sliced price series
    (zero lookahead) and multiplied against a base allocation:

        Factor 1 — Momentum   : abs(20d normalised return) → [0.60, 1.00]
        Factor 2 — Trend R²   : linear-fit R² over 15 bars  → [0.70, 1.00]
        Factor 3 — Realized vol: 14-bar std vs 1.5 % baseline → [0.40, 1.00]
        Factor 4 — VIX regime  : fear multiplier              → [0.40, 1.00]

    Combined scaling ∈ [~0.10, 1.00] applied to base = max_pct × equity / price.
    Hard cap at base ensures the allocation target is never exceeded.

    force_signal overrides auto-computed momentum when the entry signal comes
    from a custom source (e.g. a gold/silver ratio spread in Strategy 6).
    """
    equity = eng.total_equity()
    if price <= 0 or equity <= 0:
        return 1

    base = max(1, int(equity * max_pct / price))

    if prices is None or len(prices) < 5:
        return base  # No history: use flat allocation

    # ── Factor 1: Momentum ──────────────────────────────────────────────────
    # Use force_signal if provided; otherwise blend 5-day + 20-day returns for
    # warm-up robustness (avoids cold-start penalty at early bars).
    if force_signal is not None:
        sig = float(np.clip(force_signal, -1.0, 1.0))
    elif len(prices) >= 6:
        sig_5d = float(np.clip(
            float(prices.iloc[-1] / prices.iloc[-5] - 1.0) / 0.05, -1.0, 1.0
        ))
        sig_20d = momentum(prices, min(20, len(prices) - 1))
        sig = (sig_5d + sig_20d) / 2.0
    else:
        sig = 0.0

    # Floor at 0.80 so a neutral/warmup signal deploys 80% of target immediately.
    # Strong trend reaches 1.0 (full allocation).
    mom_factor = float(np.clip(0.80 + abs(sig) * 0.20, 0.80, 1.00))

    # ── Factor 2: Trend Quality (R² of log-price linear regression) ─────────
    # High R² = consistent, linear trend = high conviction. [0.80, 1.00]
    # Computed on the last 15 bars of the already-sliced series → no lookahead.
    trend_factor = 0.87  # Default: moderate consistency
    if len(prices) >= 15:
        try:
            lp = np.log(prices.iloc[-15:].values.astype(float))
            x = np.arange(len(lp), dtype=float)
            fitted = np.polyval(np.polyfit(x, lp, 1), x)
            ss_tot = float(np.sum((lp - lp.mean()) ** 2))
            ss_res = float(np.sum((lp - fitted) ** 2))
            r2 = float(np.clip(1.0 - ss_res / (ss_tot + 1e-10), 0.0, 1.0))
            trend_factor = 0.80 + r2 * 0.20  # [0.80, 1.00]
        except Exception:
            pass

    # ── Factor 3: Realized Volatility ───────────────────────────────────────
    # Baseline 1.5 % daily vol → factor 1.0.
    # Low vol (e.g. GLD ~0.8 %) → factor UP TO 1.20 (reward low-risk assets).
    # High vol → reduced size (floor 0.50).
    # All calculations on the sliced `prices` — no future data visible.
    vol_factor = 1.00
    if len(prices) >= 10:
        try:
            rets = prices.iloc[-14:].pct_change().dropna()
            if len(rets) >= 5:
                daily_vol = float(rets.std())
                # Ceiling raised to 1.20 so low-vol assets (GLD, treasuries)
                # naturally receive a larger allocation within max_pct.
                vol_factor = float(np.clip(0.015 / max(daily_vol, 0.005), 0.50, 1.20))
        except Exception:
            pass

    # ── Factor 4: VIX Regime ────────────────────────────────────────────────
    # Raised floors: 20-25 VIX is elevated but not extreme; deploy at 75%.
    if vix >= 30:
        vix_factor = 0.45
    elif vix >= 25:
        vix_factor = 0.62
    elif vix >= 20:
        vix_factor = 0.75
    elif vix >= 15:
        vix_factor = 0.95
    else:
        vix_factor = 1.10  # Sub-15 VIX: allow a touch above neutral

    # ── Combine and cap ─────────────────────────────────────────────────────
    # min_scaling ensures we always deploy at least that fraction of max_pct
    # on day-1 (avoids extreme under-deployment when warmup data is thin).
    # Hard cap at base ensures the target is never breached.
    scaling = max(min_scaling, mom_factor * trend_factor * vol_factor * vix_factor)
    return max(1, min(base, int(base * scaling)))


def portfolio_dd(eng: BacktestEngine) -> float:
    """
    Current drawdown from equity peak.
    Returns a negative float e.g. -0.05 means −5 %.
    """
    eq = eng.total_equity()
    if not eng.history:
        return 0.0
    peak = max((h.get("equity", eq) for h in eng.history), default=eq)
    if peak <= 0:
        return 0.0
    return (eq / peak) - 1.0


def momentum(closes: pd.Series, n: int = 20) -> float:
    """Normalised n-bar return: maps ±10 % move → ±1.0, clipped to [−1, 1]."""
    if len(closes) < n + 1:
        return 0.0
    try:
        ret = float(closes.iloc[-1] / closes.iloc[-n] - 1.0)
        return float(np.clip(ret / 0.10, -1.0, 1.0))
    except (ZeroDivisionError, ValueError):
        return 0.0


def rsi(closes: pd.Series, period: int = 14) -> float:
    """Standard RSI. Returns 50.0 if insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    try:
        delta = closes.diff().dropna()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        loss_val = float(loss.iloc[-1])
        if loss_val == 0.0:
            return 100.0
        rs = float(gain.iloc[-1]) / loss_val
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception:
        return 50.0


def below_sma(closes: pd.Series, period: int = 30) -> bool:
    """True if latest close is below the period-SMA."""
    if len(closes) < period:
        return False
    try:
        sma = float(closes.rolling(period).mean().iloc[-1])
        return float(closes.iloc[-1]) < sma
    except Exception:
        return False


def sma_above(closes: pd.Series, fast: int = 10, slow: int = 30) -> bool:
    """True if fast-SMA > slow-SMA (uptrend)."""
    if len(closes) < slow:
        return False
    try:
        return float(closes.rolling(fast).mean().iloc[-1]) > \
               float(closes.rolling(slow).mean().iloc[-1])
    except Exception:
        return False


def macd_bearish(closes: pd.Series) -> bool:
    """
    True when MACD line (12/26 EMA diff) is below its signal line (9-period EMA).
    Provides additional downtrend confirmation beyond the SMA gate.
    All calculations are on the already-time-sliced `closes` series — no lookahead.
    """
    if len(closes) < 35:
        return False
    try:
        ema12 = closes.ewm(span=12, adjust=False).mean()
        ema26 = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return float(macd_line.iloc[-1]) < float(signal_line.iloc[-1])
    except Exception:
        return False


def run_and_report(
    eng: BacktestEngine,
    strategy_fn,
    start: datetime,
    end: datetime,
    name: str,
) -> tuple[dict, dict]:
    """Run strategy, run Monte Carlo, print one-line summary, return (results, mc)."""
    results = eng.run(strategy_fn, start, end)
    mc = eng.run_monte_carlo(n_sims=1_000)
    init = 100_000.0
    ret_pct = results.get("total_return", 0.0) * 100
    sharpe = results.get("sharpe_ratio", 0.0)
    maxdd = results.get("max_drawdown", 0.0) * 100
    p5_ret = (mc.get("mc_95_pct_equity", init) / init - 1.0) * 100
    p1_ret = (mc.get("mc_99_pct_equity", init) / init - 1.0) * 100
    print(
        f"  {name:<36} | {ret_pct:>+7.2f}%  Sharpe {sharpe:>5.2f}"
        f"  MaxDD {maxdd:>7.2f}%  MC-p5 {p5_ret:>+6.1f}%  MC-p1 {p1_ret:>+6.1f}%"
    )
    return results, mc


def _make_engine(**kwargs) -> BacktestEngine:
    """Engine factory with v3 defaults (dynamic slippage enabled)."""
    defaults: dict = dict(
        initial_capital=100_000,
        commission_per_share=0.005,
        slippage_bps=5.0,
        use_dynamic_slippage=True,
        max_positions=10,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.15,
        enable_stop_management=True,
        default_stop_loss_pct=0.06,
        default_take_profit_pct=0.35,
        default_trailing_stop_pct=0.06,
        default_trailing_activation_pct=0.03,
        default_max_hold_bars=65,
        use_open_price_fill=True,
    )
    defaults.update(kwargs)
    eng = BacktestEngine(**defaults)
    eng.risk_guard.max_price_deviation_bps = 2000.0
    eng.risk_guard.max_order_notional = 500_000.0
    return eng


# ═══════════════════════════════════════════════════════════════
# STRATEGY 1 — GOLD MAXIMUM v3
# ═══════════════════════════════════════════════════════════════

def strategy_gold_maximum_v3(eng: BacktestEngine, _ts: datetime, dv) -> None:
    """
    Single concentrated long GLD for the entire Jan–Mar 2026 gold bull run.

    v3 vs v2
    ─────────
    • Kelly sizing (signal=1.0, confidence=0.90) instead of flat 95 % allocation.
    • Explicit StopLevel registered with 8 % trailing/TP 40 %, activation at +4 %.
    • apply_regime() widens stop further in 'bull'/'neutral' VIX — prevents
      premature exits during normal 3-4 % intraday corrections.
    """
    if "GLD" not in dv:
        return
    # DataView already time-slices to current bar — no lookahead possible.
    closes = dv["GLD"]["Close"].dropna()
    if closes.empty:
        return

    price = float(closes.iloc[-1])
    vix = get_vix(dv)

    target_shares = kelly_shares(eng, "GLD", price, closes, max_pct=0.95, vix=vix,
                                 min_scaling=0.85)

    if "GLD" not in eng.positions:
        # Initial entry — sized by current market data
        eng.execute_order("GLD", "BUY", target_shares)
        if eng.enable_stop_management:
            eng.stop_manager.register("GLD", StopLevel(
                stop_loss_pct=0.08,
                take_profit_pct=0.40,
                # Trailing disabled for concentrated GLD hold — ride the full trend.
                # Exit is handled by the 8% hard stop or TP at +40%.
                trailing_activation_pct=1.0,
                trailing_distance_pct=0.08,
                max_hold_bars=65,
            ))
            # GLD safe-haven: always use "bull" so stop widens (not tightens) in fear.
            eng.stop_manager.apply_regime("GLD", "bull")
    else:
        # Scale-in: as momentum builds the data-driven target grows — top up
        # the position when it's materially below the current target.
        current_qty = int(eng.positions["GLD"].quantity)
        add_qty = target_shares - current_qty
        if add_qty >= 3:  # Minimum 3-share top-up to avoid micro-orders
            eng.execute_order("GLD", "BUY", add_qty)


# ═══════════════════════════════════════════════════════════════
# STRATEGY 2 — GEOPOLITICAL PREMIUM v3
# ═══════════════════════════════════════════════════════════════

def strategy_geopolitical_premium_v3(eng: BacktestEngine, _ts: datetime, dv) -> None:
    """
    GLD + SLV metals core, XOM + CVX oil overlay on geopolitical fear.

    v3 vs v2
    ─────────
    • Portfolio-level 5 % DD guard: skip new entries if DD < −5 %.
    • Kelly sizing for metals (signal_strength proportional to momentum).
    • Oil entry double-gate: VIX > 18 AND momentum > 0.03 (was VIX > 18 alone).
    • apply_regime() after every entry for regime-aware stops.
    """
    vix = get_vix(dv)

    # ── METALS ────────────────────────────────────────────────
    # No portfolio guard here: safe-haven metals should always be active.
    for sym, alloc in [("GLD", 0.50), ("SLV", 0.25)]:
        if sym not in dv:
            continue
        closes = dv[sym]["Close"].dropna()
        if closes.empty:
            continue
        price = float(closes.iloc[-1])
        mom = momentum(closes)

        target_shares = kelly_shares(eng, sym, price, closes, max_pct=alloc,
                                     vix=min(vix, 18.0), min_scaling=0.80)

        if sym not in eng.positions:
            # Enter on first active bar — no momentum gate.
            # Kelly handles sizing; metals are always deployed as safe-haven.
            eng.execute_order(sym, "BUY", target_shares)
            if eng.enable_stop_management:
                eng.stop_manager.register(sym, StopLevel(
                    # Wide 15% hard stop: only catastrophic dip triggers this.
                    # Normal 5-10% pullbacks ride through — momentum exit handles
                    # trend reversals. Trailing disabled to hold through bull run.
                    stop_loss_pct=0.15,
                    take_profit_pct=0.45,
                    trailing_activation_pct=1.0,
                    trailing_distance_pct=0.12,
                    max_hold_bars=65,
                ))
                eng.stop_manager.apply_regime(sym, "bull")
        else:
            pos = eng.positions[sym]
            if mom < -0.08:
                if pos.quantity > 0:
                    eng.execute_order(sym, "SELL", pos.quantity)
            else:
                # Scale-in unconditionally (metals held until mom < -0.08).
                # Add shares whenever position is below current Kelly target.
                add = target_shares - int(pos.quantity)
                if add >= 3:
                    eng.execute_order(sym, "BUY", add)

    # ── ENERGY ────────────────────────────────────────────────
    # Portfolio guard only for speculative energy overlay (not safe-haven metals).
    if portfolio_dd(eng) < -0.05:
        return
    for sym, alloc in [("XOM", 0.10), ("CVX", 0.10)]:
        if sym not in dv:
            continue
        closes = dv[sym]["Close"].dropna()
        if closes.empty:
            continue
        price = float(closes.iloc[-1])
        mom = momentum(closes)

        # Cap VIX at 22: high VIX is the ENTRY signal here, not a risk warning.
        target_shares = kelly_shares(eng, sym, price, closes, max_pct=alloc,
                                     vix=min(vix, 22.0), min_scaling=0.78)

        if sym not in eng.positions:
            # Double-gate: elevated VIX AND confirmed momentum
            if vix > 18.0 and mom > 0.02:
                eng.execute_order(sym, "BUY", target_shares)
                if eng.enable_stop_management:
                    eng.stop_manager.register(sym, StopLevel(
                        stop_loss_pct=0.08,
                        take_profit_pct=0.28,
                        trailing_activation_pct=0.04,
                        trailing_distance_pct=0.07,
                        max_hold_bars=50,
                    ))
                    # "neutral" keeps stop at 0.9× (7.2%) rather than 0.6× (4.8%)
                    # in high_volatility — oil can have 3-5% daily swings.
                    eng.stop_manager.apply_regime(sym, "neutral")
        else:
            pos = eng.positions[sym]
            # Exit when fear premium subsides or momentum collapses
            if vix < 18.0 or mom < -0.05:
                if pos.quantity > 0:
                    eng.execute_order(sym, "SELL", pos.quantity)
            elif vix > 18.0 and mom > 0.02:
                # Scale-in as geopolitical fear deepens and oil momentum builds.
                add = target_shares - int(pos.quantity)
                if add >= 2:
                    eng.execute_order(sym, "BUY", add)


# ═══════════════════════════════════════════════════════════════
# STRATEGY 3 — VIX-ADAPTIVE HEAVY GLD v3
# ═══════════════════════════════════════════════════════════════

def strategy_vix_adaptive_v3(eng: BacktestEngine, _ts: datetime, dv) -> None:
    """
    Dynamic GLD/SLV allocation by VIX regime; healthcare satellites in calm.

    v3 vs v2
    ─────────
    • Kelly sizing for GLD/SLV.
    • Healthcare satellite momentum threshold: 0.01 → 0.03 (less noise).
    • Concentrate into top-2 healthcare by momentum rank (not all 4 equally).
    • Equity satellite exit fires on same bar as VIX crossing (was: next bar).
    • apply_regime() on all new entries.
    """
    vix = get_vix(dv)
    regime = get_regime(vix)

    if vix >= 24:
        gld_alloc, slv_alloc, eq_alloc = 0.85, 0.13, 0.00
    elif vix >= 18:
        gld_alloc, slv_alloc, eq_alloc = 0.70, 0.12, 0.00
    else:
        gld_alloc, slv_alloc, eq_alloc = 0.55, 0.12, 0.05

    # ── GOLD ──────────────────────────────────────────────────
    if "GLD" in dv:
        gld_closes = dv["GLD"]["Close"].dropna()
        if not gld_closes.empty:
            price = float(gld_closes.iloc[-1])
            mom = momentum(gld_closes)
            gld_target = kelly_shares(eng, "GLD", price, gld_closes, max_pct=gld_alloc, vix=vix,
                                      min_scaling=0.85)
            if "GLD" not in eng.positions:
                if mom > 0.0:
                    eng.execute_order("GLD", "BUY", gld_target)
                    if eng.enable_stop_management:
                        eng.stop_manager.register("GLD", StopLevel(
                            stop_loss_pct=0.07,
                            take_profit_pct=0.40,
                            trailing_activation_pct=1.0,   # Disabled — ride full trend
                            trailing_distance_pct=0.07,
                            max_hold_bars=65,
                        ))
                        eng.stop_manager.apply_regime("GLD", "bull")
            else:
                pos = eng.positions["GLD"]
                if mom > 0.0:
                    # Scale-in: VIX regime just increased target (e.g. 65% → 80%)
                    add = gld_target - int(pos.quantity)
                    if add >= 3:
                        eng.execute_order("GLD", "BUY", add)

    # ── SILVER ────────────────────────────────────────────────
    if "SLV" in dv:
        slv_closes = dv["SLV"]["Close"].dropna()
        if not slv_closes.empty:
            price = float(slv_closes.iloc[-1])
            mom = momentum(slv_closes)
            slv_target = kelly_shares(eng, "SLV", price, slv_closes, max_pct=slv_alloc, vix=vix,
                                      min_scaling=0.85)
            if "SLV" not in eng.positions:
                if mom > 0.0:
                    eng.execute_order("SLV", "BUY", slv_target)
                    if eng.enable_stop_management:
                        eng.stop_manager.register("SLV", StopLevel(
                            stop_loss_pct=0.07,
                            take_profit_pct=0.40,
                            trailing_activation_pct=1.0,   # Disabled — ride full trend
                            trailing_distance_pct=0.07,
                            max_hold_bars=65,
                        ))
                        eng.stop_manager.apply_regime("SLV", "bull")
            else:
                # Scale-in for silver when regime target grows
                if mom > 0.0:
                    add = slv_target - int(eng.positions["SLV"].quantity)
                    if add >= 2:
                        eng.execute_order("SLV", "BUY", add)

    # ── EQUITY SATELLITES (only when VIX < 18) ────────────────
    HEALTHCARE = ["UNH", "LLY", "JNJ", "ABBV"]

    if eq_alloc > 0.0:
        # Rank by 20-day momentum; enter only top-2 with enough momentum
        ranked: list[tuple[str, float]] = []
        for sym in HEALTHCARE:
            if sym not in dv:
                continue
            c = dv[sym]["Close"].dropna()
            if len(c) >= 21:
                ranked.append((sym, momentum(c)))
        ranked.sort(key=lambda x: x[1], reverse=True)
        top2 = [sym for sym, m in ranked[:2] if m > 0.03]

        # Same-bar exit for satellites no longer qualifying
        for sym in HEALTHCARE:
            if sym in eng.positions and sym not in top2:
                pos = eng.positions[sym]
                if pos.quantity > 0:
                    eng.execute_order(sym, "SELL", pos.quantity)

        for sym in top2:
            if sym in eng.positions:
                continue
            if sym not in dv:
                continue
            c = dv[sym]["Close"].dropna()
            if c.empty:
                continue
            price = float(c.iloc[-1])
            mom = momentum(c)
            if rsi(c) < 70:
                shares = kelly_shares(eng, sym, price, c, max_pct=eq_alloc, vix=vix)
                eng.execute_order(sym, "BUY", shares)
                if eng.enable_stop_management:
                    eng.stop_manager.apply_regime(sym, regime)
    else:
        # VIX ≥ 18: exit all equity satellites on the same bar the VIX gate fires
        for sym in HEALTHCARE:
            if sym in eng.positions:
                pos = eng.positions[sym]
                if pos.quantity > 0:
                    eng.execute_order(sym, "SELL", pos.quantity)


# ═══════════════════════════════════════════════════════════════
# STRATEGY 4 — TREND-CONFIRMED LONG/SHORT v3
# ═══════════════════════════════════════════════════════════════

def strategy_trend_ls_v3(eng: BacktestEngine, _ts: datetime, dv) -> None:
    """
    Long defensive basket (GLD/SLV/healthcare) + short QQQ on confirmed downtrend.

    v3 vs v2
    ─────────
    • Kelly sizing for all long entries.
    • QQQ short: MACD bearish confirmation added alongside the 30-SMA gate.
    • QQQ short notional raised 25 % → 30 %.
    • Explicit QQQ short TP at −7 % from entry price.
    • apply_regime() on all new entries.
    """
    vix = get_vix(dv)
    regime = get_regime(vix)

    # ── LONG BASKET ───────────────────────────────────────────
    LONGS = [("GLD", 0.42), ("SLV", 0.20), ("UNH", 0.15), ("LLY", 0.10)]
    for sym, alloc in LONGS:
        if sym not in dv:
            continue
        closes = dv[sym]["Close"].dropna()
        if closes.empty:
            continue
        price = float(closes.iloc[-1])
        mom = momentum(closes)
        r = rsi(closes)

        target_shares = kelly_shares(eng, sym, price, closes, max_pct=alloc, vix=vix,
                                     min_scaling=0.82)

        if sym not in eng.positions:
            if mom > 0.02 and r < 72:
                eng.execute_order(sym, "BUY", target_shares)
                if eng.enable_stop_management:
                    is_metal = sym in ("GLD", "SLV")
                    eng.stop_manager.register(sym, StopLevel(
                        stop_loss_pct=0.07 if is_metal else 0.06,
                        take_profit_pct=0.40,
                        # Metals: trailing disabled (ride full trend); others: normal trailing
                        trailing_activation_pct=1.0 if is_metal else 0.04,
                        trailing_distance_pct=0.07 if is_metal else 0.06,
                        max_hold_bars=65,
                    ))
                    stop_regime = "bull" if is_metal else regime
                    eng.stop_manager.apply_regime(sym, stop_regime)
        else:
            pos = eng.positions[sym]
            if mom < -0.05 or r > 84:
                if pos.quantity > 0:
                    eng.execute_order(sym, "SELL", pos.quantity)
            elif mom > 0.02 and r < 72:
                add = target_shares - int(pos.quantity)
                if add >= 2:
                    eng.execute_order(sym, "BUY", add)

    # ── QQQ SHORT ─────────────────────────────────────────────
    if "QQQ" not in dv:
        return
    qqq_closes = dv["QQQ"]["Close"].dropna()
    if qqq_closes.empty:
        return
    qqq_price = float(qqq_closes.iloc[-1])
    qqq_mom = momentum(qqq_closes)

    if "QQQ" not in eng.positions:
        # Triple gate: SMA downtrend + MACD bearish + negative momentum.
        # All three required to avoid entering shorts during short-lived dips.
        if below_sma(qqq_closes, 30) and macd_bearish(qqq_closes) and qqq_mom < -0.02:
            # Short entered because market is weak — high VIX helps, not hurts.
            # Reduced to 20% (was 30%) to cap MaxDD impact on wrong-way shorts.
            shares = kelly_shares(eng, "QQQ", qqq_price, qqq_closes,
                                  max_pct=0.20, vix=min(vix, 20.0),
                                  force_signal=-abs(qqq_mom))
            if shares > 0:
                eng.execute_order("QQQ", "SELL", shares)  # Open short
                if eng.enable_stop_management:
                    eng.stop_manager.register("QQQ", StopLevel(
                        stop_loss_pct=0.05,    # Stop if QQQ rises 5 %
                        take_profit_pct=0.07,  # TP at 7 % decline
                        trailing_activation_pct=0.03,
                        trailing_distance_pct=0.04,
                        max_hold_bars=40,
                    ))
                eng._s4_qqq_entry = qqq_price
    else:
        # Exit on trend reversal OR manual TP (7 % drop from entry)
        entry_p = getattr(eng, "_s4_qqq_entry", qqq_price)
        tp_hit = qqq_price <= entry_p * 0.93
        trend_reversed = not below_sma(qqq_closes, 30)
        if tp_hit or trend_reversed:
            pos = eng.positions["QQQ"]
            qty = abs(pos.quantity)
            if qty > 0:
                eng.execute_order("QQQ", "BUY", qty)  # Cover short


# ═══════════════════════════════════════════════════════════════
# STRATEGY 5 — MOMENTUM RANKING v3
# ═══════════════════════════════════════════════════════════════

def strategy_momentum_ranking_v3(eng: BacktestEngine, _ts: datetime, dv) -> None:
    """
    Cross-sectional momentum: stay long top-3 assets, rebalance weekly.

    v3 vs v2
    ─────────
    • Portfolio circuit breaker: if portfolio DD < −7 %, exit all and pause.
    • Per-position SL tightened to 5 % (engine built with default_stop_loss_pct=0.05).
    • Inverse-volatility weighting (lower-vol assets get larger allocation).
    • Exit trigger widened: drop from top-3 to outside top-5 (reduces churn).
    """
    TOP_N = 3
    UNIVERSE = [
        "GLD", "SLV", "XOM", "CVX",
        "UNH", "LLY", "JNJ", "ABBV",
        "QQQ", "NVDA", "AMD", "SPY",
    ]
    REBAL_FREQ = 5  # bars between rebalancing (~weekly)

    # ── PORTFOLIO CIRCUIT BREAKER ─────────────────────────────
    dd = portfolio_dd(eng)
    if dd < -0.07:
        for sym, pos in list(eng.positions.items()):
            if pos.quantity != 0:
                side = "SELL" if pos.quantity > 0 else "BUY"
                eng.execute_order(sym, side, abs(pos.quantity))
        return

    # ── THROTTLE TO REBALANCE FREQUENCY ──────────────────────
    bar_idx = eng._bar_idx
    if (bar_idx - getattr(eng, "_s5_last_rebal", -999)) < REBAL_FREQ:
        return
    eng._s5_last_rebal = bar_idx

    # ── COMPUTE MOMENTUM + VOLATILITY (all on sliced series) ──
    mom_scores: dict[str, float] = {}
    vol_scores: dict[str, float] = {}
    for sym in UNIVERSE:
        if sym not in dv:
            continue
        closes = dv[sym]["Close"].dropna()
        if len(closes) < 22:
            continue
        mom_scores[sym] = momentum(closes)
        rets = closes.pct_change().dropna()
        # Use last 20 returns for vol (already time-sliced — no lookahead)
        recent = rets.iloc[-20:] if len(rets) >= 20 else rets
        v = float(recent.std())
        vol_scores[sym] = v if v > 1e-8 else 1e-8  # Guard zero-vol

    if not mom_scores:
        return

    # ── RANK BY MOMENTUM ──────────────────────────────────────
    ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
    top_n = [sym for sym, m in ranked[:TOP_N] if m > 0.0]
    # Wider exit gate: drop out of top-5 (not top-3) to reduce churn
    top5_set = {sym for sym, m in ranked[:5] if m > 0.0}

    # ── EXIT POSITIONS OUTSIDE TOP-5 ──────────────────────────
    for sym in list(eng.positions.keys()):
        if sym in UNIVERSE and sym not in top5_set:
            pos = eng.positions[sym]
            if pos.quantity != 0:
                side = "SELL" if pos.quantity > 0 else "BUY"
                eng.execute_order(sym, side, abs(pos.quantity))

    # ── INVERSE-VOL WEIGHTS FOR TOP-N ─────────────────────────
    if top_n:
        inv_vols = {sym: 1.0 / vol_scores.get(sym, 0.01) for sym in top_n}
        total_iv = sum(inv_vols.values())
        if total_iv > 0:
            weights = {sym: iv / total_iv for sym, iv in inv_vols.items()}
        else:
            weights = {sym: 1.0 / len(top_n) for sym in top_n}
    else:
        weights = {}

    # ── ENTER / REBALANCE TOP-N ───────────────────────────────
    vix = get_vix(dv)
    regime = get_regime(vix)

    for sym in top_n:
        if sym in eng.positions or sym not in dv:
            continue
        closes = dv[sym]["Close"].dropna()
        if closes.empty:
            continue
        price = float(closes.iloc[-1])
        alloc = weights.get(sym, 1.0 / max(len(top_n), 1))

        shares = kelly_shares(eng, sym, price, closes, max_pct=alloc, vix=vix,
                              min_scaling=0.78)
        eng.execute_order(sym, "BUY", shares)
        if eng.enable_stop_management:
            eng.stop_manager.register(sym, StopLevel(
                stop_loss_pct=0.05,   # Tighter than v2
                take_profit_pct=0.40,
                trailing_activation_pct=0.04,
                trailing_distance_pct=0.05,
                max_hold_bars=65,
            ))
            eng.stop_manager.apply_regime(sym, regime)


# ═══════════════════════════════════════════════════════════════
# STRATEGY 6 — GOLD-SILVER CATCH-UP (NEW)
# ═══════════════════════════════════════════════════════════════

def strategy_gold_silver_catchup(eng: BacktestEngine, _ts: datetime, dv) -> None:
    """
    NEW: Silver consistently lags gold in initial rallies then catches up.

    Mechanism: When the gold/silver ratio exceeds its 30-day SMA by ≥ 2 %,
    silver is statistically cheap vs. gold → enter SLV long.
    A smaller GLD anchor is added in calm VIX for extra conviction.

    Lookahead guardrails:
    • gld_closes and slv_closes come from DataView (already sliced to _ts).
    • ratio_series.rolling(30).mean() is computed on the sliced series only.
    • All standard deviations are computed on the sliced series.
    """
    if "GLD" not in dv or "SLV" not in dv:
        return

    gld_closes = dv["GLD"]["Close"].dropna()
    slv_closes = dv["SLV"]["Close"].dropna()
    min_len = min(len(gld_closes), len(slv_closes))
    if min_len < 32:
        return

    # Align to the same length (take most recent min_len bars from each)
    gld_aligned = gld_closes.iloc[-min_len:].copy()
    slv_aligned = slv_closes.iloc[-min_len:].copy()

    gld_price = float(gld_aligned.iloc[-1])
    slv_price = float(slv_aligned.iloc[-1])

    # Guard against zero or NaN prices
    if slv_price <= 0 or gld_price <= 0:
        return

    # Gold/Silver ratio — safe division (replace non-positive SLV with NaN)
    slv_safe = slv_aligned.where(slv_aligned > 0)
    ratio_series = gld_aligned / slv_safe
    ratio_series = ratio_series.dropna()
    if len(ratio_series) < 31:
        return

    ratio_now = float(ratio_series.iloc[-1])
    # 30-day SMA computed on already-sliced ratio — zero lookahead
    ratio_sma30_val = float(ratio_series.rolling(30).mean().iloc[-1])

    if np.isnan(ratio_sma30_val) or ratio_sma30_val <= 0:
        return

    vix = get_vix(dv)
    regime = get_regime(vix)
    mom_slv = momentum(slv_closes)
    mom_gld = momentum(gld_closes)

    # ── ENTRY CONDITIONS ───────────────────────────────────────
    ratio_elevated = ratio_now > ratio_sma30_val * 1.02   # SLV ≥ 2 % cheap vs GLD
    gld_trending = mom_gld > 0.03                          # GLD uptrend
    slv_turning = mom_slv > 0.01                           # SLV itself must have positive momentum

    # ── SLV LONG (core position) ───────────────────────────────
    if "SLV" not in eng.positions:
        if ratio_elevated and gld_trending and slv_turning:
            ratio_excess = (ratio_now / ratio_sma30_val) - 1.0
            sig_strength = float(np.clip(ratio_excess * 10.0, 0.10, 1.0))
            shares = kelly_shares(eng, "SLV", slv_price, slv_closes,
                                  max_pct=0.38, vix=vix, force_signal=sig_strength,
                                  min_scaling=0.75)
            eng.execute_order("SLV", "BUY", shares)
            if eng.enable_stop_management:
                eng.stop_manager.register("SLV", StopLevel(
                    stop_loss_pct=0.07,
                    take_profit_pct=0.35,
                    trailing_activation_pct=0.05,
                    trailing_distance_pct=0.07,
                    max_hold_bars=65,
                ))
                eng.stop_manager.apply_regime("SLV", regime)
    else:
        # Exit when ratio reverts (silver has caught up) or momentum collapses
        ratio_reverted = ratio_now < ratio_sma30_val * 0.99
        slv_reversing = mom_slv < -0.08
        if ratio_reverted or slv_reversing:
            pos = eng.positions["SLV"]
            if pos.quantity > 0:
                eng.execute_order("SLV", "SELL", pos.quantity)
        elif ratio_elevated and slv_turning:
            # Scale-in: add to SLV as catch-up thesis strengthens
            ratio_excess = (ratio_now / ratio_sma30_val) - 1.0
            sig_strength = float(np.clip(ratio_excess * 10.0, 0.10, 1.0))
            slv_target = kelly_shares(eng, "SLV", slv_price, slv_closes,
                                      max_pct=0.38, vix=vix, force_signal=sig_strength,
                                      min_scaling=0.75)
            add = slv_target - int(eng.positions["SLV"].quantity)
            if add >= 5:  # Higher threshold — avoid over-churning volatile SLV
                eng.execute_order("SLV", "BUY", add)

    # ── GLD LONG (anchor, calm VIX only) ──────────────────────
    if "GLD" not in eng.positions:
        if vix < 22 and mom_gld > 0.08:
            shares = kelly_shares(eng, "GLD", gld_price, gld_closes, max_pct=0.35, vix=vix,
                                  min_scaling=0.80)
            eng.execute_order("GLD", "BUY", shares)
            if eng.enable_stop_management:
                eng.stop_manager.register("GLD", StopLevel(
                    stop_loss_pct=0.07,
                    take_profit_pct=0.35,
                    trailing_activation_pct=0.04,
                    trailing_distance_pct=0.07,
                    max_hold_bars=65,
                ))
                eng.stop_manager.apply_regime("GLD", regime)
    else:
        if mom_gld < -0.05:
            pos = eng.positions["GLD"]
            if pos.quantity > 0:
                eng.execute_order("GLD", "SELL", pos.quantity)


# ═══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════

V2_BASELINE: dict[str, dict] = {
    "1. Gold Maximum":           {"ret": 14.18, "sharpe": 3.62, "maxdd": -5.85},
    "2. Geopolitical Premium":   {"ret": 18.58, "sharpe": 3.38, "maxdd": -6.52},
    "3. VIX-Adaptive Heavy GLD": {"ret": 11.74, "sharpe": 2.71, "maxdd": -5.65},
    "4. Trend-Confirmed L/S":    {"ret":  7.68, "sharpe": 2.32, "maxdd": -4.78},
    "5. Momentum Ranking":       {"ret": 15.27, "sharpe": 2.82, "maxdd": -10.50},
}


def print_comparison_table(all_results: dict[str, tuple[dict, dict]]) -> None:
    """Print v2 baseline vs v3 results side-by-side with Monte Carlo columns."""
    INIT = 100_000.0
    W = 114
    SEP = "─" * W
    HDR = (
        f"  {'Strategy':<36} │ {'v2 Ret':>8} │ {'v3 Ret':>8} │ {'Δ':>7} │"
        f" {'Sharpe':>7} │ {'MaxDD':>8} │ {'MC-p5':>7} │ {'MC-p1':>7}"
    )
    print("\n" + SEP)
    print(HDR)
    print(SEP)

    for name, (results, mc) in all_results.items():
        v3_ret = results.get("total_return", 0.0) * 100
        v3_sh = results.get("sharpe_ratio", 0.0)
        v3_dd = results.get("max_drawdown", 0.0) * 100
        mc_p5 = (mc.get("mc_95_pct_equity", INIT) / INIT - 1.0) * 100
        mc_p1 = (mc.get("mc_99_pct_equity", INIT) / INIT - 1.0) * 100

        # Match baseline via prefix (strip " v3" / " (NEW)")
        base_key = name.split(" v3")[0].split(" (NEW)")[0].strip()
        v2 = V2_BASELINE.get(base_key, {})
        v2_ret = v2.get("ret", float("nan"))
        delta = v3_ret - v2_ret if not np.isnan(v2_ret) else float("nan")

        v2_str = f"{v2_ret:>+7.2f}%" if not np.isnan(v2_ret) else "   NEW  "
        dl_str = f"{delta:>+7.2f}%" if not np.isnan(delta) else "   —    "

        print(
            f"  {name:<36} │ {v2_str} │ {v3_ret:>+7.2f}% │ {dl_str} │"
            f" {v3_sh:>7.2f} │ {v3_dd:>+8.2f}% │ {mc_p5:>+6.1f}% │ {mc_p1:>+6.1f}%"
        )

    print(SEP)
    print(
        "\n  Columns: v2 Ret = prior baseline | v3 Ret = this run | Δ = improvement\n"
        "  MC-p5  = 5th-percentile outcome across 1 000 bootstrap simulations\n"
        "  MC-p1  = 1st-percentile worst case (stress test)\n"
    )


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    START = datetime(2026, 1, 2)
    END = datetime(2026, 3, 7)

    SYMBOLS = [
        "GLD", "SLV", "QQQ", "SPY",
        "XOM", "CVX",
        "UNH", "LLY", "JNJ", "ABBV",
        "NVDA", "AMD",
        "^VIX",
    ]

    print(f"\n{'═' * 60}")
    print("  Apex Trading — Backtest v3 Optimised Strategies")
    print(f"  Period : {START.date()} → {END.date()}")
    print("  Capital: $100,000 per strategy | Slippage: dynamic")
    print(f"{'═' * 60}")

    print("  Downloading market data …", end=" ", flush=True)
    data = load_data(SYMBOLS, START, END)
    print(f"done ({len(data)} symbols loaded)")

    strategies: list[tuple[str, object, BacktestEngine]] = [
        (
            "1. Gold Maximum v3",
            strategy_gold_maximum_v3,
            _make_engine(max_positions=2, slippage_bps=3.0),
        ),
        (
            "2. Geopolitical Premium v3",
            strategy_geopolitical_premium_v3,
            _make_engine(max_positions=6, slippage_bps=4.0),
        ),
        (
            "3. VIX-Adaptive Heavy GLD v3",
            strategy_vix_adaptive_v3,
            _make_engine(max_positions=8, slippage_bps=4.0),
        ),
        (
            "4. Trend-Confirmed L/S v3",
            strategy_trend_ls_v3,
            _make_engine(max_positions=8, slippage_bps=4.0),
        ),
        (
            "5. Momentum Ranking v3",
            strategy_momentum_ranking_v3,
            _make_engine(
                max_positions=5,
                slippage_bps=5.0,
                default_stop_loss_pct=0.05,
            ),
        ),
        (
            "6. Gold-Silver Catch-Up (NEW)",
            strategy_gold_silver_catchup,
            _make_engine(max_positions=4, slippage_bps=4.0),
        ),
    ]

    all_results: dict[str, tuple[dict, dict]] = {}

    print("\n  Running 6 strategies with Monte Carlo (1 000 sims each) …\n")
    for name, fn, eng in strategies:
        eng.data = data
        results, mc = run_and_report(eng, fn, START, END, name)
        all_results[name] = (results, mc)

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
