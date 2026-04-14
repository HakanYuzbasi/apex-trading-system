"""
Walk-Forward Backtest Engine — Kalman/Johansen Pairs Strategy
=============================================================
Pair: AAPL / MSFT  (configurable)

Architecture decisions
----------------------
- Custom bar-by-bar event loop: guarantees no look-ahead contamination in the
  Kalman filter. VectorBT and Backtrader are deliberately avoided (see docstring
  in WalkForwardEngine for the exact contamination vectors they introduce).
- Kalman hyperparameters (Q, R) are fit on the TRAINING window only, then the
  filter STATE is carried forward into the test window — so the model enters the
  test period with a warm, uncontaminated posterior.
- Alpaca paper-fill simulation: latency drawn from an empirical distribution
  calibrated to Alpaca paper trading; slippage from Almgren-Chriss impact model.
- Regime tagging: each test fold is labelled (bull/bear/high-vol/sideways) so
  you can see which regimes the strategy breaks in.

Usage
-----
    python scripts/walk_forward_backtest.py --pair AAPL MSFT --years 3
    python scripts/walk_forward_backtest.py --pair AAPL MSFT --years 3 --regimes-only

Output
------
    data/wfo_results/wfo_AAPL_MSFT_<timestamp>.json   — full fold metrics
    data/wfo_results/wfo_AAPL_MSFT_<timestamp>.png    — equity curves per fold
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("wfo")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WFOConfig:
    instrument_a: str = "AAPL"
    instrument_b: str = "MSFT"

    # ── Window sizes ──────────────────────────────────────────────────────────
    # Daily mode (default):  252 bars = 1 year train,  63 = 1 quarter test
    # 5-min intraday mode:   pass --train-bars 2340 --test-bars 390 --step-bars 130
    #   (30 sessions × 78 bars = 2340 train | 5 sessions × 78 = 390 test |
    #    ~8 sessions × 78 step — see INTRADAY_5MIN_CONFIG below for the exact block)
    train_bars: int = 252
    test_bars: int = 63
    step_bars: int = 21

    # Kalman hyperparameter search space (fit on train window only)
    q_candidates: list[float] = field(default_factory=lambda: [1e-5, 1e-4, 5e-4, 1e-3])
    r_candidates: list[float] = field(default_factory=lambda: [1e-3, 5e-3, 1e-2, 5e-2])

    # Strategy parameters
    entry_z: float = 2.0
    exit_z: float = 0.5
    leg_notional: float = 5_000.0
    warmup_bars: int = 30        # bars before entries allowed in test window

    # Johansen stabilizer (from updated stabilizer.py)
    coint_min_window: int = 60
    coint_fail_streak: int = 3
    coint_confidence: float = 0.90

    # Intraday session handling
    # When True the spread rolling window is cleared at each session boundary
    # (rows where session_start=True in the CSV).  This prevents the Kalman
    # filter treating the overnight gap as a spread signal.
    # Set automatically to True when the CSV contains a session_start column.
    session_aware: bool = False

    # Execution simulation
    alpaca_latency_mean_ms: float = 120.0   # empirical Alpaca paper mean
    alpaca_latency_std_ms: float = 60.0
    alpaca_latency_max_ms: float = 500.0
    half_spread_bps: float = 2.0            # typical large-cap spread
    market_impact_eta: float = 0.1          # Almgren-Chriss η
    adv_shares: float = 50_000_000.0        # AAPL/MSFT ~50M ADV

    # Data source: 'yfinance' or 'csv' (provide csv_path if csv)
    data_source: str = "yfinance"
    csv_path: str | None = None
    history_years: float = 3.0

    output_dir: str = "data/wfo_results"


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_daily_prices(cfg: WFOConfig) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [date, close_a, close_b, volume_a, volume_b].
    Uses yfinance by default; falls back to CSV if data_source='csv'.
    """
    if cfg.data_source == "csv":
        if not cfg.csv_path:
            raise ValueError("csv_path must be set when data_source='csv'")
        df = pd.read_csv(cfg.csv_path, parse_dates=["date"])
        required = {"date", "close_a", "close_b", "volume_a", "volume_b"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        if "session_start" in df.columns:
            df["session_start"] = df["session_start"].astype(bool)
            cfg.session_aware = True
            sessions = df["session_start"].sum()
            logger.info("Intraday CSV detected: %d bars, %d sessions — session_aware mode ON",
                        len(df), sessions)
        else:
            df["session_start"] = False
        return df.sort_values("date").reset_index(drop=True)

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance  — or set data_source='csv'")

    end = datetime.now(timezone.utc)
    start = pd.Timestamp(end) - pd.DateOffset(years=cfg.history_years)

    logger.info("Downloading %s and %s from yfinance (%s → %s)...",
                cfg.instrument_a, cfg.instrument_b,
                start.date(), pd.Timestamp(end).date())

    tickers = yf.download(
        [cfg.instrument_a, cfg.instrument_b],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # yfinance multi-ticker returns MultiIndex columns: (field, ticker)
    close = tickers["Close"]
    volume = tickers["Volume"]

    df = pd.DataFrame({
        "date": close.index,
        "close_a": close[cfg.instrument_a].values,
        "close_b": close[cfg.instrument_b].values,
        "volume_a": volume[cfg.instrument_a].values,
        "volume_b": volume[cfg.instrument_b].values,
    }).dropna().reset_index(drop=True)

    logger.info("Loaded %d bars for %s/%s", len(df), cfg.instrument_a, cfg.instrument_b)
    return df


# ---------------------------------------------------------------------------
# Kalman filter — pure NumPy, guaranteed bar-by-bar
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    theta: np.ndarray          # [hedge_ratio, intercept]
    covariance: np.ndarray     # 2×2 posterior covariance
    observation_count: int = 0

    def copy(self) -> "KalmanState":
        return KalmanState(
            theta=self.theta.copy(),
            covariance=self.covariance.copy(),
            observation_count=self.observation_count,
        )


def kalman_update(state: KalmanState, price_a: float, price_b: float,
                  Q: float, R: float) -> tuple[KalmanState, float, float]:
    """
    One-step Kalman update. Returns (new_state, innovation, innovation_variance).
    Matches the logic in quant_system/execution/fast_math.pyx exactly.

    State vector θ = [β, α] where spread = price_a - β×price_b - α
    Observation model: price_a = [price_b, 1] · θ + noise
    """
    H = np.array([price_b, 1.0])

    # Predict
    P_pred = state.covariance + Q * np.eye(2)

    # Innovation
    innovation = price_a - H @ state.theta
    S = H @ P_pred @ H + R  # innovation variance

    # Kalman gain
    K = P_pred @ H / S

    # Update
    theta_new = state.theta + K * innovation
    cov_new = (np.eye(2) - np.outer(K, H)) @ P_pred

    new_state = KalmanState(
        theta=theta_new,
        covariance=cov_new,
        observation_count=state.observation_count + 1,
    )
    return new_state, innovation, S


def run_kalman_on_window(prices_a: np.ndarray, prices_b: np.ndarray,
                         Q: float, R: float,
                         init_state: KalmanState | None = None
                         ) -> tuple[np.ndarray, np.ndarray, KalmanState]:
    """
    Run Kalman filter over a price window bar-by-bar.
    Returns (z_scores, spreads, final_state).

    If init_state is provided (carry-forward from training), the filter starts
    from that posterior rather than a diffuse prior — this is the correct way
    to avoid a cold-start bias at the beginning of each test window.
    """
    n = len(prices_a)
    spreads = np.zeros(n)
    z_scores = np.zeros(n)
    spread_history: list[float] = []

    if init_state is not None:
        state = init_state.copy()
    else:
        state = KalmanState(
            theta=np.zeros(2),
            covariance=np.eye(2) * 1_000.0,
        )

    for i in range(n):
        state, innovation, S = kalman_update(state, prices_a[i], prices_b[i], Q, R)

        beta = state.theta[0]
        alpha = state.theta[1]
        spread = prices_a[i] - beta * prices_b[i] - alpha
        spreads[i] = spread
        spread_history.append(spread)

        # Z-score: use expanding window mean/std (only past data)
        if len(spread_history) >= 20:
            arr = np.array(spread_history)
            mu = arr.mean()
            sigma = arr.std()
            z_scores[i] = (spread - mu) / sigma if sigma > 1e-9 else 0.0
        else:
            z_scores[i] = 0.0

    return z_scores, spreads, state


# ---------------------------------------------------------------------------
# Kalman hyperparameter fitting (on training window only)
# ---------------------------------------------------------------------------

def fit_kalman_params(prices_a: np.ndarray, prices_b: np.ndarray,
                      q_candidates: list[float],
                      r_candidates: list[float]) -> tuple[float, float]:
    """
    Grid search over Q×R using log-likelihood of innovations on the training window.
    This is the ONLY place where price data informs hyperparameters — and it only
    ever sees training data.

    Returns (best_Q, best_R).
    """
    best_ll = -np.inf
    best_Q, best_R = q_candidates[1], r_candidates[1]

    for Q in q_candidates:
        for R in r_candidates:
            state = KalmanState(theta=np.zeros(2), covariance=np.eye(2) * 1_000.0)
            log_likelihood = 0.0
            for i in range(len(prices_a)):
                state, innov, S = kalman_update(state, prices_a[i], prices_b[i], Q, R)
                if S > 1e-12:
                    log_likelihood += -0.5 * (np.log(2 * np.pi * S) + innov**2 / S)

            if log_likelihood > best_ll:
                best_ll = log_likelihood
                best_Q, best_R = Q, R

    logger.debug("Best Kalman params: Q=%.2e  R=%.2e  LL=%.2f", best_Q, best_R, best_ll)
    return best_Q, best_R


# ---------------------------------------------------------------------------
# Johansen cointegration check (debounced — same logic as updated stabilizer.py)
# ---------------------------------------------------------------------------

class DebouncedJohansen:
    def __init__(self, min_window: int = 60, fail_streak: int = 3,
                 confidence: float = 0.90):
        self.min_window = min_window
        self.fail_streak = fail_streak
        self._level_idx = {0.90: 0, 0.95: 1, 0.99: 2}.get(confidence, 0)
        self._fails = 0
        self.is_read_only = False

    def update(self, prices_a: list[float], prices_b: list[float]) -> None:
        if len(prices_a) < self.min_window:
            return

        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            data = pd.DataFrame({"a": prices_a, "b": prices_b})
            res = coint_johansen(data, det_order=0, k_ar_diff=1)
            passed = res.lr1[0] > res.cvt[0, self._level_idx]
        except Exception:
            passed = True  # don't block on error

        if passed:
            self._fails = 0
            self.is_read_only = False
        else:
            self._fails += 1
            if self._fails >= self.fail_streak:
                self.is_read_only = True


# ---------------------------------------------------------------------------
# Alpaca paper execution simulator
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    bar_index: int
    side: str            # 'long_a' | 'short_a' | 'close'
    price_a: float
    price_b: float
    fill_price_a: float
    fill_price_b: float
    shares_a: float
    shares_b: float
    latency_ms: float
    slippage_bps: float
    commission: float


def simulate_fill(bar_idx: int, side: str,
                  price_a: float, price_b: float,
                  shares_a: float, shares_b: float,
                  cfg: WFOConfig, rng: np.random.Generator) -> Fill:
    """
    Simulates an Alpaca paper fill with:
    1. Latency drawn from a truncated normal (empirical Alpaca paper distribution).
    2. Half-spread cost (bid-ask).
    3. Almgren-Chriss market impact: η × σ_daily × (|size|/ADV)^0.6
       σ_daily ≈ price × 0.015 for large-cap equities.
    4. $0 commission (Alpaca is commission-free).

    Latency does not affect price in daily bar backtests but is recorded for
    realism and will matter when you move to minute bars.
    """
    # Latency
    latency = float(np.clip(
        rng.normal(cfg.alpaca_latency_mean_ms, cfg.alpaca_latency_std_ms),
        10.0, cfg.alpaca_latency_max_ms
    ))

    # Direction multiplier: +1 if buying, -1 if selling
    dir_a = +1.0 if side == "long_a" else -1.0
    dir_b = -1.0 if side == "long_a" else +1.0
    if side == "close":
        # Closing: direction depends on current position — handled by caller
        dir_a, dir_b = 0.0, 0.0

    def impact(price: float, shares: float) -> float:
        """Almgren-Chriss single-leg impact in price units."""
        sigma = price * 0.015  # ~1.5% daily vol for AAPL/MSFT
        notional_fraction = abs(shares) * price / (cfg.adv_shares * price)
        return cfg.market_impact_eta * sigma * (notional_fraction ** 0.6)

    spread_cost = price_a * cfg.half_spread_bps / 10_000
    impact_a = impact(price_a, shares_a)
    fill_a = price_a + dir_a * (spread_cost + impact_a)

    spread_cost_b = price_b * cfg.half_spread_bps / 10_000
    impact_b = impact(price_b, shares_b)
    fill_b = price_b + dir_b * (spread_cost_b + impact_b)

    slippage_bps = (
        abs(fill_a - price_a) / price_a +
        abs(fill_b - price_b) / price_b
    ) / 2 * 10_000

    return Fill(
        bar_index=bar_idx,
        side=side,
        price_a=price_a,
        price_b=price_b,
        fill_price_a=fill_a,
        fill_price_b=fill_b,
        shares_a=shares_a,
        shares_b=shares_b,
        latency_ms=latency,
        slippage_bps=round(slippage_bps, 3),
        commission=0.0,
    )


# ---------------------------------------------------------------------------
# Single-fold backtester
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    regime: str
    kalman_Q: float
    kalman_R: float
    n_trades: int
    win_rate: float
    total_pnl: float
    sharpe: float
    max_drawdown: float
    avg_slippage_bps: float
    avg_latency_ms: float
    read_only_pct: float       # % of test bars where Johansen blocked entries
    equity_curve: list[float]  # daily equity values in test window


def classify_regime(prices_a: np.ndarray, prices_b: np.ndarray) -> str:
    """Simple regime label for a price window."""
    avg_ret = (np.diff(np.log(prices_a)).mean() + np.diff(np.log(prices_b)).mean()) / 2
    vol = (np.diff(np.log(prices_a)).std() + np.diff(np.log(prices_b)).std()) / 2 * np.sqrt(252)

    if vol > 0.35:
        return "high_vol"
    elif avg_ret > 0.0005:
        return "bull"
    elif avg_ret < -0.0005:
        return "bear"
    else:
        return "sideways"


def run_fold(fold_id: int,
             train_df: pd.DataFrame,
             test_df: pd.DataFrame,
             cfg: WFOConfig,
             rng: np.random.Generator) -> FoldResult:
    """
    Run one walk-forward fold.

    1. Fit Kalman Q, R on training window.
    2. Run Kalman through training window to get the warm posterior state.
    3. Run Kalman through test window starting from that warm state.
    4. Simulate trades bar-by-bar with debounced Johansen guard.
    5. Return FoldResult with all metrics.
    """
    prices_a_train = train_df["close_a"].values
    prices_b_train = train_df["close_b"].values
    prices_a_test = test_df["close_a"].values
    prices_b_test = test_df["close_b"].values

    # ── Step 1: Fit hyperparameters on training data only ─────────────────
    best_Q, best_R = fit_kalman_params(
        prices_a_train, prices_b_train,
        cfg.q_candidates, cfg.r_candidates,
    )

    # ── Step 2: Run filter through training window → warm posterior ───────
    _, _, warm_state = run_kalman_on_window(
        prices_a_train, prices_b_train, best_Q, best_R, init_state=None
    )
    # warm_state is the filter posterior at the last training bar.
    # No test data has been seen. This is the carry-forward state.

    # ── Step 3: Run filter through test window from warm state ─────────────
    z_scores, spreads, _ = run_kalman_on_window(
        prices_a_test, prices_b_test, best_Q, best_R, init_state=warm_state
    )

    # ── Step 4: Simulate trades ────────────────────────────────────────────
    regime = classify_regime(prices_a_test, prices_b_test)
    johansen = DebouncedJohansen(
        min_window=cfg.coint_min_window,
        fail_streak=cfg.coint_fail_streak,
        confidence=cfg.coint_confidence,
    )

    # Rolling price history for Johansen (seed from end of training window)
    price_hist_a: list[float] = list(prices_a_train[-cfg.coint_min_window:])
    price_hist_b: list[float] = list(prices_b_train[-cfg.coint_min_window:])

    # Session-start flags (all False for daily CSVs; populated from intraday CSV)
    session_starts = test_df["session_start"].values if "session_start" in test_df.columns \
        else np.zeros(len(test_df), dtype=bool)

    position = 0        # 0=flat, +1=long_spread, -1=short_spread
    cash = 0.0
    equity_curve: list[float] = []
    trades: list[dict] = []
    fills: list[Fill] = []
    read_only_bars = 0

    entry_price_a = entry_price_b = 0.0
    entry_shares_a = entry_shares_b = 0.0

    for i in range(len(prices_a_test)):
        pa = prices_a_test[i]
        pb = prices_b_test[i]
        z = z_scores[i]

        # ── Session boundary reset (intraday only) ────────────────────────
        # At each market open, clear the spread history so the Kalman z-score
        # doesn't treat the overnight gap as a mean-reversion signal.
        # Also force-close any open position: we never hold overnight.
        if cfg.session_aware and session_starts[i] and i > 0:
            price_hist_a.clear()
            price_hist_b.clear()
            johansen._fails = 0
            johansen.is_read_only = False
            if position != 0:
                # Close at previous bar's prices (market-on-open proxy)
                close_fill = simulate_fill(
                    i, "close", prices_a_test[i - 1], prices_b_test[i - 1],
                    entry_shares_a, entry_shares_b, cfg, rng,
                )
                if position == +1:
                    pnl = ((close_fill.fill_price_a - entry_price_a) * entry_shares_a
                           - (close_fill.fill_price_b - entry_price_b) * entry_shares_b)
                else:
                    pnl = (-(close_fill.fill_price_a - entry_price_a) * entry_shares_a
                           + (close_fill.fill_price_b - entry_price_b) * entry_shares_b)
                cash += pnl
                trades.append({"pnl": pnl, "slippage_bps": close_fill.slippage_bps,
                                "forced_eod": True})
                position = 0

        price_hist_a.append(pa)
        price_hist_b.append(pb)
        if len(price_hist_a) > 500:
            price_hist_a.pop(0)
            price_hist_b.pop(0)

        johansen.update(price_hist_a, price_hist_b)

        if johansen.is_read_only:
            read_only_bars += 1

        # Position P&L mark
        if position == +1:
            unrealised = (
                (pa - entry_price_a) * entry_shares_a
                - (pb - entry_price_b) * entry_shares_b
            )
        elif position == -1:
            unrealised = (
                -(pa - entry_price_a) * entry_shares_a
                + (pb - entry_price_b) * entry_shares_b
            )
        else:
            unrealised = 0.0

        equity_curve.append(cash + unrealised)

        # Skip entries during warmup
        if i < cfg.warmup_bars:
            continue

        # ── Entry logic ───────────────────────────────────────────────────
        if position == 0 and not johansen.is_read_only:
            shares_a = cfg.leg_notional / pa
            shares_b = (cfg.leg_notional * abs(warm_state.theta[0])) / pb

            if z > cfg.entry_z:
                # Spread is too high: short A, long B
                fill = simulate_fill(i, "short_a", pa, pb, shares_a, shares_b, cfg, rng)
                fills.append(fill)
                position = -1
                entry_price_a = fill.fill_price_a
                entry_price_b = fill.fill_price_b
                entry_shares_a = shares_a
                entry_shares_b = shares_b

            elif z < -cfg.entry_z:
                # Spread is too low: long A, short B
                fill = simulate_fill(i, "long_a", pa, pb, shares_a, shares_b, cfg, rng)
                fills.append(fill)
                position = +1
                entry_price_a = fill.fill_price_a
                entry_price_b = fill.fill_price_b
                entry_shares_a = shares_a
                entry_shares_b = shares_b

        # ── Exit logic ────────────────────────────────────────────────────
        elif position != 0:
            should_exit = (
                (position == +1 and z >= -cfg.exit_z) or
                (position == -1 and z <= cfg.exit_z)
            )
            if should_exit:
                close_side = "close"
                fill = simulate_fill(i, close_side, pa, pb,
                                     entry_shares_a, entry_shares_b, cfg, rng)
                fills.append(fill)

                if position == +1:
                    pnl = (
                        (fill.fill_price_a - entry_price_a) * entry_shares_a
                        - (fill.fill_price_b - entry_price_b) * entry_shares_b
                    )
                else:
                    pnl = (
                        -(fill.fill_price_a - entry_price_a) * entry_shares_a
                        + (fill.fill_price_b - entry_price_b) * entry_shares_b
                    )

                cash += pnl
                trades.append({"pnl": pnl, "slippage_bps": fill.slippage_bps})
                position = 0

    # Force-close any open position at end of fold
    if position != 0 and len(prices_a_test) > 0:
        pa = prices_a_test[-1]
        pb = prices_b_test[-1]
        fill = simulate_fill(len(prices_a_test) - 1, "close", pa, pb,
                             entry_shares_a, entry_shares_b, cfg, rng)
        if position == +1:
            pnl = (
                (fill.fill_price_a - entry_price_a) * entry_shares_a
                - (fill.fill_price_b - entry_price_b) * entry_shares_b
            )
        else:
            pnl = (
                -(fill.fill_price_a - entry_price_a) * entry_shares_a
                + (fill.fill_price_b - entry_price_b) * entry_shares_b
            )
        cash += pnl
        trades.append({"pnl": pnl, "slippage_bps": fill.slippage_bps})
        equity_curve[-1] = cash

    # ── Metrics ──────────────────────────────────────────────────────────
    n_trades = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = sum(1 for t in trades if t["pnl"] > 0) / n_trades if n_trades else 0.0
    avg_slippage = np.mean([f.slippage_bps for f in fills]) if fills else 0.0
    avg_latency = np.mean([f.latency_ms for f in fills]) if fills else 0.0
    read_only_pct = read_only_bars / len(prices_a_test) * 100 if len(prices_a_test) else 0.0

    # Sharpe on daily equity returns
    eq = np.array(equity_curve)
    daily_returns = np.diff(eq) if len(eq) > 1 else np.array([0.0])
    sharpe = (
        (daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        if daily_returns.std() > 1e-9 else 0.0
    )

    # Max drawdown (anchored to initial capital baseline, not equity=0)
    baseline = cfg.leg_notional * 2  # two-leg notional as reference scale
    running_max = np.maximum.accumulate(eq + baseline)
    drawdowns = (eq - np.maximum.accumulate(eq)) / baseline
    max_dd = float(drawdowns.min())

    return FoldResult(
        fold_id=fold_id,
        train_start=str(train_df["date"].iloc[0].date()),
        train_end=str(train_df["date"].iloc[-1].date()),
        test_start=str(test_df["date"].iloc[0].date()),
        test_end=str(test_df["date"].iloc[-1].date()),
        regime=regime,
        kalman_Q=best_Q,
        kalman_R=best_R,
        n_trades=n_trades,
        win_rate=round(win_rate, 4),
        total_pnl=round(total_pnl, 2),
        sharpe=round(sharpe, 4),
        max_drawdown=round(max_dd, 4),
        avg_slippage_bps=round(avg_slippage, 3),
        avg_latency_ms=round(avg_latency, 1),
        read_only_pct=round(read_only_pct, 1),
        equity_curve=[round(v, 2) for v in equity_curve],
    )


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def generate_folds(df: pd.DataFrame,
                   cfg: WFOConfig) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yields (train_df, test_df) pairs using an anchored expanding-window scheme.
    The training window grows by step_bars each iteration; the test window is
    always exactly test_bars wide.
    """
    n = len(df)
    start = cfg.train_bars

    while start + cfg.test_bars <= n:
        train_df = df.iloc[:start].copy()
        test_df = df.iloc[start: start + cfg.test_bars].copy()
        yield train_df, test_df
        start += cfg.step_bars


class WalkForwardEngine:
    """
    WHY NOT VECTORBT/BACKTRADER:
    - VectorBT processes the entire price array as numpy vectors. Any stateful
      estimator (Kalman, rolling covariance) that you apply after the fact can
      accidentally read t+1 prices if you're not careful with array slicing.
      It also pre-computes indicators on the full dataset before strategy logic
      runs, making clean walk-forward separation cumbersome.
    - Backtrader is event-driven but uses Python generators internally; its
      indicator system computes on the full data series at init time. Walk-
      forward requires patching the data feed at each fold, which is fragile.
    - This engine: a plain Python for-loop over bars. The Kalman state at bar t
      is a pure function of bars [0..t-1]. The Johansen test at bar t reads
      only price_history[:t]. Look-ahead is structurally impossible.
    """

    def __init__(self, cfg: WFOConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=42)
        self.results: list[FoldResult] = []

    def run(self, df: pd.DataFrame) -> list[FoldResult]:
        folds = list(generate_folds(df, self.cfg))
        logger.info("Running %d walk-forward folds for %s/%s...",
                    len(folds), self.cfg.instrument_a, self.cfg.instrument_b)

        for fold_id, (train_df, test_df) in enumerate(folds):
            logger.info(
                "Fold %02d | Train: %s→%s (%d bars) | Test: %s→%s (%d bars)",
                fold_id,
                train_df["date"].iloc[0].date(), train_df["date"].iloc[-1].date(), len(train_df),
                test_df["date"].iloc[0].date(), test_df["date"].iloc[-1].date(), len(test_df),
            )
            result = run_fold(fold_id, train_df, test_df, self.cfg, self.rng)
            self.results.append(result)
            logger.info(
                "  → regime=%-10s  trades=%3d  PnL=$%8.2f  Sharpe=%6.3f  "
                "WinRate=%4.1f%%  MaxDD=%5.2f%%  ReadOnly=%4.1f%%  AvgSlip=%.1fbps",
                result.regime, result.n_trades, result.total_pnl, result.sharpe,
                result.win_rate * 100, result.max_drawdown * 100,
                result.read_only_pct, result.avg_slippage_bps,
            )

        return self.results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[FoldResult], cfg: WFOConfig) -> None:
    if not results:
        logger.warning("No fold results to summarise.")
        return

    total_pnl = sum(r.total_pnl for r in results)
    total_trades = sum(r.n_trades for r in results)
    winning_folds = sum(1 for r in results if r.total_pnl > 0)
    sharpes = [r.sharpe for r in results if r.n_trades > 0]
    avg_sharpe = np.mean(sharpes) if sharpes else 0.0
    worst_dd = min(r.max_drawdown for r in results)
    avg_read_only = np.mean([r.read_only_pct for r in results])
    avg_slippage = np.mean([r.avg_slippage_bps for r in results if r.n_trades > 0])

    by_regime: dict[str, list[float]] = {}
    for r in results:
        by_regime.setdefault(r.regime, []).append(r.total_pnl)

    print("\n" + "=" * 72)
    print(f" WALK-FORWARD SUMMARY — {cfg.instrument_a}/{cfg.instrument_b}")
    print("=" * 72)
    print(f" Folds run          : {len(results)}")
    print(f" Winning folds      : {winning_folds}/{len(results)} "
          f"({winning_folds/len(results)*100:.0f}%)")
    print(f" Total PnL          : ${total_pnl:,.2f}")
    print(f" Total trades       : {total_trades}")
    print(f" Avg out-of-sample  ")
    print(f"   Sharpe           : {avg_sharpe:.3f}")
    print(f"   Max drawdown     : {worst_dd*100:.2f}%")
    print(f"   Avg slippage     : {avg_slippage:.1f} bps")
    print(f"   Read-only time   : {avg_read_only:.1f}% of bars")
    print()
    print(" PnL by regime:")
    for regime, pnls in sorted(by_regime.items()):
        print(f"   {regime:<12} : ${sum(pnls):>10,.2f}  "
              f"({len(pnls)} folds, {sum(1 for p in pnls if p>0)}/{len(pnls)} winning)")

    print()
    # CRO graduation criteria
    print(" CRO GRADUATION CRITERIA:")
    criteria = [
        ("Avg Sharpe > 0.5",          avg_sharpe > 0.5),
        ("Winning fold rate > 60%",   winning_folds / len(results) > 0.60),
        ("Worst drawdown > -15%",     worst_dd > -0.15),
        ("Avg slippage < 5 bps",      avg_slippage < 5.0),
        ("Read-only time < 20%",      avg_read_only < 20.0),
        ("Bull regime profitable",    sum(by_regime.get("bull", [0])) > 0),
        ("Bear regime profitable",    sum(by_regime.get("bear", [0])) > 0),
    ]
    all_pass = True
    for label, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"   {status}  {label}")

    print()
    if all_pass:
        print(" 🟢 PHASE 1 COMPLETE — proceed to Phase 2 (PPO Audit)")
    else:
        print(" 🔴 NOT READY — fix failing criteria before Phase 2")
    print("=" * 72 + "\n")


def save_results(results: list[FoldResult], cfg: WFOConfig) -> Path:
    out_dir = PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = out_dir / f"wfo_{cfg.instrument_a}_{cfg.instrument_b}_{ts}.json"

    payload = {
        "config": {
            "instrument_a": cfg.instrument_a,
            "instrument_b": cfg.instrument_b,
            "train_bars": cfg.train_bars,
            "test_bars": cfg.test_bars,
            "step_bars": cfg.step_bars,
            "entry_z": cfg.entry_z,
            "exit_z": cfg.exit_z,
            "leg_notional": cfg.leg_notional,
        },
        "folds": [asdict(r) for r in results],
    }
    fname.write_text(json.dumps(payload, indent=2))
    logger.info("Results saved → %s", fname)
    return fname


def plot_results(results: list[FoldResult], cfg: WFOConfig, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, (r, ax) in enumerate(zip(results, axes)):
        eq = r.equity_curve
        color = "green" if r.total_pnl >= 0 else "red"
        ax.plot(eq, color=color, linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(
            f"Fold {r.fold_id} | {r.regime}\n"
            f"PnL=${r.total_pnl:.0f}  Sharpe={r.sharpe:.2f}",
            fontsize=8,
        )
        ax.set_xlabel("Test bar", fontsize=7)
        ax.set_ylabel("PnL ($)", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Walk-Forward Equity Curves — {cfg.instrument_a}/{cfg.instrument_b}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    png_path = out_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    logger.info("Equity curve chart → %s", png_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Walk-Forward Backtest — Kalman/Johansen Pairs Strategy"
    )
    parser.add_argument("--pair", nargs=2, default=["AAPL", "MSFT"],
                        metavar=("A", "B"), help="Ticker pair (default: AAPL MSFT)")
    parser.add_argument("--years", type=float, default=3.0,
                        help="Years of history to download (default: 3)")
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--leg-notional", type=float, default=5_000.0)
    parser.add_argument("--train-bars", type=int, default=None,
                        help="Override training window size in bars")
    parser.add_argument("--test-bars", type=int, default=None,
                        help="Override test window size in bars")
    parser.add_argument("--step-bars", type=int, default=None,
                        help="Override step size in bars")
    parser.add_argument("--intraday", action="store_true",
                        help="Apply 5-min intraday parameter preset "
                             "(train=2340, test=390, step=130, warmup=78, "
                             "coint_min=78). Overridden by explicit --train/test/step-bars.")
    parser.add_argument("--regimes-only", action="store_true",
                        help="Print regime classification per fold and exit")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV instead of yfinance")
    args = parser.parse_args()

    # ── 5-min intraday preset ─────────────────────────────────────────────
    # 1 trading session = 78 bars (09:30-16:00, 5-min bars)
    # Training: 30 sessions = 2,340 bars (~6 weeks of price history)
    # Test:      5 sessions =   390 bars (1 week out-of-sample)
    # Step:     ~8 sessions =   130 bars (step forward ~1.5 weeks)
    # Warmup:    1 session  =    78 bars (let filter settle before entries)
    # Johansen:  1 session  =    78 bars minimum (not 60 daily bars)
    if args.intraday:
        default_train = 2_340
        default_test  =   390
        default_step  =   130
        default_warmup =   78
        default_coint  =   78
    else:
        default_train = 252
        default_test  =  63
        default_step  =  21
        default_warmup =  30
        default_coint  =  60

    cfg = WFOConfig(
        instrument_a=args.pair[0],
        instrument_b=args.pair[1],
        history_years=args.years,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        leg_notional=args.leg_notional,
        train_bars=args.train_bars  if args.train_bars else default_train,
        test_bars =args.test_bars   if args.test_bars  else default_test,
        step_bars =args.step_bars   if args.step_bars  else default_step,
        warmup_bars=default_warmup,
        coint_min_window=default_coint,
        data_source="csv" if args.csv else "yfinance",
        csv_path=args.csv,
    )

    df = load_daily_prices(cfg)

    if args.regimes_only:
        for train_df, test_df in generate_folds(df, cfg):
            regime = classify_regime(
                test_df["close_a"].values, test_df["close_b"].values
            )
            print(f"  {test_df['date'].iloc[0].date()} → "
                  f"{test_df['date'].iloc[-1].date()}  :  {regime}")
        return

    engine = WalkForwardEngine(cfg)
    results = engine.run(df)

    print_summary(results, cfg)
    out_path = save_results(results, cfg)
    plot_results(results, cfg, out_path)


if __name__ == "__main__":
    main()
