"""
risk/factor_hedger.py - Portfolio Beta and Factor Exposure Monitor

Computes rolling portfolio beta vs SPY (equity) and BTC/USD (crypto), decomposes
factor exposures (market, momentum, size), and produces hedge recommendations when
unintended directional risk exceeds configurable thresholds.

The FactorHedger does NOT place orders.  It returns a FactorExposure snapshot each
cycle that the execution loop can inspect and alert on.

Integration in execution_loop.py:
    self._factor_hedger = FactorHedger(
        beta_warn_threshold=ApexConfig.FACTOR_BETA_WARN_THRESHOLD,
        beta_urgent_threshold=ApexConfig.FACTOR_BETA_URGENT_THRESHOLD,
        lookback_days=ApexConfig.FACTOR_HEDGER_LOOKBACK_DAYS,
    )  # in __init__

    exposure = self._factor_hedger.get_exposure(self.positions, current_prices)
    if exposure.hedge_urgency == "urgent":
        fire_alert(
            "factor_hedge_urgent",
            exposure.hedge_recommendation or "High portfolio beta detected",
            AlertSeverity.HIGH,
        )
    elif exposure.hedge_urgency == "advisory":
        logger.warning("FactorHedger advisory: %s", exposure.hedge_recommendation)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Internal symbol keys used to look up factor return series
_SPY_KEY = "SPY"
_BTC_KEY = "BTC/USD"

# Minimum R² for a beta estimate to be considered reliable
_MIN_R2 = 0.05

# Hedge is only recommended for market (beta) risk when the book is diversified
# enough that a market hedge makes sense (HHI < this threshold).
_HHI_HEDGE_MAX = 0.30

# Notional share sizes for recommendation strings — coarse approximations;
# the execution loop should refine using live prices.
_SPY_APPROX_PRICE = 500.0   # USD per SPY share
_BTC_APPROX_PRICE = 85_000.0  # USD per BTC unit


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FactorExposure:
    """Portfolio factor exposure snapshot returned each cycle."""

    market_beta_equity: float = 0.0
    """Portfolio beta vs SPY (OLS over lookback window)."""

    market_beta_crypto: float = 0.0
    """Portfolio beta vs BTC/USD (OLS over lookback window)."""

    portfolio_vol_ann: float = 0.0
    """Annualised portfolio volatility (sqrt(252) * daily std)."""

    momentum_factor: float = 0.0
    """Approximate momentum exposure: avg(last-5-day returns) of held positions."""

    size_factor: float = 0.0
    """Proxy size exposure: avg normalised market-cap rank (0=small, 1=large).
    Populated only when size_ranks are supplied; otherwise 0."""

    largest_single_exposure: str = ""
    """Symbol with the highest absolute portfolio weight."""

    concentration_hhi: float = 0.0
    """Herfindahl-Hirschman Index of position weights (0=perfectly diversified, 1=concentrated)."""

    hedge_recommendation: Optional[str] = None
    """Human-readable hedge action, e.g. 'SHORT SPY 4 shares'."""

    hedge_urgency: str = "none"
    """Urgency level: 'none' | 'advisory' | 'urgent'."""

    r2_equity: float = 0.0
    """R² of the SPY beta regression (reliability indicator)."""

    r2_crypto: float = 0.0
    """R² of the BTC beta regression (reliability indicator)."""

    num_positions: int = 0
    """Number of open positions at snapshot time."""


# ---------------------------------------------------------------------------
# FactorHedger
# ---------------------------------------------------------------------------

class FactorHedger:
    """
    Portfolio beta and factor exposure monitor.

    Call ``update_prices(symbol, returns)`` each cycle (or whenever new daily
    returns are available) for every tracked symbol, including 'SPY' and
    'BTC/USD' as the benchmark factors.

    Then call ``get_exposure(positions, prices)`` to obtain a FactorExposure
    snapshot.  The caller is responsible for any alerting or logging — this
    class is purely advisory.

    Parameters
    ----------
    beta_warn_threshold:
        Emit advisory recommendation when |equity_beta| exceeds this level.
        Default 1.20 (matches FACTOR_BETA_WARN_THRESHOLD in config.py).
    beta_urgent_threshold:
        Emit urgent recommendation when |equity_beta| exceeds this level.
        Default 1.80 (matches FACTOR_BETA_URGENT_THRESHOLD in config.py).
    lookback_days:
        Number of daily return observations used for the OLS beta regression.
        Default 20 (one trading month).
    min_days:
        Minimum observations required before computing beta.  Below this the
        hedger returns zeros with urgency='none'.
        Default 5.
    spy_approx_price:
        Approximate SPY price used when building hedge recommendation strings.
        Will be overridden if 'SPY' appears in the ``prices`` dict passed to
        ``get_exposure()``.
    btc_approx_price:
        Approximate BTC price — same override logic as spy_approx_price.
    """

    def __init__(
        self,
        beta_warn_threshold: float = 1.20,
        beta_urgent_threshold: float = 1.80,
        lookback_days: int = 20,
        min_days: int = 5,
        spy_approx_price: float = _SPY_APPROX_PRICE,
        btc_approx_price: float = _BTC_APPROX_PRICE,
    ) -> None:
        if beta_warn_threshold <= 0:
            raise ValueError("beta_warn_threshold must be positive")
        if beta_urgent_threshold <= beta_warn_threshold:
            raise ValueError("beta_urgent_threshold must exceed beta_warn_threshold")
        if lookback_days < 2:
            raise ValueError("lookback_days must be >= 2")

        self._beta_warn = beta_warn_threshold
        self._beta_urgent = beta_urgent_threshold
        self._lookback = lookback_days
        self._min_days = max(min_days, 2)
        self._spy_approx_price = spy_approx_price
        self._btc_approx_price = btc_approx_price

        # Rolling return histories keyed by symbol (including SPY, BTC/USD)
        self._returns: Dict[str, Deque[float]] = {}

        # Optional size-rank mapping (0.0 = smallest, 1.0 = largest)
        # Callers can populate this via set_size_ranks() if desired.
        self._size_ranks: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_prices(self, symbol: str, returns: np.ndarray) -> None:
        """
        Feed a batch of daily returns for ``symbol``.

        This method appends each return in order, keeping only the most recent
        ``lookback_days`` observations.  Safe to call multiple times per cycle
        (e.g. once per new bar received from market data).

        Parameters
        ----------
        symbol:
            Asset symbol.  Use 'SPY' for the equity benchmark and 'BTC/USD'
            for the crypto benchmark.
        returns:
            1-D array of daily returns (fractional, e.g. 0.01 = +1%).
        """
        if symbol not in self._returns:
            self._returns[symbol] = deque(maxlen=self._lookback)
        q = self._returns[symbol]
        for r in returns:
            if np.isfinite(r):
                q.append(float(r))

    def set_size_ranks(self, ranks: Dict[str, float]) -> None:
        """
        Optionally supply normalised market-cap ranks for size factor exposure.

        Parameters
        ----------
        ranks:
            Mapping of symbol -> rank in [0, 1] where 0 = smallest cap, 1 = largest.
        """
        self._size_ranks = {k: float(v) for k, v in ranks.items()}

    def get_exposure(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
    ) -> FactorExposure:
        """
        Compute the current portfolio factor exposure snapshot.

        Parameters
        ----------
        positions:
            Mapping of symbol -> signed quantity (negative = short).
        prices:
            Mapping of symbol -> current mid-price in USD.

        Returns
        -------
        FactorExposure
            All fields populated.  If insufficient data, beta fields are 0
            and hedge_urgency is 'none'.
        """
        result = FactorExposure()
        result.num_positions = len([q for q in positions.values() if q != 0])

        if not positions or not prices:
            return result

        # ------------------------------------------------------------------
        # 1. Build position weights
        # ------------------------------------------------------------------
        weights, total_aum = self._build_weights(positions, prices)
        if total_aum <= 0 or not weights:
            return result

        # ------------------------------------------------------------------
        # 2. Concentration (HHI)
        # ------------------------------------------------------------------
        w_arr = np.array(list(weights.values()), dtype=float)
        result.concentration_hhi = self._hhi(w_arr)

        # Largest single exposure
        if weights:
            result.largest_single_exposure = max(weights, key=lambda s: abs(weights[s]))

        # ------------------------------------------------------------------
        # 3. Portfolio return series (weighted sum of individual returns)
        # ------------------------------------------------------------------
        portfolio_returns = self._build_portfolio_returns(weights)

        n = len(portfolio_returns)
        if n < self._min_days:
            logger.debug(
                "FactorHedger: only %d daily-return observations (need %d); skipping beta",
                n, self._min_days,
            )
            return result

        port_ret = np.array(portfolio_returns, dtype=float)

        # ------------------------------------------------------------------
        # 4. Portfolio volatility (annualised)
        # ------------------------------------------------------------------
        result.portfolio_vol_ann = float(np.std(port_ret, ddof=1)) * np.sqrt(252.0)

        # ------------------------------------------------------------------
        # 5. Equity beta vs SPY
        # ------------------------------------------------------------------
        spy_rets = self._factor_returns(_SPY_KEY, n)
        if spy_rets is not None:
            beta_eq, r2_eq = self._compute_beta_r2(port_ret, spy_rets)
            result.market_beta_equity = beta_eq
            result.r2_equity = r2_eq
        else:
            logger.debug("FactorHedger: SPY returns unavailable; synthetic equity beta set to 0")

        # ------------------------------------------------------------------
        # 6. Crypto beta vs BTC/USD
        # ------------------------------------------------------------------
        btc_rets = self._factor_returns(_BTC_KEY, n)
        if btc_rets is not None:
            beta_cr, r2_cr = self._compute_beta_r2(port_ret, btc_rets)
            result.market_beta_crypto = beta_cr
            result.r2_crypto = r2_cr
        else:
            logger.debug("FactorHedger: BTC/USD returns unavailable; crypto beta set to 0")

        # ------------------------------------------------------------------
        # 7. Momentum factor (5-day trailing average return across held symbols)
        # ------------------------------------------------------------------
        result.momentum_factor = self._compute_momentum_factor(weights)

        # ------------------------------------------------------------------
        # 8. Size factor
        # ------------------------------------------------------------------
        result.size_factor = self._compute_size_factor(weights)

        # ------------------------------------------------------------------
        # 9. Hedge recommendation
        # ------------------------------------------------------------------
        spy_price = prices.get(_SPY_KEY, self._spy_approx_price)
        btc_price = prices.get(_BTC_KEY, self._btc_approx_price)
        rec, urgency = self._build_recommendation(
            result, total_aum, spy_price, btc_price
        )
        result.hedge_recommendation = rec
        result.hedge_urgency = urgency

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_weights(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
    ) -> Tuple[Dict[str, float], float]:
        """Return (symbol -> abs_weight, total_aum).  Skips symbols with no price."""
        notional: Dict[str, float] = {}
        for sym, qty in positions.items():
            if qty == 0:
                continue
            px = prices.get(sym)
            if px is None or px <= 0:
                continue
            notional[sym] = qty * px  # signed notional

        gross_aum = sum(abs(v) for v in notional.values())
        if gross_aum <= 0:
            return {}, 0.0

        weights = {s: v / gross_aum for s, v in notional.items()}
        return weights, gross_aum

    def _build_portfolio_returns(self, weights: Dict[str, float]) -> List[float]:
        """
        Construct the portfolio daily return series as a weighted sum.

        Only days present in ALL held symbols are used (inner join by position
        in the deque).  If a symbol has no return history it is excluded from
        the weighting; remaining weights are renormalised.
        """
        # Collect available series
        avail: Dict[str, List[float]] = {}
        for sym in weights:
            q = self._returns.get(sym)
            if q and len(q) >= self._min_days:
                avail[sym] = list(q)

        if not avail:
            return []

        # Align to common length (take the shortest)
        min_len = min(len(v) for v in avail.values())
        if min_len < self._min_days:
            return []

        # Renormalise weights to available symbols
        gross = sum(abs(weights[s]) for s in avail)
        if gross <= 0:
            return []
        norm_w = {s: weights[s] / gross for s in avail}

        port_rets: List[float] = []
        for i in range(-min_len, 0):  # oldest first
            day_ret = sum(norm_w[s] * avail[s][i] for s in avail)
            port_rets.append(day_ret)
        return port_rets

    def _factor_returns(self, factor_key: str, align_len: int) -> Optional[np.ndarray]:
        """
        Return the last ``align_len`` daily returns for a factor, or None.

        If fewer observations are available than ``align_len``, returns None
        so the caller can skip the beta computation gracefully.
        """
        q = self._returns.get(factor_key)
        if q is None or len(q) < self._min_days:
            return None
        series = list(q)
        if len(series) < align_len:
            return None
        # Take the tail matching align_len
        return np.array(series[-align_len:], dtype=float)

    def _compute_beta(
        self, portfolio_returns: np.ndarray, factor_returns: np.ndarray
    ) -> float:
        """
        OLS regression: portfolio_returns = alpha + beta * factor_returns.

        Returns the beta coefficient.  Returns 0.0 on degenerate input.
        """
        beta, _ = self._ols(portfolio_returns, factor_returns)
        return beta

    def _compute_beta_r2(
        self, portfolio_returns: np.ndarray, factor_returns: np.ndarray
    ) -> Tuple[float, float]:
        """
        OLS regression returning (beta, R²).
        """
        beta, alpha = self._ols(portfolio_returns, factor_returns)
        # R²
        y_hat = alpha + beta * factor_returns
        ss_res = np.sum((portfolio_returns - y_hat) ** 2)
        ss_tot = np.sum((portfolio_returns - portfolio_returns.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return beta, float(max(0.0, r2))

    @staticmethod
    def _ols(
        y: np.ndarray, x: np.ndarray
    ) -> Tuple[float, float]:
        """Return (beta, alpha) from OLS of y on x."""
        if len(x) < 2 or len(y) < 2:
            return 0.0, 0.0
        var_x = np.var(x, ddof=1)
        if var_x < 1e-12:
            return 0.0, float(np.mean(y))
        beta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)
        alpha = float(np.mean(y) - beta * np.mean(x))
        return beta, alpha

    def _hhi(self, weights: np.ndarray) -> float:
        """
        Herfindahl-Hirschman Index.

        Parameters
        ----------
        weights:
            Array of signed or absolute portfolio weights (sum = 1 by convention,
            but function normalises internally).

        Returns
        -------
        float
            HHI in [0, 1].  0 = perfectly diversified, 1 = single position.
        """
        abs_w = np.abs(weights)
        total = abs_w.sum()
        if total < 1e-12:
            return 0.0
        norm = abs_w / total
        return float(np.sum(norm ** 2))

    def _compute_momentum_factor(self, weights: Dict[str, float]) -> float:
        """
        Approximate momentum factor: weighted average 5-day trailing return.
        """
        window = 5
        weighted_mom = 0.0
        total_w = 0.0
        for sym, w in weights.items():
            q = self._returns.get(sym)
            if q is None or len(q) < window:
                continue
            trailing_ret = sum(list(q)[-window:])
            weighted_mom += abs(w) * trailing_ret
            total_w += abs(w)
        if total_w < 1e-9:
            return 0.0
        return float(weighted_mom / total_w)

    def _compute_size_factor(self, weights: Dict[str, float]) -> float:
        """
        Approximate size-factor exposure: weighted average normalised market-cap rank.

        Returns 0 if no size ranks were supplied.
        """
        if not self._size_ranks:
            return 0.0
        weighted_size = 0.0
        total_w = 0.0
        for sym, w in weights.items():
            rank = self._size_ranks.get(sym)
            if rank is None:
                continue
            weighted_size += abs(w) * rank
            total_w += abs(w)
        if total_w < 1e-9:
            return 0.0
        return float(weighted_size / total_w)

    def _build_recommendation(
        self,
        exposure: FactorExposure,
        total_aum: float,
        spy_price: float,
        btc_price: float,
    ) -> Tuple[Optional[str], str]:
        """
        Determine hedge recommendation and urgency level.

        Logic:
        - Only recommend a market hedge when HHI < _HHI_HEDGE_MAX (diversified book).
        - Equity beta hedge: computed from equity-only portion of the book
          (equity_beta * total_AUM / SPY_price → shares).
        - Crypto beta hedge: computed similarly with BTC price.
        - Urgency 'urgent' if |beta| > beta_urgent_threshold.
        - Urgency 'advisory' if |beta| > beta_warn_threshold.
        - Low R² (<_MIN_R2) downgrades urgency by one level.

        Returns (recommendation_string_or_None, urgency_string).
        """
        recs: List[str] = []
        urgency = "none"

        def _urgency(beta_abs: float) -> str:
            if beta_abs > self._beta_urgent:
                return "urgent"
            if beta_abs > self._beta_warn:
                return "advisory"
            return "none"

        # --- Equity beta ---
        abs_eq_beta = abs(exposure.market_beta_equity)
        eq_urgency = _urgency(abs_eq_beta)
        if eq_urgency != "none" and exposure.concentration_hhi < _HHI_HEDGE_MAX:
            # Only R²-penalise to 'advisory' if already 'urgent'
            if exposure.r2_equity < _MIN_R2 and eq_urgency == "urgent":
                eq_urgency = "advisory"
            # Number of SPY shares to short/buy to neutralise excess beta
            # Excess beta = current_beta - sign * warn_threshold
            direction = 1.0 if exposure.market_beta_equity > 0 else -1.0
            excess_beta = abs_eq_beta - self._beta_warn
            hedge_notional = excess_beta * total_aum
            shares = int(round(hedge_notional / max(spy_price, 1.0)))
            if shares > 0:
                side = "SHORT" if direction > 0 else "BUY"
                recs.append(f"{side} SPY {shares} shares (equity beta hedge)")
            urgency = max(urgency, eq_urgency, key=lambda u: {"none": 0, "advisory": 1, "urgent": 2}[u])

        # --- Crypto beta ---
        abs_cr_beta = abs(exposure.market_beta_crypto)
        cr_urgency = _urgency(abs_cr_beta)
        if cr_urgency != "none" and exposure.concentration_hhi < _HHI_HEDGE_MAX:
            if exposure.r2_crypto < _MIN_R2 and cr_urgency == "urgent":
                cr_urgency = "advisory"
            direction = 1.0 if exposure.market_beta_crypto > 0 else -1.0
            excess_beta = abs_cr_beta - self._beta_warn
            hedge_notional = excess_beta * total_aum
            units_raw = hedge_notional / max(btc_price, 1.0)
            # Round to 4 decimal places for crypto
            units = round(units_raw, 4)
            if units > 0:
                side = "SHORT" if direction > 0 else "BUY"
                recs.append(f"{side} BTC {units} units (crypto beta hedge)")
            urgency = max(urgency, cr_urgency, key=lambda u: {"none": 0, "advisory": 1, "urgent": 2}[u])

        recommendation = "; ".join(recs) if recs else None
        return recommendation, urgency

    def __repr__(self) -> str:
        symbols = list(self._returns.keys())
        return (
            f"FactorHedger("
            f"beta_warn={self._beta_warn}, "
            f"beta_urgent={self._beta_urgent}, "
            f"lookback={self._lookback}, "
            f"tracked_symbols={len(symbols)}: {symbols[:5]}{'...' if len(symbols) > 5 else ''})"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fh = FactorHedger()
    # Feed dummy returns (including SPY so beta can be computed)
    np.random.seed(42)
    spy_rets = np.random.randn(20) * 0.01
    fh.update_prices("SPY", spy_rets)
    for sym in ["AAPL", "MSFT", "GOOGL"]:
        # Symbols are 80% correlated with SPY plus idiosyncratic noise
        fh.update_prices(sym, 0.8 * spy_rets + np.random.randn(20) * 0.005)

    exposure = fh.get_exposure(
        positions={"AAPL": 10, "MSFT": 5, "GOOGL": 3},
        prices={"AAPL": 175.0, "MSFT": 420.0, "GOOGL": 180.0},
    )
    print(f"Equity beta:       {exposure.market_beta_equity:.3f}")
    print(f"Crypto beta:       {exposure.market_beta_crypto:.3f}")
    print(f"Portfolio vol ann: {exposure.portfolio_vol_ann:.4f}")
    print(f"Momentum factor:   {exposure.momentum_factor:.4f}")
    print(f"HHI:               {exposure.concentration_hhi:.3f}")
    print(f"Largest exposure:  {exposure.largest_single_exposure}")
    print(f"Hedge urgency:     {exposure.hedge_urgency}")
    print(f"Hedge rec:         {exposure.hedge_recommendation}")
    print(f"R2 (equity):       {exposure.r2_equity:.3f}")
    print(f"Num positions:     {exposure.num_positions}")
