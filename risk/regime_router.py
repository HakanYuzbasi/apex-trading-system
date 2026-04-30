"""
risk/regime_router.py — Composite Market Regime Router
=======================================================
Introduced 2026-04-28 following the forensic audit that found:
  * MarketRegimeDetector and VIXRegimeManager existed but were not wired
    into the live execution path.
  * Crypto pairs used identical notional sizing to equity pairs despite
    carrying 2-4× higher beta.
  * 566 daily "insufficient balance" rejections from oversized concurrent
    crypto orders.

Architecture
------------
This module is a *thin coordination layer* that:
  1. Queries the global VIXRegimeManager singleton for macro equity-vol state.
  2. Optionally runs MarketRegimeDetector on recent BTC returns as a
     crypto-specific micro-regime proxy (because VIX only tracks S&P 500).
  3. Merges both signals into a unified CompositeRegime enum.
  4. Returns a RegimeDecision dataclass with:
       - ``notional_multiplier`` – applied to all legs this cycle
       - ``block_new_entries`` – True only in HIGH_VOL_PANIC (exits only)
       - ``crypto_beta_scalar`` – additional multiplier for crypto legs
         (default 0.55 ≈ 1/crypto_beta of ~1.8×)

Usage (in run_global_harness_v3.py)
------------------------------------
    from risk.regime_router import RegimeRouter

    # At startup (once):
    _regime_router = RegimeRouter(
        crypto_beta_scalar=float(os.getenv("CRYPTO_BETA_SCALAR", "0.55"))
    )

    # Inside _resolve_notional() or the main cycle loop:
    decision = _regime_router.evaluate()
    notional *= decision.notional_multiplier
    if is_crypto:
        notional *= decision.crypto_beta_scalar
    if decision.block_new_entries:
        return  # skip entry signals; exits still process
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Composite regime taxonomy (3 states matching the audit spec)
# ---------------------------------------------------------------------------


class CompositeRegime(Enum):
    """
    Unified regime label produced by RegimeRouter.

    TRENDING     – Bull or Bear momentum market.  Mean-reversion pairs may
                   underperform; the system continues trading with standard
                   notionals but tightens entry Z-scores mentally.
    RANGING      – Sideways / mean-reverting.  Optimal for Kalman pairs.
                   Notionals are mildly reduced to reflect lower edge.
    HIGH_VOL_PANIC – Extreme volatility spike or liquidation cascade.
                   New pair entries are blocked; exits are still processed.
                   Notionals halved for any ongoing positions.
    """
    TRENDING       = "trending"
    RANGING        = "ranging"
    HIGH_VOL_PANIC = "high_vol_panic"


# ---------------------------------------------------------------------------
# Notional multipliers and entry-block flags per regime
# ---------------------------------------------------------------------------

_NOTIONAL_MULTIPLIERS: dict[CompositeRegime, float] = {
    CompositeRegime.TRENDING:       1.00,   # no reduction — momentum favours size
    CompositeRegime.RANGING:        1.00,   # ranging = peak environment for mean-reversion pairs
    CompositeRegime.HIGH_VOL_PANIC: 0.40,   # severe reduction, survival mode
}

_BLOCK_NEW_ENTRIES: dict[CompositeRegime, bool] = {
    CompositeRegime.TRENDING:       False,
    CompositeRegime.RANGING:        False,
    CompositeRegime.HIGH_VOL_PANIC: True,   # exits only in panic
}


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class RegimeDecision:
    """
    Single-cycle regime decision returned by :meth:`RegimeRouter.evaluate`.

    Attributes
    ----------
    regime:
        The merged :class:`CompositeRegime` classification.
    notional_multiplier:
        Scalar to apply to *all* leg notionals this cycle (equity + crypto).
    block_new_entries:
        When ``True`` the caller should skip signal-based entry logic and
        only process exits / stops for existing positions.
    crypto_beta_scalar:
        Additional scalar applied **only** to crypto legs, on top of
        ``notional_multiplier``, to account for crypto's higher beta.
    vix_regime:
        Raw VIX regime label for telemetry (e.g. ``"normal"``, ``"fear"``).
    crypto_regime:
        Raw crypto micro-regime label for telemetry (e.g. ``"SIDEWAYS"``).
    reason:
        Human-readable explanation suitable for log lines.
    timestamp:
        UTC time of evaluation.
    """
    regime: CompositeRegime
    notional_multiplier: float
    block_new_entries: bool
    crypto_beta_scalar: float
    vix_regime: str
    crypto_regime: str
    reason: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# RegimeRouter
# ---------------------------------------------------------------------------


class RegimeRouter:
    """
    Merges VIX-based macro regime with a crypto-specific price-action regime
    to produce a single :class:`RegimeDecision` per evaluation cycle.

    Parameters
    ----------
    crypto_beta_scalar:
        Multiplier applied to crypto leg notionals in *addition* to the
        regime-level ``notional_multiplier``.  Default 0.55 ≈ 1/1.82 which
        normalises a crypto position's dollar-risk to an equivalent equity
        risk unit (assuming crypto beta ≈ 1.8× S&P).
    cache_seconds:
        Minimum seconds between VIX fetches (passed through to the singleton
        VIXRegimeManager's cache; the router itself also short-circuits).
    btc_lookback:
        Number of BTC bars used by MarketRegimeDetector for the crypto
        micro-regime.  Requires at least this many price observations to be
        non-trivial; otherwise defaults to RANGING.
    panic_vol_threshold:
        If the 5-day annualised BTC return volatility exceeds this multiple
        of the 60-day baseline, the regime is escalated to HIGH_VOL_PANIC
        regardless of the VIX reading.  Default 3.5×.
    """

    def __init__(
        self,
        crypto_beta_scalar: float = 0.55,
        cache_seconds: float = 60.0,
        btc_lookback: int = 60,
        panic_vol_threshold: float = 3.5,
    ) -> None:
        if not (0.0 < crypto_beta_scalar <= 1.0):
            raise ValueError(
                f"crypto_beta_scalar must be in (0, 1], got {crypto_beta_scalar!r}"
            )
        self._crypto_beta_scalar = float(crypto_beta_scalar)
        self._cache_seconds = float(cache_seconds)
        self._btc_lookback = int(btc_lookback)
        self._panic_vol_threshold = float(panic_vol_threshold)

        self._lock = threading.Lock()
        self._last_decision: Optional[RegimeDecision] = None
        self._last_eval: Optional[datetime] = None

        # Rolling BTC price buffer for crypto micro-regime detection
        # Filled by callers via record_btc_price(); evaluated lazily.
        self._btc_prices: list[float] = []

        # Late import to avoid circular deps at module level
        try:
            from risk.vix_regime_manager import get_global_vix_manager
            self._vix_manager_fn = get_global_vix_manager
        except Exception as exc:
            logger.warning("RegimeRouter: VIXRegimeManager unavailable: %s", exc)
            self._vix_manager_fn = None  # type: ignore[assignment]

        try:
            from market.market_regime_detector import MarketRegimeDetector
            self._crypto_detector: Optional[object] = MarketRegimeDetector()
        except Exception as exc:
            logger.warning("RegimeRouter: MarketRegimeDetector unavailable: %s", exc)
            self._crypto_detector = None

        logger.info(
            "✅ RegimeRouter initialised | crypto_beta_scalar=%.2f | panic_vol_threshold=%.1fx",
            self._crypto_beta_scalar,
            self._panic_vol_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_btc_price(self, price: float) -> None:
        """
        Feed the latest BTC/USD close price into the router's rolling buffer.

        Call once per bar from the data ingestion path so the crypto
        micro-regime detector always has fresh data.  Thread-safe.

        Args:
            price: BTC/USD close price (must be positive finite).
        """
        if not np.isfinite(price) or price <= 0:
            return
        with self._lock:
            self._btc_prices.append(float(price))
            # Keep 2× lookback to have a stable historical baseline
            if len(self._btc_prices) > self._btc_lookback * 2:
                self._btc_prices = self._btc_prices[-(self._btc_lookback * 2):]

    def evaluate(self) -> RegimeDecision:
        """
        Compute the current :class:`RegimeDecision`.

        Results are cached for ``cache_seconds`` to avoid hammering yfinance
        and the regime detector on every sub-second event loop iteration.
        Thread-safe.

        Returns
        -------
        RegimeDecision
            Always returns a valid decision — falls back to RANGING with
            standard multipliers on any error so the system keeps trading.
        """
        now = datetime.utcnow()
        with self._lock:
            if (
                self._last_decision is not None
                and self._last_eval is not None
                and (now - self._last_eval).total_seconds() < self._cache_seconds
            ):
                return self._last_decision

        decision = self._compute(now)
        with self._lock:
            self._last_decision = decision
            self._last_eval = now
        return decision

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute(self, now: datetime) -> RegimeDecision:
        """Build a fresh RegimeDecision — called when cache is stale."""
        vix_regime_label = "unknown"
        crypto_regime_label = "UNKNOWN"

        # ── 1. VIX macro regime ────────────────────────────────────────
        try:
            if self._vix_manager_fn is not None:
                vix_mgr = self._vix_manager_fn()
                vix_state = vix_mgr.get_current_state()
                vix_regime_label = vix_state.regime.value
                vix_risk_mult = vix_state.risk_multiplier
            else:
                vix_risk_mult = 1.0
        except Exception as exc:
            logger.debug("RegimeRouter: VIX fetch error: %s", exc)
            vix_risk_mult = 1.0
            vix_regime_label = "error"

        # ── 2. Crypto micro-regime (BTC price action) ──────────────────
        with self._lock:
            btc_prices_copy = list(self._btc_prices)

        crypto_vol_ratio = 1.0
        if self._crypto_detector is not None and len(btc_prices_copy) >= self._btc_lookback:
            try:
                prices = pd.Series(btc_prices_copy, dtype=float)
                returns = prices.pct_change().dropna()
                result = self._crypto_detector.detect_regime(  # type: ignore[attr-defined]
                    returns, lookback=self._btc_lookback
                )
                crypto_regime_label = result.get("regime", "UNKNOWN")
                crypto_vol_ratio = float(result.get("vol_ratio", 1.0))
            except Exception as exc:
                logger.debug("RegimeRouter: crypto regime detection error: %s", exc)
        else:
            # Not enough data yet — assume RANGING (mean-reversion friendly)
            crypto_regime_label = "RANGING_DEFAULT"

        # ── 3. Panic override — crypto-specific vol spike ──────────────
        # Even when VIX is normal, if BTC's realised vol explodes we enter panic.
        if crypto_vol_ratio >= self._panic_vol_threshold:
            regime = CompositeRegime.HIGH_VOL_PANIC
            reason = (
                f"Crypto panic: BTC vol_ratio={crypto_vol_ratio:.2f}x "
                f"(threshold={self._panic_vol_threshold:.1f}x) | "
                f"VIX={vix_regime_label}"
            )
        # ── 4. VIX-driven high-vol escalation ─────────────────────────
        elif vix_regime_label in ("panic", "fear"):
            regime = CompositeRegime.HIGH_VOL_PANIC
            reason = f"VIX regime={vix_regime_label} | crypto_vol_ratio={crypto_vol_ratio:.2f}x"
        # ── 5. Trending regime — clear directional bias ────────────────
        elif crypto_regime_label in ("BULL", "BEAR") and vix_regime_label in (
            "normal", "elevated", "complacency"
        ):
            regime = CompositeRegime.TRENDING
            reason = (
                f"Trending: crypto={crypto_regime_label} | "
                f"VIX={vix_regime_label} | vol_ratio={crypto_vol_ratio:.2f}x"
            )
        # ── 6. Default: RANGING (mean-reversion optimal) ──────────────
        else:
            regime = CompositeRegime.RANGING
            reason = (
                f"Ranging: crypto={crypto_regime_label} | "
                f"VIX={vix_regime_label} | vol_ratio={crypto_vol_ratio:.2f}x"
            )

        # ── 7. Final notional multiplier — min of VIX mult and regime table ─
        table_mult = _NOTIONAL_MULTIPLIERS[regime]
        # vix_risk_mult ∈ [0.45, 1.0]; blend it into the table mult
        # so a VIX spike squeezes further even within a TRENDING call.
        notional_mult = round(min(table_mult, vix_risk_mult * table_mult / table_mult) * table_mult / table_mult, 4)
        # Simplified: effective_mult = regime_table_mult × vix_risk_mult
        effective_mult = round(table_mult * vix_risk_mult, 4)
        effective_mult = max(0.10, min(1.50, effective_mult))

        decision = RegimeDecision(
            regime=regime,
            notional_multiplier=effective_mult,
            block_new_entries=_BLOCK_NEW_ENTRIES[regime],
            crypto_beta_scalar=self._crypto_beta_scalar,
            vix_regime=vix_regime_label,
            crypto_regime=crypto_regime_label,
            reason=reason,
            timestamp=now,
        )

        logger.info(
            "🌍 REGIME | %s | notional_mult=%.2f | block_entries=%s | "
            "crypto_beta=%.2f | VIX=%s | crypto=%s",
            regime.value,
            effective_mult,
            _BLOCK_NEW_ENTRIES[regime],
            self._crypto_beta_scalar,
            vix_regime_label,
            crypto_regime_label,
        )
        return decision


# ---------------------------------------------------------------------------
# Module-level singleton (process-wide, thread-safe)
# ---------------------------------------------------------------------------

_GLOBAL_ROUTER: Optional[RegimeRouter] = None
_GLOBAL_ROUTER_LOCK = threading.Lock()


def get_global_regime_router(
    *,
    crypto_beta_scalar: float = 0.55,
) -> RegimeRouter:
    """
    Return the process-wide :class:`RegimeRouter` singleton.

    Safe to call from any thread.  On first call the instance is created
    with the supplied ``crypto_beta_scalar``; subsequent calls return the
    existing instance (the kwarg is ignored after first init).

    Args:
        crypto_beta_scalar: Passed to :class:`RegimeRouter` on first call.

    Returns:
        The singleton :class:`RegimeRouter`.
    """
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        with _GLOBAL_ROUTER_LOCK:
            if _GLOBAL_ROUTER is None:
                _GLOBAL_ROUTER = RegimeRouter(
                    crypto_beta_scalar=crypto_beta_scalar
                )
    return _GLOBAL_ROUTER
