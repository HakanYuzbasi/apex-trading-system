"""
risk/fee_aware_edge_gate.py — Fee-aware entry edge gate.
========================================================

Blocks entries whose expected directional edge cannot clear the *round-trip*
transaction cost (fees + slippage + half-spread) by a configurable safety
margin. This directly eliminates the "paper-cut" trades that individually
look fine on the ML signal but bleed ROI in aggregate because fees + spread
swallow the entire expected move.

**Why this is ROI-positive:**

For every candidate trade, the system already has:

  * ``expected_return_pct``  — the ML's point estimate of realised return
  * ``confidence``           — the ML's self-rated certainty
  * an ``asset_class``       — used to look up default cost assumptions
  * optional ``realised_bps``— from the exec layer's fill history

This module computes:

    required_edge = round_trip_cost * MIN_EDGE_COST_RATIO

and blocks the trade whenever ``confidence_adjusted_edge < required_edge``.

A 1.50 ratio (the default) means the expected edge must be at least 150 %
of cost — i.e. the trade needs to overcome costs **and** leave a 0.5×
cost buffer for realised slippage variance. A 1.0 ratio (break-even) is
*guaranteed* to lose money on average due to that realised-slippage
variance — hence the default is strictly above 1.0.

**Zero side effects.** The gate is a pure function of its inputs and the
runtime config. Preserves the normal-case hot-path overhead at <10 µs.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from config import ApexConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeGateDecision:
    """
    Immutable result of a fee-aware edge-gate evaluation.

    Attributes:
        allowed: ``True`` iff the trade should proceed.
        expected_edge_pct: The ML's expected return (as a fraction, not bps).
        adjusted_edge_pct: ``expected_edge_pct × confidence`` — the
            expected edge discounted by ML self-rated certainty.
        round_trip_cost_pct: Estimated round-trip cost as a fraction.
        required_edge_pct: The adjusted edge required to pass
            (``round_trip_cost_pct × MIN_EDGE_COST_RATIO``).
        reason: Short explanation suitable for logging / audit trail.
    """
    allowed: bool
    expected_edge_pct: float
    adjusted_edge_pct: float
    round_trip_cost_pct: float
    required_edge_pct: float
    reason: str


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def _default_cost_bps(asset_class: str) -> float:
    """
    Return the default one-way cost (bps) for the given asset class.

    Used only when the execution layer hasn't supplied a realised cost
    estimate (e.g. cold-start or first trade of the session).
    """
    ac = (asset_class or "").lower()
    if "crypto" in ac:
        return float(getattr(ApexConfig, "FEE_AWARE_DEFAULT_CRYPTO_BPS", 18.0))
    if "fx" in ac or "forex" in ac:
        return float(getattr(ApexConfig, "FEE_AWARE_DEFAULT_FX_BPS", 4.0))
    # Equity is the default bucket (covers "equity", "equity_us", "stock", …)
    return float(getattr(ApexConfig, "FEE_AWARE_DEFAULT_EQUITY_BPS", 8.0))


def _round_trip_cost_pct(
    asset_class: str,
    realised_one_way_bps: Optional[float] = None,
    half_spread_bps: Optional[float] = None,
) -> float:
    """
    Convert the cost inputs into a round-trip cost **as a fraction**
    (e.g. ``0.0016`` = 16 bps round-trip).

    The round-trip cost is:

        ( 2 × one_way_cost_bps )  +  ( 2 × half_spread_bps )

    expressed as a fraction of price (÷ 10 000). The half-spread is counted
    *in addition to* the realised/default one-way because the one-way
    number often reflects the fee + slippage alone, not the crossing of
    the spread.

    Args:
        asset_class: Used for the default-cost fallback.
        realised_one_way_bps: If provided, replaces the default — trusted
            as the best available cost estimate from the exec layer.
        half_spread_bps: Optional additional cost component (spread / 2 at
            the trade's price level). Added on both entry and exit.

    Returns:
        Round-trip cost as a fraction. Always ≥ 0.
    """
    one_way = float(realised_one_way_bps) if realised_one_way_bps is not None \
        else _default_cost_bps(asset_class)
    if not math.isfinite(one_way) or one_way < 0.0:
        one_way = _default_cost_bps(asset_class)

    hs = float(half_spread_bps) if half_spread_bps is not None else 0.0
    if not math.isfinite(hs) or hs < 0.0:
        hs = 0.0

    return ((2.0 * one_way) + (2.0 * hs)) / 10_000.0


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
class FeeAwareEdgeGate:
    """
    Config-driven fee-aware edge gate.

    Usage:
        gate = FeeAwareEdgeGate()
        decision = gate.evaluate(
            expected_return_pct=0.0080,
            confidence=0.55,
            asset_class="equity_us",
            realised_one_way_bps=7.5,
            half_spread_bps=1.2,
        )
        if not decision.allowed:
            logger.info("Trade blocked: %s", decision.reason)

    All thresholds are sourced from :class:`ApexConfig`:

      * ``FEE_AWARE_EDGE_GATE_ENABLED``     — master kill switch
      * ``FEE_AWARE_MIN_EDGE_COST_RATIO``   — required edge / cost
      * ``FEE_AWARE_DEFAULT_{ASSET}_BPS``   — cold-start cost fallback

    The class holds *no* mutable state, so a single instance can be safely
    shared across threads.
    """

    def __init__(self) -> None:
        self._enabled: bool = bool(
            getattr(ApexConfig, "FEE_AWARE_EDGE_GATE_ENABLED", True)
        )
        self._min_ratio: float = float(
            getattr(ApexConfig, "FEE_AWARE_MIN_EDGE_COST_RATIO", 1.50)
        )
        if self._min_ratio < 1.0:
            # A ratio below 1.0 is mathematically negative-expectancy; clamp.
            logger.warning(
                "FEE_AWARE_MIN_EDGE_COST_RATIO=%.2f < 1.0 "
                "(negative-expectancy). Clamping to 1.0.",
                self._min_ratio,
            )
            self._min_ratio = 1.0

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        expected_return_pct: float,
        confidence: float,
        asset_class: str,
        realised_one_way_bps: Optional[float] = None,
        half_spread_bps: Optional[float] = None,
    ) -> EdgeGateDecision:
        """
        Evaluate whether a trade's confidence-weighted expected edge covers
        round-trip costs with the required safety margin.

        Args:
            expected_return_pct: Expected return from the ML signal, as a
                signed fraction (``0.01`` = +1 %). Sign is preserved — the
                gate uses |value| for the edge comparison, but it is
                returned unchanged for audit.
            confidence: ML confidence in ``[0, 1]``. The expected edge is
                discounted by this factor before comparison to cost.
            asset_class: ``"equity"``, ``"crypto"``, ``"fx"``, or a label
                containing one of those substrings.
            realised_one_way_bps: Realised one-way cost from the execution
                layer (fees + slippage). If ``None``, the per-asset-class
                default from ApexConfig is used.
            half_spread_bps: Half of the bid-ask spread at the trade's
                price level, in bps. Added on both entry and exit.

        Returns:
            :class:`EdgeGateDecision`. ``allowed`` is ``True`` when either
            the gate is disabled via config OR the adjusted edge meets the
            required threshold.

        Raises:
            ValueError: If ``confidence`` is outside ``[0, 1]`` or
                ``expected_return_pct`` is non-finite.
        """
        if not math.isfinite(expected_return_pct):
            raise ValueError(
                f"expected_return_pct must be finite, got {expected_return_pct!r}"
            )
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {confidence!r}"
            )

        expected_edge = abs(float(expected_return_pct))
        adjusted_edge = expected_edge * float(confidence)
        cost_pct = _round_trip_cost_pct(
            asset_class=asset_class,
            realised_one_way_bps=realised_one_way_bps,
            half_spread_bps=half_spread_bps,
        )
        required = cost_pct * self._min_ratio

        if not self._enabled:
            return EdgeGateDecision(
                allowed=True,
                expected_edge_pct=float(expected_return_pct),
                adjusted_edge_pct=adjusted_edge,
                round_trip_cost_pct=cost_pct,
                required_edge_pct=required,
                reason="gate_disabled",
            )

        if adjusted_edge >= required:
            return EdgeGateDecision(
                allowed=True,
                expected_edge_pct=float(expected_return_pct),
                adjusted_edge_pct=adjusted_edge,
                round_trip_cost_pct=cost_pct,
                required_edge_pct=required,
                reason=(
                    f"edge {adjusted_edge*100:.2f}% >= required "
                    f"{required*100:.2f}% (cost {cost_pct*100:.2f}%)"
                ),
            )

        return EdgeGateDecision(
            allowed=False,
            expected_edge_pct=float(expected_return_pct),
            adjusted_edge_pct=adjusted_edge,
            round_trip_cost_pct=cost_pct,
            required_edge_pct=required,
            reason=(
                f"edge {adjusted_edge*100:.2f}% < required "
                f"{required*100:.2f}% (cost {cost_pct*100:.2f}% × "
                f"{self._min_ratio:.2f})"
            ),
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_GATE_SINGLETON: Optional[FeeAwareEdgeGate] = None


def get_fee_aware_edge_gate() -> FeeAwareEdgeGate:
    """Return the process-wide :class:`FeeAwareEdgeGate` instance."""
    global _GATE_SINGLETON
    if _GATE_SINGLETON is None:
        _GATE_SINGLETON = FeeAwareEdgeGate()
    return _GATE_SINGLETON
