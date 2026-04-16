"""
signals/signal_aggregator.py — Independent Signal Vote Combiner (v2)
=====================================================================
Combines votes from independent signal sources (funding rate, candlestick
patterns, and future sources) into a confidence adjustment and optional
entry block for the main execution loop.

Design principles (v2):
- Does NOT replace the ML signal — it adjusts confidence only.
- Each source votes independently; they don't know about each other.
- **Primary-signal gate:** a weak/noise primary signal (|s| < MIN_PRIMARY_ABS)
  receives no boost from aligned votes. The previous version boosted
  confidence whenever ``primary_signal * vote.signal > 0`` which fires even
  when the primary is near-zero noise.
- **Vote-count normalization:** multi-vote boosts are scaled by
  ``count^COUNT_EXPONENT`` (default 0.5 → √N) so four weak aligned votes
  can't outvote a single strong contrary vote in pure magnitude.
- **Absolute caps:** total boost ≤ MAX_BOOST, total penalty ≤ MAX_PENALTY,
  preventing runaway stacking on any single cycle.
- Zero impact when no votes exist (neutral pass-through).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

from config import ApexConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vote type (produced by each signal module)
# ---------------------------------------------------------------------------
@dataclass
class SignalVote:
    signal: float       # [-1, 1] direction from this source
    confidence: float   # [0, 1] how confident this source is
    source: str         # e.g. "funding_rate", "pattern"
    applies_to: str     # "crypto" | "equity" | "all"


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
class SignalAggregator:
    """
    Combines a list of :class:`SignalVote` objects with the primary ML
    signal to produce an adjusted confidence and a block flag.

    All thresholds sourced from :class:`ApexConfig`. No magic numbers.
    """

    def __init__(self) -> None:
        self._agree_boost: float = float(ApexConfig.SIGNAL_AGGREGATOR_AGREE_BOOST)
        self._contra_penalty: float = float(ApexConfig.SIGNAL_AGGREGATOR_CONTRA_PENALTY)
        self._contra_threshold: float = float(ApexConfig.SIGNAL_AGGREGATOR_CONTRA_THRESHOLD)
        self._min_conf_gate: float = float(ApexConfig.SIGNAL_AGGREGATOR_MIN_CONF_GATE)
        self._min_primary_abs: float = float(ApexConfig.SIGNAL_AGGREGATOR_MIN_PRIMARY_ABS)
        self._count_exponent: float = float(ApexConfig.SIGNAL_AGGREGATOR_COUNT_EXPONENT)
        self._max_boost: float = float(ApexConfig.SIGNAL_AGGREGATOR_MAX_BOOST)
        self._max_penalty: float = float(ApexConfig.SIGNAL_AGGREGATOR_MAX_PENALTY)

        if not (0.0 < self._count_exponent <= 1.0):
            raise ValueError(
                f"SIGNAL_AGGREGATOR_COUNT_EXPONENT must be in (0,1], "
                f"got {self._count_exponent!r}"
            )

    def combine(
        self,
        primary_signal: float,
        votes: List[SignalVote],
        primary_confidence: float,
        asset_class: str,
    ) -> Tuple[float, bool]:
        """
        Combine external votes with the primary ML signal/confidence.

        Args:
            primary_signal: The ML model's directional score in ``[-1, 1]``.
            votes: Independent votes. An empty list is a no-op pass-through.
            primary_confidence: The ML model's confidence in ``[0, 1]``.
            asset_class: ``"crypto"``, ``"equity"``, ``"fx"`` or a label
                containing one of those substrings (e.g.
                ``"crypto_perp"``). Used to filter votes whose
                ``applies_to`` attribute is not ``"all"``.

        Returns:
            Tuple ``(adjusted_confidence, block_entry)``:

            - ``adjusted_confidence``: The clipped confidence after
              applying boosts/penalties. Always in ``[0, 1]``.
            - ``block_entry``: ``True`` iff every applicable vote
              contradicts the primary *and* the adjusted confidence is
              below ``SIGNAL_AGGREGATOR_MIN_CONF_GATE``.

        Raises:
            TypeError: If ``votes`` is not iterable.
        """
        if votes is None:
            votes = []
        try:
            ac_lower = str(asset_class).lower()
        except Exception:
            ac_lower = ""

        applicable = [
            v for v in votes
            if v.applies_to == "all"
            or v.applies_to == ac_lower
            or (v.applies_to == "crypto" and "crypto" in ac_lower)
            or (v.applies_to == "equity" and "equity" in ac_lower)
        ]
        if not applicable:
            return float(primary_confidence), False

        primary_abs = abs(float(primary_signal))
        primary_gated = primary_abs >= self._min_primary_abs

        agree_accum: float = 0.0
        contra_accum: float = 0.0
        contradicting_count = 0
        agreeing_count = 0
        total_applicable = 0

        for vote in applicable:
            vc = float(vote.confidence)
            if vc < 0.10:
                continue
            total_applicable += 1
            alignment = float(primary_signal) * float(vote.signal)

            if alignment > 0.0 and vc >= 0.20 and primary_gated:
                # Weighted boost contribution — accumulated, normalized later.
                agree_accum += self._agree_boost * vc
                agreeing_count += 1
                logger.debug(
                    "SignalAggregator: %s agrees (s=%.3f c=%.2f)",
                    vote.source, vote.signal, vc,
                )
            elif alignment < -self._contra_threshold and vc >= 0.20:
                contra_accum += self._contra_penalty * vc
                contradicting_count += 1
                logger.debug(
                    "SignalAggregator: %s contradicts (s=%.3f c=%.2f)",
                    vote.source, vote.signal, vc,
                )

        # Count normalisation: divide by count^exponent so k votes produce
        # boost ~= single_vote_boost * k^(1 - exponent). With exponent=0.5
        # and k=4: net boost = avg * 4^0.5 = 2× single-vote boost (not 4×).
        if agreeing_count > 0:
            agree_accum = (agree_accum / agreeing_count) * (agreeing_count ** (1.0 - self._count_exponent))
        if contradicting_count > 0:
            contra_accum = (contra_accum / contradicting_count) * (contradicting_count ** (1.0 - self._count_exponent))

        # Absolute caps
        if agree_accum > self._max_boost:
            agree_accum = self._max_boost
        if contra_accum > self._max_penalty:
            contra_accum = self._max_penalty

        adj_conf = float(primary_confidence) + agree_accum - contra_accum
        if adj_conf < 0.0:
            adj_conf = 0.0
        elif adj_conf > 1.0:
            adj_conf = 1.0

        block_entry = (
            total_applicable > 0
            and contradicting_count == total_applicable
            and adj_conf < self._min_conf_gate
        )

        return float(adj_conf), bool(block_entry)
