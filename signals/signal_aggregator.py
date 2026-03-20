"""
signals/signal_aggregator.py — Independent Signal Vote Combiner
================================================================
Combines votes from independent signal sources (funding rate, candlestick
patterns, and future sources) into a confidence adjustment and optional
entry block for the main execution loop.

Design principles:
- Does NOT replace the ML signal — it adjusts confidence only
- Each source votes independently; they don't know about each other
- Strong multi-source agreement = confidence boost (max +8%)
- Strong multi-source contradiction = confidence penalty (max -15%) + possible block
- Zero impact when no votes exist (neutral pass-through)
"""

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
    Combines a list of SignalVote objects with the primary ML signal to
    produce an adjusted confidence and a block flag.
    """

    def __init__(self) -> None:
        self._agree_boost = float(
            getattr(ApexConfig, "SIGNAL_AGGREGATOR_AGREE_BOOST", 0.08)
        )
        self._contra_penalty = float(
            getattr(ApexConfig, "SIGNAL_AGGREGATOR_CONTRA_PENALTY", 0.15)
        )
        self._contra_threshold = float(
            getattr(ApexConfig, "SIGNAL_AGGREGATOR_CONTRA_THRESHOLD", 0.35)
        )
        self._min_conf_gate = float(
            getattr(ApexConfig, "SIGNAL_AGGREGATOR_MIN_CONF_GATE", 0.60)
        )

    def combine(
        self,
        primary_signal: float,
        votes: List[SignalVote],
        primary_confidence: float,
        asset_class: str,
    ) -> Tuple[float, bool]:
        """
        Returns (adjusted_confidence, block_entry).

        For each applicable vote:
          - alignment > 0 and vote.confidence > 0.20:
              confidence += AGREE_BOOST * vote.confidence
          - alignment < -CONTRA_THRESHOLD and vote.confidence > 0.20:
              confidence -= CONTRA_PENALTY * vote.confidence

        block_entry = True only if ALL applicable votes contradict AND
        adjusted_confidence < MIN_CONF_GATE.
        """
        ac_lower = str(asset_class).lower()
        applicable = [
            v for v in votes
            if v.applies_to == "all"
            or v.applies_to == ac_lower
            or (v.applies_to == "crypto" and "crypto" in ac_lower)
            or (v.applies_to == "equity" and "equity" in ac_lower)
        ]

        if not applicable:
            return primary_confidence, False

        adj_conf = primary_confidence
        contradicting_count = 0
        total_applicable = len(applicable)

        for vote in applicable:
            if vote.confidence < 0.10:   # ignore very low-confidence votes
                continue

            alignment = float(primary_signal) * float(vote.signal)

            if alignment > 0 and vote.confidence >= 0.20:
                boost = self._agree_boost * vote.confidence
                adj_conf = min(1.0, adj_conf + boost)
                logger.debug(
                    "SignalAggregator: %s agrees (signal=%.3f conf=%.2f) → conf +%.3f",
                    vote.source, vote.signal, vote.confidence, boost,
                )

            elif alignment < -self._contra_threshold and vote.confidence >= 0.20:
                penalty = self._contra_penalty * vote.confidence
                adj_conf = max(0.0, adj_conf - penalty)
                contradicting_count += 1
                logger.debug(
                    "SignalAggregator: %s contradicts (signal=%.3f conf=%.2f) → conf -%.3f",
                    vote.source, vote.signal, vote.confidence, penalty,
                )

        # Block only if ALL applicable votes contradict AND confidence is too low
        block_entry = (
            contradicting_count == total_applicable
            and total_applicable > 0
            and adj_conf < self._min_conf_gate
        )

        return float(adj_conf), block_entry
