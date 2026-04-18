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
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

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

        # Dynamic per-source weights. Start uniform; refit via softmax over
        # rolling mean PnL once a source accumulates ML_WEIGHT_UPDATE_BARS
        # outcomes. Weights multiply the source's boost/penalty contribution
        # so historically-accurate sources dominate and historically-bad
        # sources are down-weighted (never to zero — ML_WEIGHT_FLOOR).
        self._weight_temperature: float = float(ApexConfig.ML_WEIGHT_TEMPERATURE)
        self._weight_update_bars: int = int(ApexConfig.ML_WEIGHT_UPDATE_BARS)
        self._weight_floor: float = float(ApexConfig.ML_WEIGHT_FLOOR)
        self._source_pnl_hist: Dict[str, Deque[float]] = {}
        self._source_weight: Dict[str, float] = {}
        # Rolling buffer length — cap at 4× update window so refits use the
        # most-recent evidence. Always at least 1 to avoid zero-size deque.
        self._pnl_buffer_size: int = max(1, self._weight_update_bars * 4)

        if not (0.0 < self._count_exponent <= 1.0):
            raise ValueError(
                f"SIGNAL_AGGREGATOR_COUNT_EXPONENT must be in (0,1], "
                f"got {self._count_exponent!r}"
            )
        if self._weight_temperature < 0.0:
            raise ValueError(
                f"ML_WEIGHT_TEMPERATURE must be >= 0, "
                f"got {self._weight_temperature!r}"
            )
        if not (0.0 <= self._weight_floor < 1.0):
            raise ValueError(
                f"ML_WEIGHT_FLOOR must be in [0,1), got {self._weight_floor!r}"
            )

    # ── Dynamic weight learning ──────────────────────────────────────────────

    def record_source_outcome(self, source: str, pnl_pct: float) -> None:
        """
        Feed a realised trade outcome back to the aggregator so per-source
        weights can adapt.

        Args:
            source: Source label matching :attr:`SignalVote.source` (e.g.
                ``"funding_rate"``, ``"pattern"``). Required — empty strings
                are ignored.
            pnl_pct: Realised PnL as a fraction (e.g. ``+0.012`` = +1.2%).

        Raises:
            TypeError: If ``source`` is not a string.
        """
        if not isinstance(source, str):
            raise TypeError(f"source must be str, got {type(source).__name__}")
        if not source or self._weight_update_bars <= 0:
            return
        try:
            pnl = float(pnl_pct)
        except (TypeError, ValueError):
            return
        if not math.isfinite(pnl):
            return

        buf = self._source_pnl_hist.get(source)
        if buf is None:
            buf = deque(maxlen=self._pnl_buffer_size)
            self._source_pnl_hist[source] = buf
        buf.append(pnl)

        if len(buf) >= self._weight_update_bars and (
            len(buf) % self._weight_update_bars == 0
        ):
            self._refit_weights()

    def _refit_weights(self) -> None:
        """
        Softmax-normalise per-source mean-PnL into weights, floored at
        ``ML_WEIGHT_FLOOR`` and renormalised so Σ weights = n_sources × 1.0
        (each source's *baseline* weight is 1.0; the softmax redistributes
        that baseline according to historical edge).
        """
        means: Dict[str, float] = {}
        for src, buf in self._source_pnl_hist.items():
            if not buf:
                continue
            means[src] = sum(buf) / len(buf)
        if not means:
            return

        if self._weight_temperature <= 1e-9:
            # Hard-max: winner gets ~1, others floored.
            best = max(means, key=means.get)
            for src in means:
                self._source_weight[src] = (
                    len(means) * (1.0 - self._weight_floor)
                    if src == best
                    else self._weight_floor
                )
            return

        # Numerical-stability softmax
        scaled = {src: m / self._weight_temperature for src, m in means.items()}
        mx = max(scaled.values())
        exps = {src: math.exp(v - mx) for src, v in scaled.items()}
        z = sum(exps.values())
        if z <= 0.0:
            return
        n = len(exps)
        # Baseline weight = 1.0 per source; softmax redistributes that
        # total (n) among sources. Floor each at ML_WEIGHT_FLOOR, then
        # renormalise remaining mass above the floor.
        raw = {src: n * (e / z) for src, e in exps.items()}
        floored = {src: max(w, self._weight_floor) for src, w in raw.items()}
        total = sum(floored.values())
        if total > 0.0:
            self._source_weight = {src: (w * n / total) for src, w in floored.items()}

    def _weight_for(self, source: str) -> float:
        """Return the current dynamic weight for ``source`` (1.0 if unseen)."""
        return float(self._source_weight.get(source, 1.0))

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

            src_weight = self._weight_for(vote.source)
            if alignment > 0.0 and vc >= 0.20 and primary_gated:
                # Weighted boost contribution — accumulated, normalized later.
                agree_accum += self._agree_boost * vc * src_weight
                agreeing_count += 1
                logger.debug(
                    "SignalAggregator: %s agrees (s=%.3f c=%.2f w=%.2f)",
                    vote.source, vote.signal, vc, src_weight,
                )
            elif alignment < -self._contra_threshold and vc >= 0.20:
                contra_accum += self._contra_penalty * vc * src_weight
                contradicting_count += 1
                logger.debug(
                    "SignalAggregator: %s contradicts (s=%.3f c=%.2f w=%.2f)",
                    vote.source, vote.signal, vc, src_weight,
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
