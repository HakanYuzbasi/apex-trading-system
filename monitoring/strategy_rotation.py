"""
monitoring/strategy_rotation.py — Regime-Conditional Strategy Rotation

Tracks which signal components (ML, technical, sentiment, pairs, momentum)
have positive realized alpha *in each regime* over a rolling window, then
dynamically up-weights the working components and down-weights the laggards.

This turns the fixed blend weights (e.g. ML 55% / Tech 45%) into a live
adaptive weight vector that responds to recent evidence — a true
self-improving strategy-mix controller.

Algorithm
---------
1. On each trade close, record the signal components and P&L outcome.
2. Maintain a rolling window of (component_contribution, pnl) per regime.
3. Compute per-component "alpha score" = mean(pnl × component_weight).
4. Map alpha scores → blend weights via softmax with temperature T.
5. Apply significance gate: only deviate from equal weights if score
   differences are statistically significant (Wilson CI).
6. Output: blend_weights dict — used in execution_loop signal blending.

Components tracked:
    ml          — GodLevelSignalGenerator / AdvancedSignalGenerator output
    tech        — Technical indicators (RSI, MACD, Bollinger) composite
    sentiment   — News + social sentiment
    momentum    — Cross-sectional momentum score
    pairs       — Pairs-trading overlay signal
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_COMPONENTS = ("ml", "tech", "sentiment", "momentum", "pairs")
_DEFAULT_WEIGHT = 1.0 / len(_COMPONENTS)    # equal-weight prior
_TEMP = 2.0     # softmax temperature: higher = closer to equal
_MIN_RECORDS = 10   # min records per regime before deviating from prior
_MAX_RECORDS = 200  # rolling window cap


@dataclass
class ComponentRecord:
    """One closed-trade outcome annotated with signal component weights."""
    regime: str
    pnl_pct: float           # realised P&L fraction
    components: Dict[str, float]   # component → weight used at entry


@dataclass
class RegimeWeightSet:
    """Current blend weights for one regime."""
    regime: str
    weights: Dict[str, float]
    alpha_scores: Dict[str, float]
    record_count: int
    significant: bool
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class StrategyRotationController:
    """
    Adaptive signal blend controller.

    Wire in three ways:
        1. record_outcome(regime, pnl_pct, components) — call at trade close
        2. get_blend_weights(regime) — call before signal blending at entry
        3. get_report() — for dashboard / walk-forward dashboard
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        max_records: int = _MAX_RECORDS,
        min_records: int = _MIN_RECORDS,
        temperature: float = _TEMP,
        components: Tuple[str, ...] = _COMPONENTS,
        significance_alpha: float = 0.10,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self._max_records = max_records
        self._min_records = min_records
        self._temperature = temperature
        self._components = components
        self._sig_alpha = significance_alpha

        # regime → deque of ComponentRecord
        self._records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_records))

        # Cached weight sets (recomputed lazily after new records)
        self._weight_cache: Dict[str, RegimeWeightSet] = {}
        self._dirty_regimes: set = set()

        self._load_state()

    # ------------------------------------------------------------------
    # Public: record outcome
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        regime: str,
        pnl_pct: float,
        components: Dict[str, float],
    ) -> None:
        """
        Record a closed-trade outcome.

        Args:
            regime:     Market regime at entry (e.g. "bull", "bear", "volatile")
            pnl_pct:    Realised return as fraction (e.g. 0.02 = +2%)
            components: {component_name: weight_used_at_entry}
                        Missing components default to 0.0.
        """
        rec = ComponentRecord(regime=regime, pnl_pct=pnl_pct, components=dict(components))
        self._records[regime].append(rec)
        self._dirty_regimes.add(regime)

    # ------------------------------------------------------------------
    # Public: query blend weights
    # ------------------------------------------------------------------

    def get_blend_weights(self, regime: str) -> Dict[str, float]:
        """
        Return normalised blend weights for each component in this regime.

        Falls back to equal weights when insufficient data or when the
        deviation from equal is not statistically significant.

        Returns:
            Dict mapping component name → weight (sum = 1.0)
        """
        if regime in self._dirty_regimes:
            self._recompute(regime)
            self._dirty_regimes.discard(regime)

        ws = self._weight_cache.get(regime)
        if ws is None or not ws.significant:
            return {c: _DEFAULT_WEIGHT for c in self._components}
        return dict(ws.weights)

    def get_all_regimes(self) -> List[str]:
        return list(self._records.keys())

    def get_report(self) -> dict:
        """Full diagnostic dict for dashboard."""
        # Flush all dirty
        for r in list(self._dirty_regimes):
            self._recompute(r)
        self._dirty_regimes.clear()

        return {
            "regimes": {
                regime: asdict(ws)
                for regime, ws in self._weight_cache.items()
            },
            "record_counts": {r: len(d) for r, d in self._records.items()},
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------
    # Private: weight computation
    # ------------------------------------------------------------------

    def _recompute(self, regime: str) -> None:
        records = list(self._records[regime])
        n = len(records)

        if n < self._min_records:
            self._weight_cache[regime] = RegimeWeightSet(
                regime=regime,
                weights={c: _DEFAULT_WEIGHT for c in self._components},
                alpha_scores={c: 0.0 for c in self._components},
                record_count=n,
                significant=False,
            )
            return

        # Compute per-component alpha score: mean(pnl × component_weight)
        alpha: Dict[str, List[float]] = {c: [] for c in self._components}
        for rec in records:
            for comp in self._components:
                w = rec.components.get(comp, 0.0)
                alpha[comp].append(rec.pnl_pct * w)

        alpha_scores = {c: float(np.mean(alpha[c])) if alpha[c] else 0.0 for c in self._components}

        # Softmax with temperature → blend weights
        scores_arr = np.array([alpha_scores[c] for c in self._components], dtype=float)
        weights_arr = self._softmax(scores_arr / self._temperature)
        weights = {c: float(w) for c, w in zip(self._components, weights_arr)}

        # Significance gate: check if best vs worst component alpha difference is real
        best_c = max(alpha_scores, key=alpha_scores.get)
        worst_c = min(alpha_scores, key=alpha_scores.get)
        sig = self._is_deviation_significant(alpha[best_c], alpha[worst_c], n)

        if sig:
            logger.info(
                "StrategyRotation [%s]: %s (n=%d, best=%s α=%.4f, worst=%s α=%.4f)",
                regime, "SIGNIFICANT" if sig else "NOT_SIG", n,
                best_c, alpha_scores[best_c],
                worst_c, alpha_scores[worst_c],
            )

        self._weight_cache[regime] = RegimeWeightSet(
            regime=regime,
            weights=weights,
            alpha_scores=alpha_scores,
            record_count=n,
            significant=sig,
        )
        self._persist()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _is_deviation_significant(
        self, best_series: list, worst_series: list, n: int
    ) -> bool:
        """
        Test if best-component alpha is significantly greater than worst using
        a one-sided sign test (wins = best > worst per trade).
        """
        try:
            from monitoring.stat_significance import is_significant
            paired = list(zip(best_series, worst_series))
            if not paired:
                return False
            wins = sum(1 for b, w in paired if b > w)
            return is_significant(wins=wins, n=len(paired), alpha=self._sig_alpha)
        except ImportError:
            return n >= self._min_records * 2

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if not self._data_dir:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            path = self._data_dir / "strategy_rotation.json"
            payload = {
                "weight_cache": {r: asdict(ws) for r, ws in self._weight_cache.items()},
                "record_counts": {r: len(d) for r, d in self._records.items()},
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.debug("StrategyRotation persist error: %s", exc)

    def _load_state(self) -> None:
        if not self._data_dir:
            return
        path = Path(self._data_dir) / "strategy_rotation.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for regime, ws_dict in data.get("weight_cache", {}).items():
                self._weight_cache[regime] = RegimeWeightSet(**ws_dict)
            logger.info(
                "StrategyRotation: loaded weights for %d regimes",
                len(self._weight_cache),
            )
        except Exception as exc:
            logger.debug("StrategyRotation load error: %s", exc)
