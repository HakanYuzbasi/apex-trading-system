"""
monitoring/correlation_regime_detector.py — Correlation Matrix Regime Detector

Detects when inter-asset correlations spike, signalling crisis conditions where
diversification breaks down. During correlation spikes:
  - All assets move together (portfolio acts as a single bet)
  - Normal signal edge disappears
  - Position sizes should be reduced proactively

Algorithm:
  1. Maintain a rolling return matrix for tracked symbols (default: last 30 bars).
  2. Compute the average pairwise Pearson correlation.
  3. Classify regime:
       normal    : avg_corr < WARNING_THRESHOLD (0.55)
       elevated  : WARNING_THRESHOLD ≤ avg_corr < CRISIS_THRESHOLD (0.75)
       crisis    : avg_corr ≥ CRISIS_THRESHOLD
  4. Apply position sizing multiplier:
       normal    : 1.0
       elevated  : 0.75
       crisis    : 0.50

Usage:
    detector = CorrelationRegimeDetector()
    detector.update_prices(symbol, price)      # call each cycle
    mult = detector.get_sizing_multiplier()    # apply to all new positions
    report = detector.get_report()             # dashboard/API

Config keys:
    CORR_REGIME_ENABLED                = True
    CORR_REGIME_LOOKBACK               = 30    # bars for correlation window
    CORR_REGIME_WARNING_THRESHOLD      = 0.55  # avg corr above this = elevated
    CORR_REGIME_CRISIS_THRESHOLD       = 0.75  # avg corr above this = crisis
    CORR_REGIME_MIN_SYMBOLS            = 4     # min symbols needed to compute
    CORR_REGIME_ELEVATED_MULT          = 0.75  # sizing mult in elevated regime
    CORR_REGIME_CRISIS_MULT            = 0.50  # sizing mult in crisis regime
    CORR_REGIME_UPDATE_INTERVAL        = 10    # update every N calls to update_prices
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "CORR_REGIME_ENABLED":            True,
    "CORR_REGIME_LOOKBACK":           30,
    "CORR_REGIME_WARNING_THRESHOLD":  0.55,
    "CORR_REGIME_CRISIS_THRESHOLD":   0.75,
    "CORR_REGIME_MIN_SYMBOLS":        4,
    "CORR_REGIME_ELEVATED_MULT":      0.75,
    "CORR_REGIME_CRISIS_MULT":        0.50,
    "CORR_REGIME_UPDATE_INTERVAL":    10,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


# ── Core maths ─────────────────────────────────────────────────────────────────

def compute_avg_pairwise_correlation(returns_matrix: np.ndarray) -> float:
    """
    Compute average pairwise Pearson correlation from a (n_symbols × n_bars) matrix.

    Args:
        returns_matrix: float array of shape (n_symbols, n_bars), log-returns.

    Returns:
        Average pairwise correlation in [-1, +1]. Returns 0.0 if not computable.
    """
    n_syms = returns_matrix.shape[0]
    if n_syms < 2:
        return 0.0

    # Compute correlation matrix
    try:
        corr_mat = np.corrcoef(returns_matrix)
    except Exception:
        return 0.0

    if corr_mat.shape != (n_syms, n_syms):
        return 0.0

    # Average of upper triangle (off-diagonal)
    n_pairs = 0
    total = 0.0
    for i in range(n_syms):
        for j in range(i + 1, n_syms):
            v = corr_mat[i, j]
            if not np.isnan(v) and not np.isinf(v):
                total += v
                n_pairs += 1

    if n_pairs == 0:
        return 0.0
    return float(total / n_pairs)


def classify_corr_regime(avg_corr: float, warning_thresh: float, crisis_thresh: float) -> str:
    """
    Classify correlation regime.
    Returns: 'normal' | 'elevated' | 'crisis'
    """
    if avg_corr >= crisis_thresh:
        return "crisis"
    elif avg_corr >= warning_thresh:
        return "elevated"
    return "normal"


# ── CorrelationRegimeDetector ─────────────────────────────────────────────────

@dataclass
class CorrRegimeState:
    """Snapshot of correlation regime state."""
    avg_pairwise_correlation: float = 0.0
    regime: str = "normal"             # normal / elevated / crisis
    sizing_multiplier: float = 1.0
    n_symbols: int = 0
    n_bars: int = 0
    top_correlated_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "avg_pairwise_correlation": round(self.avg_pairwise_correlation, 4),
            "regime": self.regime,
            "sizing_multiplier": round(self.sizing_multiplier, 4),
            "n_symbols": self.n_symbols,
            "n_bars": self.n_bars,
            "top_correlated_pairs": [
                {"sym1": s1, "sym2": s2, "corr": round(c, 4)}
                for s1, s2, c in self.top_correlated_pairs
            ],
            "timestamp": self.timestamp,
        }


class CorrelationRegimeDetector:
    """
    Inter-asset correlation regime detector.

    Tracks rolling log-returns for monitored symbols and computes average
    pairwise correlation. Classifies regime and provides a position sizing
    multiplier to reduce exposure during correlation spikes.
    """

    def __init__(self):
        lookback = int(_cfg("CORR_REGIME_LOOKBACK"))
        self._price_buffers: Dict[str, Deque[float]] = {}
        self._return_buffers: Dict[str, Deque[float]] = {}
        self._lookback = lookback
        self._call_count: int = 0
        self._last_state: Optional[CorrRegimeState] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_prices(self, symbol: str, price: float) -> None:
        """
        Feed a new price observation for a symbol.
        Computes log-return and appends to the rolling buffer.
        Triggers regime recomputation every UPDATE_INTERVAL calls.
        """
        if price <= 0:
            return

        if symbol not in self._price_buffers:
            self._price_buffers[symbol] = deque(maxlen=2)
            self._return_buffers[symbol] = deque(maxlen=self._lookback)

        prev_deque = self._price_buffers[symbol]
        if len(prev_deque) > 0:
            prev_price = float(prev_deque[-1])
            if prev_price > 0:
                log_ret = float(np.log(price / prev_price))
                self._return_buffers[symbol].append(log_ret)

        self._price_buffers[symbol].append(float(price))

        self._call_count += 1
        interval = int(_cfg("CORR_REGIME_UPDATE_INTERVAL"))
        if self._call_count % interval == 0:
            self._recompute()

    def get_sizing_multiplier(self) -> float:
        """Return position sizing multiplier based on correlation regime."""
        if not _cfg("CORR_REGIME_ENABLED"):
            return 1.0
        if self._last_state is None:
            self._recompute()
        return (self._last_state or CorrRegimeState()).sizing_multiplier

    def get_state(self) -> CorrRegimeState:
        """Return current correlation regime state."""
        if self._last_state is None:
            self._recompute()
        return self._last_state or CorrRegimeState()

    def get_report(self) -> Dict:
        """Return JSON-serialisable report."""
        return self.get_state().to_dict()

    def get_avg_correlation(self) -> float:
        """Return current average pairwise correlation."""
        return self.get_state().avg_pairwise_correlation

    # ── Internal ───────────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        if not _cfg("CORR_REGIME_ENABLED"):
            self._last_state = CorrRegimeState()
            return

        min_syms = int(_cfg("CORR_REGIME_MIN_SYMBOLS"))
        warn_thresh = float(_cfg("CORR_REGIME_WARNING_THRESHOLD"))
        crisis_thresh = float(_cfg("CORR_REGIME_CRISIS_THRESHOLD"))
        elev_mult = float(_cfg("CORR_REGIME_ELEVATED_MULT"))
        crisis_mult = float(_cfg("CORR_REGIME_CRISIS_MULT"))
        min_bars = 5  # minimum bars per symbol

        # Filter to symbols with sufficient history
        eligible = {
            sym: list(buf)
            for sym, buf in self._return_buffers.items()
            if len(buf) >= min_bars
        }

        if len(eligible) < min_syms:
            self._last_state = CorrRegimeState(n_symbols=len(eligible))
            return

        # Align to common length
        syms = list(eligible.keys())
        min_len = min(len(v) for v in eligible.values())
        mat = np.array([eligible[s][-min_len:] for s in syms])

        avg_corr = compute_avg_pairwise_correlation(mat)
        regime = classify_corr_regime(avg_corr, warn_thresh, crisis_thresh)

        # Sizing multiplier
        if regime == "crisis":
            mult = crisis_mult
        elif regime == "elevated":
            mult = elev_mult
        else:
            mult = 1.0

        # Top correlated pairs (for diagnostics)
        top_pairs = self._find_top_pairs(syms, mat, top_n=3)

        if regime != "normal" and (self._last_state is None or self._last_state.regime != regime):
            logger.warning(
                "CorrelationRegimeDetector: regime=%s avg_corr=%.3f n_syms=%d mult=%.2f",
                regime.upper(), avg_corr, len(syms), mult,
            )

        self._last_state = CorrRegimeState(
            avg_pairwise_correlation=avg_corr,
            regime=regime,
            sizing_multiplier=mult,
            n_symbols=len(syms),
            n_bars=min_len,
            top_correlated_pairs=top_pairs,
        )

    def _find_top_pairs(
        self, syms: List[str], mat: np.ndarray, top_n: int = 3
    ) -> List[Tuple[str, str, float]]:
        """Find the most correlated symbol pairs."""
        try:
            corr = np.corrcoef(mat)
            pairs = []
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    v = float(corr[i, j])
                    if not np.isnan(v):
                        pairs.append((syms[i], syms[j], v))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            return pairs[:top_n]
        except Exception:
            return []


# ── Module-level singleton ────────────────────────────────────────────────────

_corr_regime_detector: Optional[CorrelationRegimeDetector] = None


def get_corr_regime_detector() -> CorrelationRegimeDetector:
    global _corr_regime_detector
    if _corr_regime_detector is None:
        _corr_regime_detector = CorrelationRegimeDetector()
    return _corr_regime_detector
