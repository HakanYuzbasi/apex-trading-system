"""
execution/slippage_tracker.py — Realised vs modelled slippage monitor
======================================================================

Every fill produces two numbers: the slippage the cost model *predicted* and
the slippage the venue actually charged. When the live ratio drifts above a
configurable multiple, the liquidity assumptions in the sizer and the cost
model are stale — the right response is to shrink position sizes and flag the
symbol for investigation.

This module keeps a per-symbol rolling buffer of ``(realised_bps, model_bps)``
pairs and exposes:

- :meth:`SlippageTracker.record` — ingest one fill observation.
- :meth:`SlippageTracker.current_ratio` — most-recent mean realised / model
  ratio over the tracked window.
- :meth:`SlippageTracker.alert_active` — whether the ratio currently exceeds
  ``SLIPPAGE_ALERT_MULT``.
- :meth:`SlippageTracker.snapshot` — JSON-serialisable diagnostic dump for the
  metrics endpoint.

All thresholds come from :class:`ApexConfig`:

- ``SLIPPAGE_TRACK_WINDOW`` — rolling window length.
- ``SLIPPAGE_ALERT_MULT`` — WARNING fires when ``realised / model`` exceeds
  this multiple over the window.
- ``SLIPPAGE_MIN_MODEL_BPS`` — model-slippage floor applied when the cost
  model returns zero; without a floor the ratio is undefined.
"""
from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from config import ApexConfig

logger = logging.getLogger(__name__)


@dataclass
class SlippageObservation:
    """Single fill observation stored in the rolling window."""
    symbol: str
    realised_bps: float
    model_bps: float
    ratio: float


class SlippageTracker:
    """
    Rolling realised-vs-modelled slippage tracker.

    Attributes:
        window: Rolling window length in observations.
        alert_mult: Ratio threshold for WARNING emission.
        min_model_bps: Floor for the model slippage (prevents div-by-zero).
    """

    def __init__(
        self,
        window: Optional[int] = None,
        alert_mult: Optional[float] = None,
        min_model_bps: Optional[float] = None,
    ) -> None:
        self.window: int = int(
            window
            if window is not None
            else getattr(ApexConfig, "SLIPPAGE_TRACK_WINDOW", 50)
        )
        if self.window < 1:
            raise ValueError(
                f"SLIPPAGE_TRACK_WINDOW must be >= 1, got {self.window}"
            )
        self.alert_mult: float = float(
            alert_mult
            if alert_mult is not None
            else getattr(ApexConfig, "SLIPPAGE_ALERT_MULT", 2.0)
        )
        if self.alert_mult <= 0.0:
            raise ValueError(
                f"SLIPPAGE_ALERT_MULT must be > 0, got {self.alert_mult}"
            )
        self.min_model_bps: float = float(
            min_model_bps
            if min_model_bps is not None
            else getattr(ApexConfig, "SLIPPAGE_MIN_MODEL_BPS", 1.0)
        )
        self._lock = threading.Lock()
        self._buffer: Deque[SlippageObservation] = deque(maxlen=self.window)
        self._per_symbol: Dict[str, Deque[SlippageObservation]] = {}
        self._alert_active: bool = False

    # ── Ingest ───────────────────────────────────────────────────────────────

    def record(
        self,
        symbol: str,
        realised_bps: float,
        model_bps: float,
    ) -> SlippageObservation:
        """
        Record one fill observation.

        Args:
            symbol: Instrument the fill was for.
            realised_bps: Observed slippage in basis points (always ≥ 0 —
                pass ``abs(obs_bps)``).
            model_bps: Cost-model slippage projection in basis points.

        Returns:
            The stored :class:`SlippageObservation`.

        Raises:
            TypeError: If ``symbol`` is not a string.
            ValueError: If ``realised_bps`` is not finite.
        """
        if not isinstance(symbol, str):
            raise TypeError(f"symbol must be str, got {type(symbol).__name__}")
        sym = symbol.strip().upper()
        if not sym:
            raise ValueError("symbol must be non-empty")
        try:
            realised = float(realised_bps)
            model = float(model_bps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bps values must be numeric: {exc}") from exc
        if not (math.isfinite(realised) and math.isfinite(model)):
            raise ValueError("realised_bps and model_bps must be finite")

        denom = max(abs(model), self.min_model_bps)
        ratio = abs(realised) / denom
        obs = SlippageObservation(sym, abs(realised), abs(model), ratio)

        with self._lock:
            self._buffer.append(obs)
            sym_buf = self._per_symbol.get(sym)
            if sym_buf is None:
                sym_buf = deque(maxlen=self.window)
                self._per_symbol[sym] = sym_buf
            sym_buf.append(obs)
            ratio_now = self._mean_ratio_locked(self._buffer)
            if ratio_now >= self.alert_mult and not self._alert_active:
                self._alert_active = True
                logger.warning(
                    "Slippage alert: realised/model ratio %.2f ≥ %.2f over "
                    "last %d fills (most recent: %s realised=%.1fbps "
                    "model=%.1fbps)",
                    ratio_now, self.alert_mult, len(self._buffer),
                    sym, obs.realised_bps, obs.model_bps,
                )
            elif ratio_now < self.alert_mult and self._alert_active:
                self._alert_active = False
                logger.info(
                    "Slippage alert cleared: realised/model ratio %.2f < %.2f",
                    ratio_now, self.alert_mult,
                )
        return obs

    # ── Read-only accessors ──────────────────────────────────────────────────

    def current_ratio(self, symbol: Optional[str] = None) -> float:
        """
        Mean realised/model slippage ratio over the tracked window.

        Args:
            symbol: If provided, restrict to observations for this symbol.
                ``None`` returns the overall ratio.

        Returns:
            Mean ratio, or ``0.0`` when no observations exist.
        """
        with self._lock:
            if symbol is None:
                return self._mean_ratio_locked(self._buffer)
            buf = self._per_symbol.get(symbol.strip().upper())
            if not buf:
                return 0.0
            return self._mean_ratio_locked(buf)

    def alert_active(self) -> bool:
        """Return ``True`` while the overall ratio exceeds ``alert_mult``."""
        with self._lock:
            return self._alert_active

    def snapshot(self) -> Dict[str, object]:
        """
        JSON-serialisable dump suitable for /diagnostics endpoints.

        Returns:
            Dict with ``window``, ``alert_mult``, ``alert_active``, overall
            ``mean_ratio``, ``count``, and per-symbol ``mean_ratio`` breakdown.
        """
        with self._lock:
            per_symbol: Dict[str, Dict[str, float]] = {}
            for sym, buf in self._per_symbol.items():
                if not buf:
                    continue
                per_symbol[sym] = {
                    "count": float(len(buf)),
                    "mean_ratio": self._mean_ratio_locked(buf),
                    "mean_realised_bps": sum(o.realised_bps for o in buf) / len(buf),
                    "mean_model_bps": sum(o.model_bps for o in buf) / len(buf),
                }
            return {
                "window": self.window,
                "alert_mult": self.alert_mult,
                "alert_active": self._alert_active,
                "count": len(self._buffer),
                "mean_ratio": self._mean_ratio_locked(self._buffer),
                "per_symbol": per_symbol,
            }

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _mean_ratio_locked(buf: Deque[SlippageObservation]) -> float:
        if not buf:
            return 0.0
        return float(sum(o.ratio for o in buf) / len(buf))


_GLOBAL_TRACKER: Optional[SlippageTracker] = None
_GLOBAL_LOCK = threading.Lock()


def get_slippage_tracker() -> SlippageTracker:
    """Return the process-wide :class:`SlippageTracker` singleton."""
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_TRACKER is None:
                _GLOBAL_TRACKER = SlippageTracker()
    return _GLOBAL_TRACKER


def record_fill_slippage(
    symbol: str,
    realised_bps: float,
    model_bps: float,
) -> SlippageObservation:
    """
    Convenience wrapper for callers that don't need the tracker handle.

    Args:
        symbol: Traded instrument.
        realised_bps: Observed slippage in bps.
        model_bps: Cost-model projection in bps.

    Returns:
        The stored :class:`SlippageObservation`.
    """
    return get_slippage_tracker().record(symbol, realised_bps, model_bps)
