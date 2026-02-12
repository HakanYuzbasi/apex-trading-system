"""
risk/performance_governor.py

Adaptive performance guardrails for live trading.
Uses realized equity behavior to dynamically tighten entry quality
and reduce risk when Sharpe/Sortino degrade or drawdown rises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GovernorTier(str, Enum):
    """Risk appetite tiers based on realized portfolio performance."""

    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class GovernorSnapshot:
    """Current controls produced by PerformanceGovernor."""

    tier: GovernorTier
    sharpe: float
    sortino: float
    drawdown: float
    sample_count: int
    size_multiplier: float
    signal_threshold_boost: float
    confidence_boost: float
    halt_new_entries: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Serialize snapshot for dashboards/apis."""
        return {
            "tier": self.tier.value,
            "sharpe": float(self.sharpe),
            "sortino": float(self.sortino),
            "drawdown": float(self.drawdown),
            "sample_count": int(self.sample_count),
            "size_multiplier": float(self.size_multiplier),
            "signal_threshold_boost": float(self.signal_threshold_boost),
            "confidence_boost": float(self.confidence_boost),
            "halt_new_entries": bool(self.halt_new_entries),
            "reasons": list(self.reasons),
        }


class PerformanceGovernor:
    """
    Live performance governor with hysteresis.

    It samples equity at a coarse interval, computes rolling Sharpe/Sortino/max-DD,
    and emits controls used by execution logic.
    """

    _SEVERITY = {
        GovernorTier.GREEN: 0,
        GovernorTier.YELLOW: 1,
        GovernorTier.ORANGE: 2,
        GovernorTier.RED: 3,
    }
    _TIER_ORDER = {
        0: GovernorTier.GREEN,
        1: GovernorTier.YELLOW,
        2: GovernorTier.ORANGE,
        3: GovernorTier.RED,
    }
    _CONTROLS: Dict[GovernorTier, Tuple[float, float, float, bool]] = {
        GovernorTier.GREEN: (1.00, 0.00, 0.00, False),
        GovernorTier.YELLOW: (0.75, 0.03, 0.05, False),
        GovernorTier.ORANGE: (0.50, 0.07, 0.10, False),
        GovernorTier.RED: (0.25, 0.12, 0.20, True),
    }

    def __init__(
        self,
        target_sharpe: float = 1.5,
        target_sortino: float = 2.0,
        max_drawdown: float = 0.08,
        sample_interval_minutes: int = 15,
        min_samples: int = 30,
        lookback_points: int = 200,
        recovery_points: int = 3,
        risk_free_rate: float = 0.02,
        points_per_year: int = 3276,
    ):
        self.target_sharpe = max(0.1, float(target_sharpe))
        self.target_sortino = max(0.1, float(target_sortino))
        self.max_drawdown = max(0.01, float(max_drawdown))
        self.sample_interval = timedelta(minutes=max(1, int(sample_interval_minutes)))
        self.min_samples = max(5, int(min_samples))
        self.lookback_points = max(self.min_samples + 1, int(lookback_points))
        self.recovery_points = max(1, int(recovery_points))
        self.risk_free_rate = float(risk_free_rate)
        self.points_per_year = max(100, int(points_per_year))

        self._samples: List[Tuple[datetime, float]] = []
        self._last_sample_time: Optional[datetime] = None
        self._recovery_streak = 0
        self._tier = GovernorTier.GREEN
        self._snapshot = self._build_snapshot(
            tier=GovernorTier.GREEN,
            sharpe=0.0,
            sortino=0.0,
            drawdown=0.0,
            sample_count=0,
            reasons=["Warming up: insufficient performance samples"],
        )

    def get_snapshot(self) -> GovernorSnapshot:
        """Get current controls without updating state."""
        return self._snapshot

    def update(self, equity_value: float, timestamp: Optional[datetime] = None) -> GovernorSnapshot:
        """Update governor state using latest portfolio equity."""
        if equity_value <= 0:
            return self._snapshot

        ts = timestamp or datetime.now()
        if self._last_sample_time and (ts - self._last_sample_time) < self.sample_interval:
            return self._snapshot

        self._samples.append((ts, float(equity_value)))
        self._last_sample_time = ts

        if len(self._samples) > self.lookback_points:
            self._samples = self._samples[-self.lookback_points :]

        if len(self._samples) < self.min_samples:
            self._snapshot = self._build_snapshot(
                tier=self._tier,
                sharpe=0.0,
                sortino=0.0,
                drawdown=0.0,
                sample_count=len(self._samples),
                reasons=["Warming up: insufficient performance samples"],
            )
            return self._snapshot

        sharpe, sortino, drawdown = self._compute_metrics()
        suggested_tier, reasons = self._determine_tier(sharpe, sortino, drawdown)
        self._apply_hysteresis(suggested_tier)

        self._snapshot = self._build_snapshot(
            tier=self._tier,
            sharpe=sharpe,
            sortino=sortino,
            drawdown=drawdown,
            sample_count=len(self._samples),
            reasons=reasons,
        )
        return self._snapshot

    def _compute_metrics(self) -> Tuple[float, float, float]:
        values = np.array([v for _, v in self._samples], dtype=float)
        if values.size < 3:
            return 0.0, 0.0, 0.0

        returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
        returns = returns[np.isfinite(returns)]
        if returns.size < 2:
            return 0.0, 0.0, 0.0

        rf_per_point = self.risk_free_rate / self.points_per_year
        excess = returns - rf_per_point

        vol = float(np.std(excess))
        sharpe = float(np.mean(excess) / vol * np.sqrt(self.points_per_year)) if vol > 1e-12 else 0.0

        downside = excess[excess < 0.0]
        downside_std = float(np.std(downside)) if downside.size > 0 else 0.0
        sortino = (
            float(np.mean(excess) / downside_std * np.sqrt(self.points_per_year))
            if downside_std > 1e-12
            else 0.0
        )

        running_peak = np.maximum.accumulate(values)
        drawdowns = (values - running_peak) / np.maximum(running_peak, 1e-9)
        max_drawdown = abs(float(np.min(drawdowns)))

        return sharpe, sortino, max_drawdown

    def _determine_tier(self, sharpe: float, sortino: float, drawdown: float) -> Tuple[GovernorTier, List[str]]:
        reasons: List[str] = []

        sharpe_ratio = sharpe / self.target_sharpe
        sortino_ratio = sortino / self.target_sortino

        if drawdown >= self.max_drawdown or sharpe < -0.25 or sortino < -0.10:
            reasons.append(f"drawdown={drawdown:.2%}, sharpe={sharpe:.2f}, sortino={sortino:.2f}")
            return GovernorTier.RED, reasons

        if (
            drawdown >= self.max_drawdown * 0.80
            or sharpe_ratio < 0.40
            or sortino_ratio < 0.40
        ):
            reasons.append(f"performance stressed: sharpe={sharpe:.2f}, sortino={sortino:.2f}, dd={drawdown:.2%}")
            return GovernorTier.ORANGE, reasons

        if (
            drawdown >= self.max_drawdown * 0.60
            or sharpe_ratio < 0.75
            or sortino_ratio < 0.75
        ):
            reasons.append(f"performance below target: sharpe={sharpe:.2f}, sortino={sortino:.2f}, dd={drawdown:.2%}")
            return GovernorTier.YELLOW, reasons

        reasons.append(f"performance healthy: sharpe={sharpe:.2f}, sortino={sortino:.2f}, dd={drawdown:.2%}")
        return GovernorTier.GREEN, reasons

    def _apply_hysteresis(self, suggested_tier: GovernorTier) -> None:
        current = self._SEVERITY[self._tier]
        suggested = self._SEVERITY[suggested_tier]

        if suggested > current:
            # Deterioration: act immediately.
            self._tier = suggested_tier
            self._recovery_streak = 0
            return

        if suggested == current:
            self._recovery_streak = 0
            return

        # Improvement: require consecutive recoveries to avoid flapping.
        self._recovery_streak += 1
        if self._recovery_streak >= self.recovery_points:
            next_tier = self._TIER_ORDER[current - 1]
            # Never jump below suggested tier in one step.
            if self._SEVERITY[next_tier] < suggested:
                next_tier = suggested_tier
            self._tier = next_tier
            self._recovery_streak = 0

    def _build_snapshot(
        self,
        tier: GovernorTier,
        sharpe: float,
        sortino: float,
        drawdown: float,
        sample_count: int,
        reasons: List[str],
    ) -> GovernorSnapshot:
        size_mult, threshold_boost, confidence_boost, halt_entries = self._CONTROLS[tier]
        return GovernorSnapshot(
            tier=tier,
            sharpe=sharpe,
            sortino=sortino,
            drawdown=drawdown,
            sample_count=sample_count,
            size_multiplier=size_mult,
            signal_threshold_boost=threshold_boost,
            confidence_boost=confidence_boost,
            halt_new_entries=halt_entries,
            reasons=reasons,
        )
