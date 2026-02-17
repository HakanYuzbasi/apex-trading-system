"""
risk/equity_outlier_guard.py

Rejects one-off equity glitches before they contaminate live risk metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class EquityValidationDecision:
    accepted: bool
    raw_value: float
    filtered_value: float
    reason: str
    deviation_pct: float
    suspect_count: int


class EquityOutlierGuard:
    """
    Filters implausible point-to-point equity jumps.

    Large jumps must repeat for N consecutive samples to be accepted.
    This protects Sharpe/DD from single feed glitches while still allowing
    genuine regime shifts to pass after confirmation.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_step_move_pct: float = 0.25,
        confirmations_required: int = 3,
        suspect_match_tolerance_pct: float = 0.02,
    ):
        self.enabled = bool(enabled)
        self.max_step_move_pct = max(0.01, float(max_step_move_pct))
        self.confirmations_required = max(1, int(confirmations_required))
        self.suspect_match_tolerance_pct = max(0.001, float(suspect_match_tolerance_pct))

        self.last_accepted_equity: Optional[float] = None
        self.last_accepted_at: Optional[datetime] = None
        self.suspect_value: Optional[float] = None
        self.suspect_count: int = 0
        self.rejections_total: int = 0

    def seed(self, equity_value: float, observed_at: Optional[datetime] = None) -> bool:
        """Seed baseline accepted equity when available."""
        try:
            value = float(equity_value)
        except Exception:
            return False
        if value <= 0:
            return False
        self.last_accepted_equity = value
        self.last_accepted_at = observed_at or datetime.now()
        self.suspect_value = None
        self.suspect_count = 0
        return True

    def evaluate(
        self,
        raw_equity_value: float,
        observed_at: Optional[datetime] = None,
    ) -> EquityValidationDecision:
        """Validate a raw equity sample and return safe value for risk metrics."""
        now = observed_at or datetime.now()
        try:
            raw_value = float(raw_equity_value)
        except Exception:
            raw_value = 0.0

        if raw_value <= 0:
            self.rejections_total += 1
            return EquityValidationDecision(
                accepted=False,
                raw_value=raw_value,
                filtered_value=float(self.last_accepted_equity or 0.0),
                reason="non_positive",
                deviation_pct=0.0,
                suspect_count=self.suspect_count,
            )

        if not self.enabled:
            self.seed(raw_value, observed_at=now)
            return EquityValidationDecision(
                accepted=True,
                raw_value=raw_value,
                filtered_value=raw_value,
                reason="guard_disabled",
                deviation_pct=0.0,
                suspect_count=0,
            )

        if self.last_accepted_equity is None:
            self.seed(raw_value, observed_at=now)
            return EquityValidationDecision(
                accepted=True,
                raw_value=raw_value,
                filtered_value=raw_value,
                reason="seeded",
                deviation_pct=0.0,
                suspect_count=0,
            )

        baseline = max(1e-9, float(self.last_accepted_equity))
        deviation_pct = abs(raw_value - baseline) / baseline
        if deviation_pct <= self.max_step_move_pct:
            self.seed(raw_value, observed_at=now)
            return EquityValidationDecision(
                accepted=True,
                raw_value=raw_value,
                filtered_value=raw_value,
                reason="within_threshold",
                deviation_pct=deviation_pct,
                suspect_count=0,
            )

        if self._matches_suspect(raw_value):
            self.suspect_count += 1
        else:
            self.suspect_value = raw_value
            self.suspect_count = 1

        if self.suspect_count >= self.confirmations_required:
            self.seed(raw_value, observed_at=now)
            return EquityValidationDecision(
                accepted=True,
                raw_value=raw_value,
                filtered_value=raw_value,
                reason="confirmed_large_move",
                deviation_pct=deviation_pct,
                suspect_count=self.suspect_count,
            )

        self.rejections_total += 1
        return EquityValidationDecision(
            accepted=False,
            raw_value=raw_value,
            filtered_value=float(self.last_accepted_equity),
            reason="outlier_rejected",
            deviation_pct=deviation_pct,
            suspect_count=self.suspect_count,
        )

    def _matches_suspect(self, raw_value: float) -> bool:
        if self.suspect_value is None:
            return False
        denom = max(1e-9, abs(float(self.suspect_value)))
        return (abs(raw_value - float(self.suspect_value)) / denom) <= self.suspect_match_tolerance_pct
