"""
reconciliation/equity_reconciler.py

Equity reconciliation between broker-reported equity and locally modeled equity.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


@dataclass
class EquityReconciliationSnapshot:
    timestamp: datetime
    broker_equity: float
    modeled_equity: float
    gap_dollars: float
    gap_pct: float
    max_gap_dollars: float
    max_gap_pct: float
    breached: bool
    block_entries: bool
    reason: str
    breach_streak: int
    healthy_streak: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class EquityReconciler:
    """
    Reconcile broker equity with modeled equity and drive an entry-block latch.

    A breach is triggered when either absolute gap or relative gap exceeds limits.
    Entry blocking can be configured to fail-closed when modeled data is unavailable.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_gap_dollars: float = 20_000.0,
        max_gap_pct: float = 0.015,
        breach_confirmations: int = 2,
        heal_confirmations: int = 3,
        fail_closed_on_unavailable: bool = True,
    ):
        self.enabled = bool(enabled)
        self.max_gap_dollars = max(0.0, float(max_gap_dollars))
        self.max_gap_pct = max(0.0, float(max_gap_pct))
        self.breach_confirmations = max(1, int(breach_confirmations))
        self.heal_confirmations = max(1, int(heal_confirmations))
        self.fail_closed_on_unavailable = bool(fail_closed_on_unavailable)

        self.breach_streak: int = 0
        self.healthy_streak: int = 0
        self.block_entries_latch: bool = False
        self.last_snapshot: Optional[EquityReconciliationSnapshot] = None

    def evaluate(
        self,
        *,
        broker_equity: Optional[float],
        modeled_equity: Optional[float],
        timestamp: Optional[datetime] = None,
    ) -> EquityReconciliationSnapshot:
        now = timestamp or datetime.now()

        broker = self._safe_float(broker_equity)
        modeled = self._safe_float(modeled_equity)

        if not self.enabled:
            snapshot = EquityReconciliationSnapshot(
                timestamp=now,
                broker_equity=max(0.0, broker),
                modeled_equity=max(0.0, modeled),
                gap_dollars=0.0,
                gap_pct=0.0,
                max_gap_dollars=self.max_gap_dollars,
                max_gap_pct=self.max_gap_pct,
                breached=False,
                block_entries=False,
                reason="disabled",
                breach_streak=0,
                healthy_streak=0,
            )
            self.last_snapshot = snapshot
            return snapshot

        unavailable = broker <= 0 or modeled <= 0
        if unavailable:
            breached = self.fail_closed_on_unavailable
            reason = "data_unavailable" if breached else "data_unavailable_ignored"
            gap_dollars = abs(broker - modeled)
            gap_pct = 1.0 if broker > 0 else 0.0
        else:
            gap_dollars = abs(broker - modeled)
            gap_pct = gap_dollars / max(abs(broker), 1e-9)
            breached = (gap_dollars > self.max_gap_dollars) or (gap_pct > self.max_gap_pct)
            reason = "gap_threshold_exceeded" if breached else "ok"

        if breached:
            self.breach_streak += 1
            self.healthy_streak = 0
        else:
            self.breach_streak = 0
            self.healthy_streak += 1

        if breached and self.breach_streak >= self.breach_confirmations:
            self.block_entries_latch = True
        elif (
            self.block_entries_latch
            and not breached
            and self.healthy_streak >= self.heal_confirmations
        ):
            self.block_entries_latch = False

        snapshot = EquityReconciliationSnapshot(
            timestamp=now,
            broker_equity=max(0.0, broker),
            modeled_equity=max(0.0, modeled),
            gap_dollars=float(gap_dollars),
            gap_pct=float(gap_pct),
            max_gap_dollars=self.max_gap_dollars,
            max_gap_pct=self.max_gap_pct,
            breached=bool(breached),
            block_entries=bool(self.block_entries_latch),
            reason=reason,
            breach_streak=int(self.breach_streak),
            healthy_streak=int(self.healthy_streak),
        )
        self.last_snapshot = snapshot
        return snapshot

    @staticmethod
    def _safe_float(value: Optional[float]) -> float:
        try:
            val = float(value)
        except Exception:
            return 0.0
        if val != val:  # NaN
            return 0.0
        return val
