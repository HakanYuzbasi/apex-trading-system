"""
monitoring/strategy_health_monitor.py — Rolling Strategy Health Monitor

Computes a rolling 30-day Sharpe ratio from closed-trade P&L records.
When Sharpe degrades below a configurable threshold, the system enters
"paper-only" mode: new entries are simulated but no real orders are placed.

Integration:
    - Call record_trade(pnl_pct, timestamp) at every trade close.
    - Call check_health() each cycle (cheap — runs in-memory).
    - If check_health().paper_only is True, skip real order submission.

Config keys (ApexConfig defaults):
    STRATEGY_HEALTH_ENABLED          = True
    STRATEGY_HEALTH_LOOKBACK_DAYS    = 30
    STRATEGY_HEALTH_MIN_TRADES       = 5      # need ≥5 trades to evaluate
    STRATEGY_HEALTH_PAPER_THRESHOLD  = 0.0    # Sharpe below this → paper-only
    STRATEGY_HEALTH_RECOVER_THRESHOLD= 0.30   # Sharpe above this to exit paper mode
    STRATEGY_HEALTH_PERSIST_PATH     = "data/strategy_health_state.json"
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEF = {
    "STRATEGY_HEALTH_ENABLED":           True,
    "STRATEGY_HEALTH_LOOKBACK_DAYS":     30,
    "STRATEGY_HEALTH_MIN_TRADES":        5,
    "STRATEGY_HEALTH_PAPER_THRESHOLD":   0.0,
    "STRATEGY_HEALTH_RECOVER_THRESHOLD": 0.30,
    "STRATEGY_HEALTH_PERSIST_PATH":      "data/strategy_health_state.json",
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


@dataclass
class HealthState:
    rolling_sharpe: float        # annualised Sharpe over lookback window
    trade_count: int             # trades in the window
    paper_only: bool             # True → skip real orders
    reason: str                  # human-readable explanation
    last_updated: str            # ISO timestamp


@dataclass
class _TradeRecord:
    pnl_pct: float
    ts: str   # ISO timestamp


class StrategyHealthMonitor:
    """
    Tracks rolling trade P&L and gates live trading when Sharpe is degraded.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._persist_path = persist_path or str(_cfg("STRATEGY_HEALTH_PERSIST_PATH"))
        self._trades: List[_TradeRecord] = []
        self._paper_only: bool = False
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade(self, pnl_pct: float, timestamp: Optional[datetime] = None) -> None:
        """Record a closed-trade P&L (as a percentage, e.g. 0.03 = 3%)."""
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        self._trades.append(_TradeRecord(pnl_pct=float(pnl_pct), ts=ts))
        self._evict_old_trades()
        self._save()

    def check_health(self) -> HealthState:
        """
        Evaluate current health state and update paper_only flag.
        Returns a HealthState describing the current situation.
        """
        if not _cfg("STRATEGY_HEALTH_ENABLED"):
            return HealthState(
                rolling_sharpe=0.0, trade_count=0,
                paper_only=False, reason="disabled",
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

        self._evict_old_trades()
        min_trades = int(_cfg("STRATEGY_HEALTH_MIN_TRADES"))

        if len(self._trades) < min_trades:
            # Not enough data — stay live (benefit of the doubt)
            self._paper_only = False
            return HealthState(
                rolling_sharpe=0.0,
                trade_count=len(self._trades),
                paper_only=False,
                reason=f"insufficient_data ({len(self._trades)}/{min_trades} trades)",
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

        pnls = [t.pnl_pct for t in self._trades]
        sharpe = self._rolling_sharpe(pnls)
        paper_thresh  = float(_cfg("STRATEGY_HEALTH_PAPER_THRESHOLD"))
        recover_thresh = float(_cfg("STRATEGY_HEALTH_RECOVER_THRESHOLD"))

        if self._paper_only:
            # Currently in paper-only: recover only once Sharpe crosses recover_threshold
            if sharpe >= recover_thresh:
                self._paper_only = False
                reason = f"recovered (sharpe={sharpe:.2f} >= {recover_thresh:.2f})"
                logger.info(
                    "✅ StrategyHealth: LIVE mode restored — rolling Sharpe=%.2f (recover_thresh=%.2f)",
                    sharpe, recover_thresh,
                )
            else:
                reason = f"paper_only (sharpe={sharpe:.2f} < {recover_thresh:.2f} recover threshold)"
        else:
            if sharpe < paper_thresh:
                self._paper_only = True
                reason = f"degraded (sharpe={sharpe:.2f} < {paper_thresh:.2f})"
                logger.warning(
                    "⚠️  StrategyHealth: PAPER-ONLY mode — rolling Sharpe=%.2f (threshold=%.2f)",
                    sharpe, paper_thresh,
                )
            else:
                reason = f"healthy (sharpe={sharpe:.2f})"

        self._save()
        return HealthState(
            rolling_sharpe=round(sharpe, 4),
            trade_count=len(self._trades),
            paper_only=self._paper_only,
            reason=reason,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def paper_only(self) -> bool:
        return self._paper_only

    def get_state_dict(self) -> dict:
        return asdict(self.check_health())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_old_trades(self) -> None:
        lookback = int(_cfg("STRATEGY_HEALTH_LOOKBACK_DAYS"))
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback)
        self._trades = [
            t for t in self._trades
            if datetime.fromisoformat(t.ts).replace(tzinfo=timezone.utc) >= cutoff
        ]

    @staticmethod
    def _rolling_sharpe(pnls: List[float]) -> float:
        """Annualised Sharpe from a list of per-trade P&L percentages."""
        n = len(pnls)
        if n < 2:
            return 0.0
        mean = sum(pnls) / n
        var = sum((x - mean) ** 2 for x in pnls) / (n - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-9:
            # Near-zero variance: sign of mean determines direction
            if mean > 1e-9:
                return 10.0   # perfectly consistent wins → very high Sharpe
            elif mean < -1e-9:
                return -10.0  # perfectly consistent losses → very low Sharpe
            return 0.0
        # Annualise assuming ~252 trades/year (daily frequency approximation)
        return (mean / std) * math.sqrt(252)

    def _load(self) -> None:
        try:
            path = Path(self._persist_path)
            if path.exists():
                with open(path) as f:
                    state = json.load(f)
                self._paper_only = bool(state.get("paper_only", False))
                self._trades = [
                    _TradeRecord(**r) for r in state.get("trades", [])
                ]
                self._evict_old_trades()
        except Exception as e:
            logger.debug("StrategyHealth: could not load state — %s", e)

    def _save(self) -> None:
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "paper_only": self._paper_only,
                "trades": [{"pnl_pct": t.pnl_pct, "ts": t.ts} for t in self._trades],
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.debug("StrategyHealth: could not save state — %s", e)
