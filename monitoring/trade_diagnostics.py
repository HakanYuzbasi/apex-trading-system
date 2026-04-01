"""
monitoring/trade_diagnostics.py

Tracks the full gate-decision context for every entry attempt.
Enables root-cause analysis: which gates block most entries, per symbol,
per regime, over any lookback window.

Usage in execution_loop::

    # init
    self._diagnostics = TradeDiagnosticsTracker(data_dir=ApexConfig.DATA_DIR)

    # before each blocking return
    self._diagnostics.record_decision(TradeDecisionRecord(
        symbol=symbol, ts=time.time(), regime=str(self._current_regime),
        signal=float(signal), confidence=float(confidence),
        action="blocked", block_gate="confidence_threshold",
        gates=[GateDecision("confidence_threshold", True, True,
                            float(confidence), float(effective_confidence_threshold))],
    ))

    # after successful entry
    self._diagnostics.record_decision(TradeDecisionRecord(..., action="entered"))

    # at trade close
    self._diagnostics.record_outcome(symbol, pnl_pct, hold_hours, exit_reason)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

_DEFAULT_MAX_RECORDS = 5000
_PERSIST_TAIL = 1000  # lines kept on disk


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class GateDecision:
    """Snapshot of a single gate evaluation."""
    gate: str           # e.g. "confidence_threshold", "drawdown_gate"
    fired: bool         # True when gate evaluated (not necessarily blocking)
    blocked: bool       # True when gate caused an early return
    value: float        # actual value (e.g. confidence=0.61)
    threshold: float    # gate threshold (e.g. 0.68)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeDecisionRecord:
    """Full context of one entry attempt."""
    symbol: str
    ts: float                   # unix timestamp
    regime: str
    signal: float
    confidence: float
    gates: List[GateDecision]
    action: str                 # "entered" | "blocked" | "skipped"
    block_gate: str = ""        # name of the first gate that blocked
    # Outcome — filled in when position closes
    pnl_pct: Optional[float] = None
    hold_hours: Optional[float] = None
    exit_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "ts": self.ts,
            "regime": self.regime,
            "signal": round(self.signal, 6),
            "confidence": round(self.confidence, 6),
            "action": self.action,
            "block_gate": self.block_gate,
            "pnl_pct": self.pnl_pct,
            "hold_hours": self.hold_hours,
            "exit_reason": self.exit_reason,
            "gates": [g.to_dict() for g in self.gates],
        }


# ── Main tracker ──────────────────────────────────────────────────────────────

class TradeDiagnosticsTracker:
    """Accumulates trade decision records and produces attribution reports.

    Parameters
    ----------
    data_dir : Path | None
        If given, records are persisted to ``trade_diagnostics.jsonl``.
    max_records : int
        Maximum records kept in memory.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        max_records: int = _DEFAULT_MAX_RECORDS,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self._max_records = max_records
        self._records: List[TradeDecisionRecord] = []
        # symbol → latest open "entered" record (awaiting outcome)
        self._pending: Dict[str, TradeDecisionRecord] = {}
        if self._data_dir:
            self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_decision(self, record: TradeDecisionRecord) -> None:
        """Append a decision record (entered or blocked)."""
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]
        if record.action == "entered":
            self._pending[record.symbol] = record
        self._persist()

    def record_outcome(
        self,
        symbol: str,
        pnl_pct: float,
        hold_hours: float,
        exit_reason: str,
    ) -> None:
        """Attach trade outcome to the most recent 'entered' record for symbol."""
        rec = self._pending.pop(symbol, None)
        if rec is not None:
            rec.pnl_pct = round(float(pnl_pct), 6)
            rec.hold_hours = round(float(hold_hours), 3)
            rec.exit_reason = str(exit_reason)
            self._persist()

    # ── Reports ───────────────────────────────────────────────────────────────

    def get_gate_attribution(
        self,
        symbol: Optional[str] = None,
        lookback_days: int = 7,
    ) -> Dict[str, dict]:
        """Per-gate block counts over the lookback window.

        Returns dict sorted by block count descending.
        """
        recs = self._filter(symbol=symbol, lookback_days=lookback_days)
        total = len(recs)
        by_gate: Dict[str, Dict[str, int]] = defaultdict(lambda: {"fires": 0, "blocks": 0})

        for r in recs:
            for g in r.gates:
                if g.fired:
                    by_gate[g.gate]["fires"] += 1
                if g.blocked:
                    by_gate[g.gate]["blocks"] += 1

        result: Dict[str, dict] = {}
        for gate, counts in by_gate.items():
            result[gate] = {
                "fires": counts["fires"],
                "blocks": counts["blocks"],
                "block_rate": round(counts["blocks"] / total, 4) if total else 0.0,
                "total_decisions": total,
            }
        return dict(sorted(result.items(), key=lambda x: x[1]["blocks"], reverse=True))

    def get_blocked_analysis(
        self,
        symbol: Optional[str] = None,
        lookback_days: int = 7,
    ) -> dict:
        """Summary of blocked entries: count, rate, and first-gate breakdown."""
        all_recs = self._filter(symbol=symbol, lookback_days=lookback_days)
        blocked = [r for r in all_recs if r.action == "blocked"]

        by_gate: Dict[str, int] = defaultdict(int)
        for r in blocked:
            by_gate[r.block_gate or "unknown"] += 1

        total = len(all_recs)
        return {
            "total_decisions": total,
            "total_blocked": len(blocked),
            "block_rate": round(len(blocked) / total, 4) if total else 0.0,
            "by_first_gate": dict(
                sorted(by_gate.items(), key=lambda x: x[1], reverse=True)
            ),
        }

    def get_symbol_report(
        self, symbol: str, lookback_days: int = 14
    ) -> dict:
        """Per-symbol diagnosis: entry rate, win rate, and top blocking gates."""
        recs = self._filter(symbol=symbol, lookback_days=lookback_days)
        entered = [r for r in recs if r.action == "entered"]
        blocked = [r for r in recs if r.action == "blocked"]
        with_outcome = [r for r in entered if r.pnl_pct is not None]

        wins = sum(1 for r in with_outcome if (r.pnl_pct or 0.0) > 0)
        win_rate = wins / len(with_outcome) if with_outcome else None
        avg_pnl = (
            sum(r.pnl_pct or 0.0 for r in with_outcome) / len(with_outcome)
            if with_outcome else None
        )

        by_gate: Dict[str, int] = defaultdict(int)
        for r in blocked:
            by_gate[r.block_gate or "unknown"] += 1

        return {
            "symbol": symbol,
            "lookback_days": lookback_days,
            "total_decisions": len(recs),
            "entered": len(entered),
            "blocked": len(blocked),
            "block_rate": round(len(blocked) / len(recs), 4) if recs else 0.0,
            "completed_trades": len(with_outcome),
            "win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_pnl_pct": round(avg_pnl, 6) if avg_pnl is not None else None,
            "top_blocking_gates": dict(
                sorted(by_gate.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
        }

    def get_report(self, lookback_days: int = 7) -> dict:
        """Aggregate diagnostic report across all symbols."""
        recs = self._filter(lookback_days=lookback_days)
        entered = [r for r in recs if r.action == "entered"]
        with_outcome = [r for r in entered if r.pnl_pct is not None]
        wins = sum(1 for r in with_outcome if (r.pnl_pct or 0.0) > 0)

        # Per-symbol block rates (top 10 most blocked)
        by_sym: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "blocked": 0})
        for r in recs:
            by_sym[r.symbol]["total"] += 1
            if r.action == "blocked":
                by_sym[r.symbol]["blocked"] += 1

        most_blocked = sorted(
            [
                {
                    "symbol": sym,
                    "block_rate": round(v["blocked"] / v["total"], 4) if v["total"] else 0.0,
                    "blocked": v["blocked"],
                    "total": v["total"],
                }
                for sym, v in by_sym.items()
                if v["total"] >= 3
            ],
            key=lambda x: x["block_rate"],
            reverse=True,
        )[:10]

        return {
            "lookback_days": lookback_days,
            "total_records": len(recs),
            "entered": len(entered),
            "completed_trades": len(with_outcome),
            "overall_win_rate": round(wins / len(with_outcome), 4) if with_outcome else None,
            "gate_attribution": self.get_gate_attribution(lookback_days=lookback_days),
            "blocked_analysis": self.get_blocked_analysis(lookback_days=lookback_days),
            "most_blocked_symbols": most_blocked,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _filter(
        self,
        symbol: Optional[str] = None,
        lookback_days: int = 7,
    ) -> List[TradeDecisionRecord]:
        cutoff = time.time() - lookback_days * 86400.0
        recs = [r for r in self._records if r.ts >= cutoff]
        if symbol:
            recs = [r for r in recs if r.symbol == symbol]
        return recs

    # ── Persistence ───────────────────────────────────────────────────────────

    def _path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / "trade_diagnostics.jsonl"

    def _persist(self) -> None:
        if self._data_dir is None:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            tail = self._records[-_PERSIST_TAIL:]
            lines = [json.dumps(r.to_dict()) for r in tail]
            tmp = self._path().with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tmp.replace(self._path())
        except Exception:
            pass

    def _load(self) -> None:
        try:
            p = self._path()
            if not p.exists():
                return
            for raw in p.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                    gates = [
                        GateDecision(
                            gate=g.get("gate", ""),
                            fired=bool(g.get("fired", False)),
                            blocked=bool(g.get("blocked", False)),
                            value=float(g.get("value", 0.0)),
                            threshold=float(g.get("threshold", 0.0)),
                            note=str(g.get("note", "")),
                        )
                        for g in d.get("gates", [])
                    ]
                    rec = TradeDecisionRecord(
                        symbol=d["symbol"],
                        ts=float(d["ts"]),
                        regime=d.get("regime", "unknown"),
                        signal=float(d.get("signal", 0.0)),
                        confidence=float(d.get("confidence", 0.0)),
                        gates=gates,
                        action=d.get("action", "blocked"),
                        block_gate=d.get("block_gate", ""),
                        pnl_pct=d.get("pnl_pct"),
                        hold_hours=d.get("hold_hours"),
                        exit_reason=d.get("exit_reason"),
                    )
                    self._records.append(rec)
                    if rec.action == "entered" and rec.pnl_pct is None:
                        self._pending[rec.symbol] = rec
                except Exception:
                    pass
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
        except Exception:
            pass
