"""
monitoring/trade_audit.py — Audit-Grade Trade Lifecycle Logger

Appends one JSONL record per order event (entry, exit, partial TP).
Richer than execution_latency.jsonl — includes signal, confidence, exit reason,
pnl_pct, regime, and pretrade outcome in a single structured record.

One file per UTC date: data/users/admin/audit/trade_audit_YYYYMMDD.jsonl
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TradeAuditLogger:
    """
    Append-only JSONL logger capturing the full order lifecycle per trade.

    Call .log(**fields) from any execution context.  I/O is synchronous and
    fast (single append) — safe to call from the async execution loop.
    """

    def __init__(self, audit_dir: Path):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._log_file: Optional[Path] = None
        self._log_date: Optional[str] = None

    def _current_file(self) -> Path:
        today = datetime.utcnow().strftime("%Y%m%d")
        if today != self._log_date:
            self._log_date = today
            self._log_file = self.audit_dir / f"trade_audit_{today}.jsonl"
        return self._log_file

    def log(self, **fields: Any) -> None:
        """
        Append one audit record.

        Recommended fields:
          event        : "ENTRY" | "EXIT" | "PARTIAL_TP" | "REJECTION"
          symbol       : str
          side         : "BUY" | "SELL"
          qty          : float
          fill_price   : float (0.0 if no fill)
          expected_price: float
          slippage_bps : float
          fill_ms      : float  (order-to-fill latency in ms)
          signal       : float
          confidence   : float
          regime       : str
          pnl_pct      : float  (exits / partials only)
          exit_reason  : str    (exits only)
          pretrade     : "PASS" | "REJECT" | "SKIPPED"
          broker       : "alpaca" | "ibkr"
          notes        : str    (optional free text)
        """
        record = {"ts": datetime.utcnow().isoformat() + "Z", **fields}
        try:
            with self._current_file().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.debug("TradeAuditLogger: write failed: %s", exc)
