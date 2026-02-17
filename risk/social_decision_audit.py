"""
risk/social_decision_audit.py

Immutable audit trail for social-governor decisions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class SocialDecisionAuditRepository:
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def append_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = dict(payload)
        row.setdefault("audit_id", f"sgd-{uuid4().hex[:12]}")
        row.setdefault("timestamp", datetime.utcnow().isoformat())
        row["prev_hash"] = self._last_hash()

        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        row["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def load_events(
        self,
        *,
        limit: int = 200,
        asset_class: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        if not self.filepath.exists():
            return []
        asset_filter = str(asset_class).upper().strip() if asset_class else None
        regime_filter = str(regime).lower().strip() if regime else None
        rows: List[Dict[str, Any]] = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                if asset_filter and str(payload.get("asset_class", "")).upper() != asset_filter:
                    continue
                if regime_filter and str(payload.get("regime", "")).lower() != regime_filter:
                    continue
                rows.append(payload)
        return rows[-limit:]

    def _last_hash(self) -> Optional[str]:
        if not self.filepath.exists():
            return None
        last_line = ""
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return None
        try:
            payload = json.loads(last_line)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get("hash")
        return str(value) if value else None
