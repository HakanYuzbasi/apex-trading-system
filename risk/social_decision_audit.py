"""
risk/social_decision_audit.py

Immutable audit trail for social-governor decisions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4


class SocialDecisionAuditRepository:
    """Immutable append-only audit store with optional read fallbacks."""

    def __init__(
        self,
        filepath: Path,
        fallback_filepaths: Optional[Iterable[Path]] = None,
    ) -> None:
        """Initialize primary audit file and optional legacy read paths."""
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_filepaths: List[Path] = []
        for candidate in fallback_filepaths or []:
            path_candidate = Path(candidate)
            if path_candidate == self.filepath:
                continue
            if path_candidate in self.fallback_filepaths:
                continue
            self.fallback_filepaths.append(path_candidate)

    def append_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Append a hash-chained audit event to the primary runtime file."""
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
        """Load the most recent events from all files.

        When no filters are active, uses a fast tail-read (reads from the end of
        the file) so 3+ MB files don't block for 1–2 s on every API call.
        Filtered queries still do a full scan since we can't skip arbitrary rows.
        """
        if limit <= 0:
            return []
        asset_filter = str(asset_class).upper().strip() if asset_class else None
        regime_filter = str(regime).lower().strip() if regime else None
        filtered = bool(asset_filter or regime_filter)

        rows: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        for path in [*self.fallback_filepaths, self.filepath]:
            if not path.exists():
                continue
            lines = (
                self._read_all_lines(path) if filtered
                else self._tail_lines(path, limit * 4)  # overshoot for dedup
            )
            for line in lines:
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
                dedupe_key = str(payload.get("hash") or payload.get("audit_id") or line)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                rows.append(payload)

        return rows[-limit:]

    @staticmethod
    def _read_all_lines(path: Path) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()

    @staticmethod
    def _tail_lines(path: Path, n: int) -> List[str]:
        """Return the last n lines of a file without reading the whole thing."""
        chunk = 1 << 14  # 16 KB chunks
        collected: List[bytes] = []
        total_read = 0
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            pos = size
            while pos > 0 and total_read < chunk * 64:
                read_size = min(chunk, pos)
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size)
                collected.append(data)
                total_read += read_size
                # Stop once we have enough newlines
                combined = b"".join(reversed(collected))
                if combined.count(b"\n") > n + 1:
                    break
        combined = b"".join(reversed(collected))
        return combined.decode("utf-8", errors="replace").splitlines()[-n:]

    def _last_hash(self) -> Optional[str]:
        """Return last known hash, preferring the primary runtime file."""
        primary_hash = self._last_hash_for_path(self.filepath)
        if primary_hash:
            return primary_hash
        for path in self.fallback_filepaths:
            fallback_hash = self._last_hash_for_path(path)
            if fallback_hash:
                return fallback_hash
        return None

    @staticmethod
    def _last_hash_for_path(path: Path) -> Optional[str]:
        """Read final row hash from a single JSONL audit file."""
        if not path.exists():
            return None
        last_line = ""
        with open(path, "r", encoding="utf-8") as f:
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
