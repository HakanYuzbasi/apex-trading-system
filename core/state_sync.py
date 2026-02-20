"""State synchronization utilities for trading runtime and API streaming."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class StateSync:
    """Read/write helper for shared runtime state files."""

    state_file: Path

    def read(self) -> Dict[str, Any]:
        """Read current state from disk, returning empty defaults when missing."""
        if not self.state_file.exists():
            return {}
        with open(self.state_file, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if not isinstance(payload, dict):
            return {}
        return payload

    def write(self, state: Dict[str, Any]) -> None:
        """Write full runtime state atomically with timestamp normalization."""
        state_with_timestamp = dict(state)
        state_with_timestamp.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as file_handle:
            json.dump(state_with_timestamp, file_handle, indent=2, sort_keys=True, default=str)
