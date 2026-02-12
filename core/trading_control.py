"""
core/trading_control.py

File-backed control command helpers for coordinating external operational actions
with the live trading loop (e.g. kill-switch reset requests).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from uuid import uuid4


def _default_state() -> Dict[str, object]:
    return {
        "kill_switch_reset_requested": False,
        "request_id": None,
        "requested_at": None,
        "requested_by": None,
        "reason": None,
        "processed_at": None,
        "processed_by": None,
        "processing_note": None,
        "governor_policy_reload_requested": False,
        "governor_policy_reload_request_id": None,
        "governor_policy_reload_requested_at": None,
        "governor_policy_reload_requested_by": None,
        "governor_policy_reload_reason": None,
        "governor_policy_reload_processed_at": None,
        "governor_policy_reload_processed_by": None,
        "governor_policy_reload_processing_note": None,
    }


def read_control_state(filepath: Path) -> Dict[str, object]:
    if not filepath.exists():
        return _default_state()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_state()
        baseline = _default_state()
        baseline.update(data)
        return baseline
    except Exception:
        return _default_state()


def write_control_state(filepath: Path, state: Dict[str, object]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = _default_state()
    payload.update(state)
    tmp = filepath.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(filepath)


def request_kill_switch_reset(filepath: Path, requested_by: str, reason: str) -> Dict[str, object]:
    state = read_control_state(filepath)
    state.update(
        {
            "kill_switch_reset_requested": True,
            "request_id": f"ksr-{uuid4().hex[:12]}",
            "requested_at": datetime.utcnow().isoformat(),
            "requested_by": requested_by,
            "reason": reason,
            "processed_at": None,
            "processed_by": None,
            "processing_note": None,
        }
    )
    write_control_state(filepath, state)
    return state


def mark_kill_switch_reset_processed(
    filepath: Path,
    processed_by: str,
    note: str,
) -> Dict[str, object]:
    state = read_control_state(filepath)
    state.update(
        {
            "kill_switch_reset_requested": False,
            "processed_at": datetime.utcnow().isoformat(),
            "processed_by": processed_by,
            "processing_note": note,
        }
    )
    write_control_state(filepath, state)
    return state


def request_governor_policy_reload(filepath: Path, requested_by: str, reason: str) -> Dict[str, object]:
    state = read_control_state(filepath)
    state.update(
        {
            "governor_policy_reload_requested": True,
            "governor_policy_reload_request_id": f"gpr-{uuid4().hex[:12]}",
            "governor_policy_reload_requested_at": datetime.utcnow().isoformat(),
            "governor_policy_reload_requested_by": requested_by,
            "governor_policy_reload_reason": reason,
            "governor_policy_reload_processed_at": None,
            "governor_policy_reload_processed_by": None,
            "governor_policy_reload_processing_note": None,
        }
    )
    write_control_state(filepath, state)
    return state


def mark_governor_policy_reload_processed(
    filepath: Path,
    processed_by: str,
    note: str,
) -> Dict[str, object]:
    state = read_control_state(filepath)
    state.update(
        {
            "governor_policy_reload_requested": False,
            "governor_policy_reload_processed_at": datetime.utcnow().isoformat(),
            "governor_policy_reload_processed_by": processed_by,
            "governor_policy_reload_processing_note": note,
        }
    )
    write_control_state(filepath, state)
    return state
