"""
Startup trust controls for live/runtime initialization.

This module verifies model artifact integrity before the trading runtime starts
and records an append-only startup audit trail with hash chaining.
"""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from models.model_manifest import load_manifest, verify_manifest

logger = logging.getLogger(__name__)
_GENESIS_HASH = hashlib.sha256(b"apex_startup_trust_genesis").hexdigest()


@dataclass(frozen=True)
class StartupTrustReport:
    status: str
    checked_at: str
    live_trading: bool
    manifest_path: str
    manifest_verified: bool
    manifest_file_count: int
    manifest_generated_at: str | None
    max_manifest_age_days: int
    fail_closed: bool
    effective_fail_closed: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_manifest_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_last_hash(audit_path: Path) -> str:
    if not audit_path.exists():
        return _GENESIS_HASH
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return _GENESIS_HASH
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return _GENESIS_HASH
        return str(payload.get("hash") or _GENESIS_HASH)
    return _GENESIS_HASH


def evaluate_startup_trust(config: Any) -> StartupTrustReport:
    """
    Evaluate startup trust checks without mutating runtime state.

    The config object is expected to expose the ApexConfig-style attributes used
    below. A plain object or test double is sufficient.
    """
    manifest_path = Path(getattr(config, "MODEL_MANIFEST_PATH", "models/model_manifest.json"))
    max_manifest_age_days = int(getattr(config, "MODEL_MANIFEST_MAX_AGE_DAYS", 0))
    fail_closed = bool(getattr(config, "MODEL_MANIFEST_FAIL_CLOSED", True))
    enabled = bool(getattr(config, "MODEL_MANIFEST_VERIFICATION_ENABLED", True))
    live_trading = bool(getattr(config, "LIVE_TRADING", False))
    effective_fail_closed = fail_closed and live_trading
    errors: List[str] = []
    manifest_verified = False
    manifest_file_count = 0
    manifest_generated_at: str | None = None

    if enabled:
        if not manifest_path.exists():
            errors.append(f"manifest_missing: {manifest_path.as_posix()}")
        else:
            try:
                manifest = load_manifest(manifest_path)
                manifest_file_count = len(manifest.get("files", []) or [])
                manifest_generated_at = manifest.get("generated_at")
            except Exception as exc:
                errors.append(f"manifest_load_failed: {manifest_path.as_posix()} ({exc})")
            else:
                errors.extend(verify_manifest(manifest_path))
                manifest_verified = not errors
                if max_manifest_age_days > 0:
                    generated_at = _parse_manifest_timestamp(manifest_generated_at)
                    if generated_at is None:
                        errors.append("manifest_generated_at_invalid")
                    else:
                        age_days = (_utc_now() - generated_at).days
                        if age_days > max_manifest_age_days:
                            errors.append(
                                f"manifest_stale: age_days={age_days} limit={max_manifest_age_days}"
                            )
    else:
        manifest_verified = True

    status = "passed" if not errors else ("blocked" if effective_fail_closed else "degraded")
    return StartupTrustReport(
        status=status,
        checked_at=_utc_now().isoformat(),
        live_trading=live_trading,
        manifest_path=manifest_path.as_posix(),
        manifest_verified=manifest_verified and not errors,
        manifest_file_count=manifest_file_count,
        manifest_generated_at=manifest_generated_at,
        max_manifest_age_days=max_manifest_age_days,
        fail_closed=fail_closed,
        effective_fail_closed=effective_fail_closed,
        errors=errors,
    )


def append_startup_trust_audit(report: StartupTrustReport, audit_root: Path) -> Path:
    """Append a tamper-evident startup trust audit record."""
    audit_dir = Path(audit_root) / "audit" / "startup_trust"
    audit_dir.mkdir(parents=True, exist_ok=True)
    date_str = _utc_now().strftime("%Y%m%d")
    audit_path = audit_dir / f"startup_trust_{date_str}.jsonl"
    prev_hash = _resolve_last_hash(audit_path)
    payload = report.to_dict()
    raw = json.dumps(payload, sort_keys=True)
    record_hash = hashlib.sha256(f"{prev_hash}|{raw}".encode("utf-8")).hexdigest()
    record = {
        **payload,
        "prev_hash": prev_hash,
        "hash": record_hash,
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return audit_path


def enforce_startup_trust(config: Any) -> StartupTrustReport:
    """
    Evaluate and persist startup trust checks, raising on fail-closed violations.
    """
    report = evaluate_startup_trust(config)
    audit_enabled = bool(getattr(config, "STARTUP_TRUST_AUDIT_ENABLED", True))
    audit_path: Path | None = None
    if audit_enabled:
        audit_path = append_startup_trust_audit(report, Path(config.DATA_DIR))
        logger.info("Startup trust audit appended to %s", audit_path)

    if report.errors:
        log_method = logger.error if report.effective_fail_closed else logger.warning
        log_method("Startup trust checks failed: %s", "; ".join(report.errors))
        if report.effective_fail_closed:
            suffix = f" Audit: {audit_path}" if audit_path is not None else ""
            raise RuntimeError(
                "Startup trust checks blocked runtime start. "
                f"Failures: {', '.join(report.errors)}.{suffix}"
            )
    else:
        logger.info(
            "Startup trust checks passed for manifest %s (%s files)",
            report.manifest_path,
            report.manifest_file_count,
        )
    return report
