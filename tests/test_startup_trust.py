"""Tests for startup trust enforcement and audit logging."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.startup_trust import enforce_startup_trust
from models.model_manifest import build_manifest, write_manifest


def _build_config(
    tmp_path: Path,
    manifest_path: Path,
    *,
    fail_closed: bool = True,
    live_trading: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        DATA_DIR=tmp_path / "data",
        LIVE_TRADING=live_trading,
        MODEL_MANIFEST_PATH=str(manifest_path),
        MODEL_MANIFEST_VERIFICATION_ENABLED=True,
        MODEL_MANIFEST_FAIL_CLOSED=fail_closed,
        MODEL_MANIFEST_MAX_AGE_DAYS=0,
        STARTUP_TRUST_AUDIT_ENABLED=True,
    )


def _read_audit_records(data_dir: Path) -> list[dict]:
    audit_files = sorted((data_dir / "audit" / "startup_trust").glob("startup_trust_*.jsonl"))
    assert audit_files
    with audit_files[-1].open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_enforce_startup_trust_passes_and_hash_chains_audit(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.json").write_text('{"weights":[1,2,3]}', encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    write_manifest(build_manifest(base_dirs=[model_dir], patterns=("*.json",)), manifest_path)
    config = _build_config(tmp_path, manifest_path)

    first_report = enforce_startup_trust(config)
    second_report = enforce_startup_trust(config)

    assert first_report.status == "passed"
    assert second_report.status == "passed"

    records = _read_audit_records(config.DATA_DIR)
    assert len(records) == 2
    assert records[0]["status"] == "passed"
    assert records[1]["prev_hash"] == records[0]["hash"]
    assert records[1]["manifest_verified"] is True


def test_enforce_startup_trust_fail_closed_raises_and_writes_failure_audit(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "weights.json"
    model_file.write_text('{"weights":[1]}', encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    write_manifest(build_manifest(base_dirs=[model_dir], patterns=("*.json",)), manifest_path)
    model_file.write_text('{"weights":[99]}', encoding="utf-8")
    config = _build_config(tmp_path, manifest_path, fail_closed=True, live_trading=True)

    with pytest.raises(RuntimeError, match="Startup trust checks blocked runtime start"):
        enforce_startup_trust(config)

    records = _read_audit_records(config.DATA_DIR)
    assert len(records) == 1
    assert records[0]["status"] == "blocked"
    assert records[0]["effective_fail_closed"] is True
    assert any("checksum_mismatch" in error for error in records[0]["errors"])


def test_enforce_startup_trust_degrades_in_paper_mode_even_if_fail_closed_enabled(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "weights.json"
    model_file.write_text('{"weights":[1]}', encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    write_manifest(build_manifest(base_dirs=[model_dir], patterns=("*.json",)), manifest_path)
    model_file.write_text('{"weights":[42]}', encoding="utf-8")
    config = _build_config(tmp_path, manifest_path, fail_closed=True, live_trading=False)

    report = enforce_startup_trust(config)

    assert report.status == "degraded"
    assert report.fail_closed is True
    assert report.effective_fail_closed is False

    records = _read_audit_records(config.DATA_DIR)
    assert len(records) == 1
    assert records[0]["status"] == "degraded"
    assert records[0]["effective_fail_closed"] is False
    assert any("checksum_mismatch" in error for error in records[0]["errors"])
