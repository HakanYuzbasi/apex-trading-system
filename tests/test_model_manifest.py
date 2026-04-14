"""Tests for saved model manifest generation and verification."""

from pathlib import Path

from models.model_manifest import (
    build_manifest,
    load_manifest,
    sync_manifest_entry,
    verify_manifest,
    write_manifest,
)


def test_manifest_build_and_verify(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "a.json").write_text('{"model":"a"}', encoding="utf-8")
    (model_dir / "b.bin").write_bytes(b"\x00\x01\x02")

    manifest = build_manifest(base_dirs=[model_dir], patterns=("*.json", "*.bin"))
    assert len(manifest["files"]) == 2

    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)
    assert verify_manifest(manifest_path) == []


def test_manifest_detects_tampering(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "weights.json"
    model_file.write_text('{"w":1}', encoding="utf-8")

    manifest = build_manifest(base_dirs=[model_dir], patterns=("*.json",))
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)

    model_file.write_text('{"w":2}', encoding="utf-8")
    errors = verify_manifest(manifest_path)
    assert errors
    assert "checksum_mismatch" in errors[0]


def test_sync_manifest_entry_updates_checksum_for_mutated_file(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "models" / "saved_ultimate"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "q_table_governor.json"
    model_file.write_text('{"weights":[1]}', encoding="utf-8")

    manifest_path = tmp_path / "models" / "model_manifest.json"
    write_manifest(build_manifest(base_dirs=[model_dir], patterns=("*.json",)), manifest_path)

    model_file.write_text('{"weights":[2]}', encoding="utf-8")
    assert verify_manifest(manifest_path)

    sync_manifest_entry(model_file, manifest_path)

    assert verify_manifest(manifest_path) == []


def test_sync_manifest_entry_ignores_files_outside_managed_model_dirs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "models" / "saved_ultimate"
    model_dir.mkdir(parents=True)
    manifest_path = tmp_path / "models" / "model_manifest.json"
    tracked_file = model_dir / "q_table_governor.json"
    tracked_file.write_text('{"weights":[1]}', encoding="utf-8")
    write_manifest(build_manifest(base_dirs=[model_dir], patterns=("*.json",)), manifest_path)

    external_file = tmp_path / "tmp" / "q_table_governor.json"
    external_file.parent.mkdir(parents=True)
    external_file.write_text('{"weights":[99]}', encoding="utf-8")

    sync_manifest_entry(external_file, manifest_path)

    manifest = load_manifest(manifest_path)
    assert verify_manifest(manifest_path) == []
    assert [entry["path"] for entry in manifest["files"]] == ["models/saved_ultimate/q_table_governor.json"]
