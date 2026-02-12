"""Tests for saved model manifest generation and verification."""

from pathlib import Path

from models.model_manifest import build_manifest, verify_manifest, write_manifest


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
