"""Tests for model registry versioning and checksum controls."""

from pathlib import Path

import pytest

from models.model_registry import ModelMetrics, ModelRegistry


def test_model_registry_semantic_versioning(tmp_path: Path):
    registry = ModelRegistry(registry_dir=tmp_path / "registry")
    metrics = ModelMetrics(sharpe_ratio=1.2, win_rate=0.55)

    v1 = registry.register_model({"weights": [1, 2, 3]}, "signal_model", metrics)
    assert v1.version == 1
    assert v1.semantic_version == "0.0.1"

    v2 = registry.register_model(
        {"weights": [4, 5, 6]},
        "signal_model",
        metrics,
        version_bump="minor",
    )
    assert v2.version == 2
    assert v2.semantic_version == "0.1.0"
    assert v2.parent_version == 1

    loaded_model, loaded_meta = registry.load_model("signal_model", semantic_version="0.1.0")
    assert loaded_model["weights"] == [4, 5, 6]
    assert loaded_meta.version == 2


def test_model_registry_checksum_enforcement(tmp_path: Path):
    registry = ModelRegistry(registry_dir=tmp_path / "registry")
    metrics = ModelMetrics(sharpe_ratio=1.0, win_rate=0.5)
    version = registry.register_model({"weights": [1]}, "risk_model", metrics)

    with version.model_path.open("ab") as f:
        f.write(b"tamper")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        registry.load_model("risk_model", version=version.version, strict_checksum=True)

    _, meta = registry.load_model("risk_model", version=version.version, strict_checksum=False)
    assert meta.version == version.version
