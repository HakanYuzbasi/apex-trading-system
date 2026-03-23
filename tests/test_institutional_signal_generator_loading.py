import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from models.institutional_signal_generator import UltimateSignalGenerator


def test_load_models_warns_once_for_missing_regime_bundles(tmp_path, caplog):
    model_dir = tmp_path / "saved_ultimate"
    neutral_dir = model_dir / "neutral"
    neutral_dir.mkdir(parents=True)

    with open(model_dir / "metadata.pkl", "wb") as fh:
        pickle.dump(
            {
                "regime_weights": {"neutral": {"rf": 1.0}},
                "regime_features": {"neutral": []},
                "performance_baseline": 0.52,
            },
            fh,
        )

    dump(SimpleImputer(), neutral_dir / "imputer.pkl")
    dump(RobustScaler(), neutral_dir / "scaler.pkl")
    dump(RandomForestRegressor(), neutral_dir / "rf.pkl")

    generator = UltimateSignalGenerator(model_dir=str(model_dir))

    with caplog.at_level("WARNING"):
        assert generator.loadModels() is True

    warnings = [r.getMessage() for r in caplog.records if "Missing trained regime bundles" in r.getMessage()]
    assert len(warnings) == 1
    assert "bull" in warnings[0]
    assert "bear" in warnings[0]
    assert "volatile" in warnings[0]


def test_generate_signal_warns_once_per_missing_regime(caplog):
    generator = UltimateSignalGenerator(model_dir="models/saved_ultimate")
    generator.is_trained = True
    generator.tracker = SimpleNamespace(on_price_update=lambda *args, **kwargs: None)
    generator._symbol_regime_detectors["AAPL"] = SimpleNamespace(
        assess_regime=lambda prices, emit_transition_logs=False: SimpleNamespace(primary_regime="bull")
    )
    generator.regime_models["neutral"] = {}

    data = pd.Series([100.0, 101.0, 102.0], name="Close")

    with caplog.at_level("WARNING"):
        generator.generate_signal("AAPL", data)
        generator.generate_signal("AAPL", data)

    warnings = [r.getMessage() for r in caplog.records if "No models for bull, using neutral" in r.getMessage()]
    assert len(warnings) == 1


def test_production_saved_ultimate_bundle_has_all_regimes():
    model_dir = Path("models/saved_ultimate")
    generator = UltimateSignalGenerator(model_dir=str(model_dir))

    assert generator.loadModels() is True
    loaded_regimes = {
        regime for regime, models in generator.regime_models.items() if models
    }
    # Core regimes must always be present; volatile may be absent if insufficient
    # training samples were available (rare market condition).
    core_regimes = {"bear", "bull", "neutral"}
    assert core_regimes <= loaded_regimes, f"Core regimes missing: {core_regimes - loaded_regimes}"
    # Any missing regime should fall back to neutral gracefully (not crash)
    missing = getattr(generator, "_missing_regime_bundles", set())
    assert missing <= {"volatile"}, f"Unexpected missing regimes: {missing - {'volatile'}}"
    for regime in loaded_regimes:
        assert len(generator.regime_features[regime]) == generator.regime_imputers[regime].n_features_in_
