"""Tests for MetaLabeler — covers BUG-1 (is_bootstrapped_on_synthetic not cleared on load)."""
from __future__ import annotations

import json
import os
import tempfile

import lightgbm as lgb
import numpy as np
import pytest

from core.logic.ml.meta_labeler import MetaLabeler, FEATURE_NAMES, N_FEATURES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trained_model(path: str) -> None:
    """Train a minimal LGB model and save it to *path*."""
    X = np.random.default_rng(0).random((100, N_FEATURES))
    y = (np.random.default_rng(1).random(100) > 0.5).astype(int)
    ds = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    params = {"objective": "binary", "verbosity": -1, "num_leaves": 4, "num_boost_round": 5}
    model = lgb.train(params, ds, num_boost_round=5)
    model.save_model(path)


# ---------------------------------------------------------------------------
# BUG-1: is_bootstrapped_on_synthetic must be False after loading a real model
# ---------------------------------------------------------------------------

def test_loaded_model_clears_synthetic_flag():
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "meta_labeler.lgb")
        _make_trained_model(model_path)

        ml = MetaLabeler(model_path=model_path)

        assert ml._is_trained is True
        assert ml.is_bootstrapped_on_synthetic is False, (
            "BUG-1: loaded model must clear is_bootstrapped_on_synthetic"
        )


def test_loaded_model_returns_real_prediction_not_1():
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "meta_labeler.lgb")
        _make_trained_model(model_path)

        ml = MetaLabeler(model_path=model_path)

        # With a real model predict_confidence() must NOT always return 1.0.
        # We check across a range of inputs — at least one must differ from 1.0.
        results = {
            ml.predict_confidence(kalman_residual=v, bayesian_prob=0.5, vix_level=20.0, sector_concentration=0.0)
            for v in np.linspace(-5, 5, 11)
        }
        assert results != {1.0}, (
            "BUG-1: predict_confidence() always returns 1.0 — synthetic flag not cleared"
        )


# ---------------------------------------------------------------------------
# Pass-through when no model file exists
# ---------------------------------------------------------------------------

def test_no_model_returns_1():
    with tempfile.TemporaryDirectory() as d:
        ml = MetaLabeler(model_path=os.path.join(d, "nonexistent.lgb"))
        assert ml._is_trained is False
        assert ml.predict_confidence() == 1.0


# ---------------------------------------------------------------------------
# Stale model (wrong feature count) → quarantine → pass-through, flag unchanged
# ---------------------------------------------------------------------------

def test_stale_model_quarantined():
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "meta_labeler.lgb")
        # Build a 3-feature model (old v1)
        X = np.random.default_rng(0).random((80, 3))
        y = (np.random.default_rng(2).random(80) > 0.5).astype(int)
        ds = lgb.Dataset(X, label=y)
        lgb.train({"objective": "binary", "verbosity": -1}, ds, num_boost_round=5).save_model(model_path)

        ml = MetaLabeler(model_path=model_path)

        assert ml._is_trained is False
        assert ml.predict_confidence() == 1.0
        assert os.path.exists(model_path + ".v1.bak"), "stale model should be quarantined"
        assert not os.path.exists(model_path), "original stale model should be moved"


# ---------------------------------------------------------------------------
# auto_train_from_attribution sets flag correctly
# ---------------------------------------------------------------------------

def test_auto_train_sets_bootstrapped_false():
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "meta_labeler.lgb")
        attr_path = os.path.join(d, "performance_attribution.json")

        rng = np.random.default_rng(42)
        closed = []
        for _ in range(60):
            closed.append({
                "kalman_residual": float(rng.normal()),
                "bayesian_prob": float(rng.uniform(0.3, 0.8)),
                "vix_level": float(rng.uniform(12, 40)),
                "sector_concentration": float(rng.uniform(0, 0.5)),
                "net_pnl": float(rng.normal(10, 50)),
            })
        with open(attr_path, "w") as f:
            json.dump({"closed_trades": closed}, f)

        ml = MetaLabeler(model_path=model_path)
        assert ml._is_trained is False  # no model on disk yet

        trained = ml.auto_train_from_attribution(attribution_json_path=attr_path, min_samples=50)

        assert trained is True
        assert ml._is_trained is True
        assert ml.is_bootstrapped_on_synthetic is False


def test_auto_train_skipped_when_already_trained():
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "meta_labeler.lgb")
        _make_trained_model(model_path)

        ml = MetaLabeler(model_path=model_path)
        result = ml.auto_train_from_attribution(attribution_json_path="/nonexistent/path.json")
        assert result is False  # already trained — skipped


def test_auto_train_skipped_when_too_few_samples():
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "meta_labeler.lgb")
        attr_path = os.path.join(d, "attr.json")
        with open(attr_path, "w") as f:
            json.dump({"closed_trades": [{"net_pnl": 10, "kalman_residual": 0.1}]}, f)

        ml = MetaLabeler(model_path=model_path)
        result = ml.auto_train_from_attribution(attribution_json_path=attr_path, min_samples=50)
        assert result is False


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

def test_score_untrained_returns_zeros():
    with tempfile.TemporaryDirectory() as d:
        ml = MetaLabeler(model_path=os.path.join(d, "x.lgb"))
        s = ml.score(np.zeros((10, N_FEATURES)), np.zeros(10, dtype=int))
        assert s == {"accuracy": 0.0, "auc": 0.0}


def test_score_trained_returns_metrics():
    with tempfile.TemporaryDirectory() as d:
        model_path = os.path.join(d, "meta_labeler.lgb")
        _make_trained_model(model_path)
        ml = MetaLabeler(model_path=model_path)

        X = np.random.default_rng(7).random((50, N_FEATURES))
        y = (np.random.default_rng(8).random(50) > 0.5).astype(int)
        s = ml.score(X, y)
        assert 0.0 <= s["accuracy"] <= 1.0
        assert 0.0 <= s["auc"] <= 1.0
