"""
Tests for monitoring/hmm_regime_detector.py — HMM-Based Market Regime Detector.

Tests cover: VIX fallback, feature building, state labelling, classify(),
should_retrain(), persistence, and snapshot API. Does NOT require hmmlearn
(all HMM-dependent paths degrade gracefully).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from monitoring.hmm_regime_detector import (
    HMMRegimeDetector,
    _STATE_LABELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector(tmp_path=None, retrain_interval_s: float = 3600.0, **kwargs) -> HMMRegimeDetector:
    return HMMRegimeDetector(
        state_dir=tmp_path,
        min_fit_samples=30,
        retrain_interval_s=retrain_interval_s,
        **kwargs,
    )


def _random_returns(n: int = 120, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.standard_normal(n) * 0.01)


def _vix_series(n: int = 120, base: float = 20.0) -> list[float]:
    return [base + i * 0.01 for i in range(n)]


# ---------------------------------------------------------------------------
# VIX fallback
# ---------------------------------------------------------------------------

class TestVixFallback:
    def test_low_vix_returns_bull(self):
        d = _make_detector()
        label, conf, probs = d.classify([], [10.0])
        assert label == "bull"

    def test_medium_vix_returns_neutral(self):
        d = _make_detector()
        label, conf, probs = d.classify([], [18.0])
        assert label == "neutral"

    def test_high_vix_returns_bear(self):
        d = _make_detector()
        label, conf, probs = d.classify([], [30.0])
        assert label == "bear"

    def test_very_high_vix_returns_volatile(self):
        d = _make_detector()
        label, conf, probs = d.classify([], [50.0])
        assert label == "volatile"

    def test_probs_sum_to_one(self):
        d = _make_detector()
        _, _, probs = d.classify([], [20.0])
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_all_state_labels_present_in_probs(self):
        d = _make_detector()
        _, _, probs = d.classify([], [20.0])
        for lbl in _STATE_LABELS:
            assert lbl in probs

    def test_fallback_method_is_vix_fallback(self):
        d = _make_detector()
        snap = d.get_snapshot()
        assert snap["method"] == "vix_fallback"

    def test_fallback_available_false_when_not_trained(self):
        d = _make_detector()
        snap = d.get_snapshot()
        assert snap["available"] is False


# ---------------------------------------------------------------------------
# _build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_returns_ndarray_for_valid_input(self):
        rets = _random_returns(60)
        vix = _vix_series(60)
        X = HMMRegimeDetector._build_features(rets, vix)
        assert X is not None
        assert isinstance(X, np.ndarray)

    def test_shape_is_n_by_3(self):
        rets = _random_returns(60)
        vix = _vix_series(60)
        X = HMMRegimeDetector._build_features(rets, vix)
        assert X.shape[1] == 3

    def test_returns_none_for_short_input(self):
        X = HMMRegimeDetector._build_features([0.01, 0.02], [20.0, 21.0])
        assert X is None

    def test_vix_feature_clipped_to_01(self):
        rets = _random_returns(30)
        vix = [100.0] * 30  # extreme VIX
        X = HMMRegimeDetector._build_features(rets, vix)
        assert X is not None
        assert X[:, 1].max() <= 1.0

    def test_filters_nan_from_returns(self):
        rets = _random_returns(30)
        rets[5] = float("nan")
        vix = _vix_series(30)
        X = HMMRegimeDetector._build_features(rets, vix)
        # nan row dropped → shorter sequence
        if X is not None:
            assert not np.any(np.isnan(X))


# ---------------------------------------------------------------------------
# _label_states
# ---------------------------------------------------------------------------

class TestLabelStates:
    def test_returns_dict_of_length_n(self):
        means = np.array([0.02, 0.00, -0.01, 0.01])
        vols  = np.array([0.01, 0.01, 0.02, 0.05])
        labels = HMMRegimeDetector._label_states(means, vols)
        assert len(labels) == 4

    def test_highest_vol_state_is_volatile(self):
        means = np.array([0.02, 0.00, -0.01, 0.01])
        vols  = np.array([0.01, 0.01, 0.02, 0.05])
        labels = HMMRegimeDetector._label_states(means, vols)
        volatile_idx = int(np.argmax(vols))
        assert labels[volatile_idx] == "volatile"

    def test_best_return_non_volatile_state_is_bull(self):
        means = np.array([0.05, 0.00, -0.03, 0.01])
        vols  = np.array([0.01, 0.01, 0.01, 0.10])
        labels = HMMRegimeDetector._label_states(means, vols)
        assert labels[0] == "bull"   # highest mean return

    def test_worst_return_non_volatile_state_is_bear(self):
        means = np.array([0.05, 0.00, -0.03, 0.01])
        vols  = np.array([0.01, 0.01, 0.01, 0.10])
        labels = HMMRegimeDetector._label_states(means, vols)
        assert labels[2] == "bear"   # most negative mean return


# ---------------------------------------------------------------------------
# should_retrain
# ---------------------------------------------------------------------------

class TestShouldRetrain:
    def test_true_when_never_trained(self):
        d = _make_detector(retrain_interval_s=3600.0)
        assert d.should_retrain() is True

    def test_false_immediately_after_train_sets_timestamp(self):
        d = _make_detector(retrain_interval_s=3600.0)
        d._trained_at = time.time()
        assert d.should_retrain() is False

    def test_true_after_interval_elapsed(self):
        d = _make_detector(retrain_interval_s=1.0)
        d._trained_at = time.time() - 2.0
        assert d.should_retrain() is True


# ---------------------------------------------------------------------------
# get_state / get_snapshot
# ---------------------------------------------------------------------------

class TestGetSnapshot:
    def test_snapshot_has_required_keys(self):
        d = _make_detector()
        snap = d.get_snapshot()
        for key in ["available", "method", "current_label", "confidence", "state_probs", "n_states"]:
            assert key in snap

    def test_state_probs_sum_to_one_before_train(self):
        d = _make_detector()
        snap = d.get_snapshot()
        total = sum(snap["state_probs"].values())
        assert abs(total - 1.0) < 1e-6

    def test_n_states_matches_constructor(self):
        d = HMMRegimeDetector(n_states=4)
        assert d.get_snapshot()["n_states"] == 4


# ---------------------------------------------------------------------------
# fit (only runs if hmmlearn is installed)
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_with_insufficient_data_returns_false(self):
        d = _make_detector()
        result = d.fit([0.01, 0.02], [20.0, 21.0])
        assert result is False

    def test_fit_with_sufficient_data(self):
        pytest.importorskip("hmmlearn")
        d = HMMRegimeDetector(min_fit_samples=30)
        rets = _random_returns(120)
        vix  = _vix_series(120)
        result = d.fit(rets, vix)
        assert result is True
        assert d._model is not None
        assert len(d._state_map) == 4

    def test_classify_returns_valid_label_after_fit(self):
        pytest.importorskip("hmmlearn")
        d = HMMRegimeDetector(min_fit_samples=30)
        rets = _random_returns(120)
        vix  = _vix_series(120)
        d.fit(rets, vix)
        label, conf, probs = d.classify(rets[-30:], vix[-30:])
        assert label in _STATE_LABELS
        assert 0.0 <= conf <= 1.0
        assert abs(sum(probs.values()) - 1.0) < 1e-3

    def test_available_true_after_fit(self):
        pytest.importorskip("hmmlearn")
        d = HMMRegimeDetector(min_fit_samples=30)
        d.fit(_random_returns(120), _vix_series(120))
        assert d.get_snapshot()["available"] is True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_load_survives_missing_files(self, tmp_path):
        d = _make_detector(tmp_path=tmp_path)
        assert d._model is None  # nothing to load

    def test_load_survives_corrupt_json(self, tmp_path):
        (tmp_path / "hmm_meta.json").write_text("NOT JSON")
        d = _make_detector(tmp_path=tmp_path)  # should not raise
        assert d._model is None

    def test_save_and_load_roundtrip(self, tmp_path):
        pytest.importorskip("hmmlearn")
        d = HMMRegimeDetector(state_dir=tmp_path, min_fit_samples=30)
        d.fit(_random_returns(120), _vix_series(120))
        state_map = dict(d._state_map)

        d2 = _make_detector(tmp_path=tmp_path)
        assert d2._model is not None
        assert d2._state_map == state_map

    def test_meta_json_valid_after_save(self, tmp_path):
        pytest.importorskip("hmmlearn")
        d = HMMRegimeDetector(state_dir=tmp_path, min_fit_samples=30)
        d.fit(_random_returns(120), _vix_series(120))
        meta = json.loads((tmp_path / "hmm_meta.json").read_text())
        assert "state_map" in meta
        assert "trained_at" in meta
