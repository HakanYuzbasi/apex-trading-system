"""tests/test_regime_transition_predictor.py — Regime transition predictor tests."""
from __future__ import annotations

import math

from monitoring.regime_transition_predictor import (
    RegimeTransitionPredictor,
    TransitionPrediction,
    _WEIGHTS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_predictor(**kwargs) -> RegimeTransitionPredictor:
    return RegimeTransitionPredictor(**kwargs)


def _feed_rising_vix(p: RegimeTransitionPredictor, base_vix: float = 15.0, n: int = 20) -> None:
    """Simulate a fast-rising VIX scenario."""
    for i in range(n):
        p.update(vix=base_vix + i * 1.2, spy_return_1h=-0.002, current_regime="bull")


def _feed_stable(p: RegimeTransitionPredictor, n: int = 25) -> None:
    """Simulate stable market conditions."""
    for i in range(n):
        p.update(vix=18.0, spy_return_1h=0.001, current_regime="neutral")


# ── TransitionPrediction dataclass ────────────────────────────────────────────

class TestTransitionPrediction:
    def test_to_dict_keys(self):
        pred = TransitionPrediction(
            probability=0.45,
            direction="risk_off",
            indicator_scores={"vix_velocity": 0.3},
            size_multiplier=0.85,
            regime_age_hours=12.0,
        )
        d = pred.to_dict()
        assert "probability" in d
        assert "direction" in d
        assert "indicator_scores" in d
        assert "size_multiplier" in d
        assert "regime_age_hours" in d
        assert "updated_at" in d

    def test_values_rounded(self):
        pred = TransitionPrediction(
            probability=0.12345678,
            direction="unknown",
            indicator_scores={},
            size_multiplier=0.99999,
            regime_age_hours=1.23456,
        )
        d = pred.to_dict()
        assert d["probability"] == round(0.12345678, 4)
        assert d["regime_age_hours"] == round(1.23456, 2)


# ── Indicator scores ──────────────────────────────────────────────────────────

class TestIndicatorScores:
    def test_vix_velocity_zero_when_insufficient_data(self):
        p = _make_predictor()
        p.update(vix=20.0, spy_return_1h=0.0)
        s = p._score_vix_velocity()
        assert s == 0.0

    def test_vix_velocity_positive_when_vix_rising(self):
        p = _make_predictor()
        _feed_rising_vix(p)
        s = p._score_vix_velocity()
        assert s > 0

    def test_vix_velocity_clamped_to_one(self):
        p = _make_predictor()
        for i in range(30):
            p.update(vix=15.0 + i * 3.0)  # extreme rise
        s = p._score_vix_velocity()
        assert s <= 1.0

    def test_vix_extreme_high_vix(self):
        p = _make_predictor()
        p.update(vix=45.0)
        s = p._score_vix_extreme()
        assert s > 0.0

    def test_vix_extreme_low_vix(self):
        p = _make_predictor()
        p.update(vix=10.0)
        s = p._score_vix_extreme()
        assert s > 0.0

    def test_vix_extreme_normal_vix(self):
        p = _make_predictor()
        p.update(vix=20.0)
        s = p._score_vix_extreme()
        assert s == 0.0

    def test_momentum_exhaustion_zero_when_insufficient_data(self):
        p = _make_predictor()
        p.update(vix=20.0, spy_return_1h=0.005)
        s = p._score_momentum_exhaustion()
        assert s == 0.0

    def test_momentum_exhaustion_high_when_oscillating(self):
        p = _make_predictor()
        for i in range(15):
            sign = 1 if i % 2 == 0 else -1
            p.update(vix=20.0, spy_return_1h=sign * 0.01)
        s = p._score_momentum_exhaustion()
        assert s > 0.5

    def test_momentum_exhaustion_low_when_trending(self):
        p = _make_predictor()
        for _ in range(15):
            p.update(vix=20.0, spy_return_1h=0.005)
        s = p._score_momentum_exhaustion()
        assert s < 0.3

    def test_regime_age_increases_with_time(self):
        """Inject known age via monkeypatching start timestamp."""
        import time
        p = _make_predictor()
        p._regime_start_ts = time.time() - 72 * 3600  # 72 hours ago
        s = p._score_regime_age()
        assert s > 0.5  # 72h > 48h inflection point

    def test_regime_age_low_when_fresh(self):
        p = _make_predictor()
        # Just started (default start = now)
        s = p._score_regime_age()
        assert s < 0.5

    def test_signal_variance_zero_when_insufficient(self):
        p = _make_predictor()
        s = p._score_signal_variance()
        assert s == 0.0

    def test_signal_variance_high_when_noisy(self):
        p = _make_predictor()
        import random
        random.seed(42)
        for _ in range(25):
            p.update(vix=20.0, composite_signal=random.uniform(-0.2, 0.2))
        s = p._score_signal_variance()
        assert s > 0.0


# ── Probability aggregation ───────────────────────────────────────────────────

class TestProbabilityAggregation:
    def test_weights_sum_to_one(self):
        total = sum(_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_zero_scores_gives_zero_probability(self):
        p = _make_predictor()
        scores = {k: 0.0 for k in _WEIGHTS}
        prob = p._aggregate_probability(scores)
        assert prob == 0.0

    def test_all_one_scores_gives_one_probability(self):
        p = _make_predictor()
        scores = {k: 1.0 for k in _WEIGHTS}
        prob = p._aggregate_probability(scores)
        assert abs(prob - 1.0) < 1e-9

    def test_probability_bounded(self):
        p = _make_predictor()
        scores = {k: 2.0 for k in _WEIGHTS}  # intentionally > 1
        prob = p._aggregate_probability(scores)
        assert 0.0 <= prob <= 1.0

    def test_negative_scores_treated_as_zero(self):
        p = _make_predictor()
        scores = {k: -1.0 for k in _WEIGHTS}
        prob = p._aggregate_probability(scores)
        assert prob == 0.0


# ── Size multiplier ───────────────────────────────────────────────────────────

class TestSizeMultiplier:
    def test_below_threshold_returns_one(self):
        p = _make_predictor(high_prob_threshold=0.60, min_size_mult=0.60)
        mult = p._size_multiplier(0.40)
        assert mult == 1.0

    def test_at_threshold_returns_one(self):
        p = _make_predictor(high_prob_threshold=0.60)
        mult = p._size_multiplier(0.60)
        assert mult == 1.0

    def test_at_one_returns_min(self):
        p = _make_predictor(high_prob_threshold=0.60, min_size_mult=0.60)
        mult = p._size_multiplier(1.0)
        assert abs(mult - 0.60) < 1e-9

    def test_intermediate_probability(self):
        p = _make_predictor(high_prob_threshold=0.60, min_size_mult=0.60)
        mult = p._size_multiplier(0.80)  # halfway between 0.60 and 1.0
        assert 0.60 < mult < 1.0

    def test_mult_always_non_negative(self):
        p = _make_predictor()
        for prob in [0.0, 0.5, 0.7, 0.9, 1.0, 1.5]:
            assert p._size_multiplier(prob) >= 0.0


# ── Direction inference ───────────────────────────────────────────────────────

class TestDirectionInference:
    def test_risk_off_when_bull_plus_high_vix_velocity(self):
        p = _make_predictor()
        p._current_regime = "bull"
        scores = {"vix_velocity": 0.50, "momentum_exhaustion": 0.30,
                  "vix_extreme": 0.0, "regime_age": 0.0, "signal_variance": 0.0}
        direction = p._infer_direction(scores)
        assert direction == "risk_off"

    def test_unknown_when_neutral(self):
        p = _make_predictor()
        p._current_regime = "neutral"
        scores = {k: 0.0 for k in _WEIGHTS}
        direction = p._infer_direction(scores)
        assert direction == "unknown"

    def test_risk_on_when_bear_falling_vix(self):
        p = _make_predictor()
        p._current_regime = "bear"
        scores = {"vix_velocity": -0.20, "momentum_exhaustion": 0.10,
                  "vix_extreme": 0.0, "regime_age": 0.0, "signal_variance": 0.0}
        direction = p._infer_direction(scores)
        assert direction == "risk_on"


# ── predict() integration ─────────────────────────────────────────────────────

class TestPredict:
    def test_returns_transition_prediction(self):
        p = _make_predictor()
        _feed_stable(p)
        pred = p.predict()
        assert isinstance(pred, TransitionPrediction)

    def test_probability_in_range(self):
        p = _make_predictor()
        _feed_stable(p)
        pred = p.predict()
        assert 0.0 <= pred.probability <= 1.0

    def test_size_mult_in_range(self):
        p = _make_predictor()
        _feed_stable(p)
        pred = p.predict()
        assert 0.0 <= pred.size_multiplier <= 1.0

    def test_high_probability_under_stress(self):
        p = _make_predictor()
        _feed_rising_vix(p, base_vix=14.0, n=25)
        pred = p.predict()
        # With rapid VIX rise, probability should be elevated
        assert pred.probability > 0.0

    def test_get_size_multiplier_uses_cache(self):
        p = _make_predictor()
        _feed_stable(p)
        pred = p.predict()
        # get_size_multiplier should return same value as last predict
        assert p.get_size_multiplier() == pred.size_multiplier

    def test_get_last_prediction_none_before_predict(self):
        p = _make_predictor()
        assert p.get_last_prediction() is None

    def test_get_last_prediction_after_predict(self):
        p = _make_predictor()
        _feed_stable(p)
        p.predict()
        result = p.get_last_prediction()
        assert result is not None
        assert "probability" in result

    def test_regime_change_resets_age(self):
        import time
        p = _make_predictor()
        p._regime_start_ts = time.time() - 48 * 3600
        old_age = p._regime_age_hours()
        # Simulate regime transition
        p.update(vix=20.0, current_regime="bear")
        new_age = p._regime_age_hours()
        assert new_age < old_age
