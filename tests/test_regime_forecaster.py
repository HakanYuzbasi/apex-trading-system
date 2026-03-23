"""Tests for RegimeTransitionForecaster."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from monitoring.regime_forecaster import RegimeTransitionForecaster, RegimeTransitionForecast


def _make_forecaster(**kwargs) -> RegimeTransitionForecaster:
    defaults = dict(window=10, warn_prob=0.60, caution_prob=0.40)
    defaults.update(kwargs)
    return RegimeTransitionForecaster(**defaults)


def _feed_calm(fc: RegimeTransitionForecaster, n: int = 12) -> None:
    """Feed calm market data: low VIX, stable PCR, rising HYG/SPY."""
    for i in range(n):
        fc.update(vix=15.0, pcr=0.85, hyg_price=80.0 + i * 0.01, spy_price=500.0 + i * 0.05, vix3m=16.0)


def _feed_stressed(fc: RegimeTransitionForecaster, n: int = 12) -> None:
    """Feed stressed data: spiking VIX, rising PCR, falling HYG."""
    for i in range(n):
        fc.update(vix=25.0 + i * 1.5, pcr=1.3 + i * 0.05,
                  hyg_price=78.0 - i * 0.2, spy_price=490.0 - i * 0.5,
                  vix3m=22.0 + i * 0.5)


class TestDefaultForecast:

    def test_no_data_returns_clear(self):
        fc = _make_forecaster()
        result = fc.get_forecast()
        assert result.signal == "clear"
        assert result.size_multiplier == 1.0

    def test_insufficient_data_clear(self):
        fc = _make_forecaster()
        fc.update(vix=20.0, pcr=1.0, hyg_price=80.0, spy_price=500.0)
        result = fc.get_forecast()
        assert result.signal == "clear"
        assert result.size_multiplier == pytest.approx(1.0, abs=0.01)

    def test_forecast_is_regime_transition_forecast(self):
        fc = _make_forecaster()
        result = fc.get_forecast()
        assert isinstance(result, RegimeTransitionForecast)


class TestCalmMarket:

    def test_calm_data_stays_clear(self):
        fc = _make_forecaster()
        _feed_calm(fc)
        result = fc.get_forecast()
        assert result.signal == "clear"
        assert result.size_multiplier == pytest.approx(1.0, abs=0.05)

    def test_calm_prob_low(self):
        fc = _make_forecaster()
        _feed_calm(fc)
        assert fc.get_forecast().transition_prob < 0.40


class TestStressedMarket:

    def test_stressed_data_raises_signal(self):
        fc = _make_forecaster()
        _feed_stressed(fc, n=12)
        result = fc.get_forecast()
        assert result.signal in ("caution", "warning")

    def test_stressed_mult_below_one(self):
        fc = _make_forecaster()
        _feed_stressed(fc, n=12)
        assert fc.get_forecast().size_multiplier < 1.0

    def test_warning_mult_lower_than_caution(self):
        fc = _make_forecaster(warn_prob=0.30, caution_prob=0.20)
        _feed_stressed(fc, n=15)
        result = fc.get_forecast()
        if result.signal == "warning":
            assert result.size_multiplier <= 0.65
        elif result.signal == "caution":
            assert result.size_multiplier <= 0.82


class TestSigmoid:

    def test_sigmoid_zero_is_half(self):
        fc = _make_forecaster()
        assert fc._sigmoid(0.0) == pytest.approx(0.5, abs=0.001)

    def test_sigmoid_positive_above_half(self):
        fc = _make_forecaster()
        assert fc._sigmoid(2.0) > 0.5

    def test_sigmoid_negative_below_half(self):
        fc = _make_forecaster()
        assert fc._sigmoid(-2.0) < 0.5

    def test_sigmoid_bounded(self):
        fc = _make_forecaster()
        assert 0 < fc._sigmoid(100.0) <= 1.0
        assert 0 <= fc._sigmoid(-100.0) < 1.0


class TestZscore:

    def test_zscore_constant_returns_zero(self):
        fc = _make_forecaster()
        assert fc._zscore([5.0, 5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-6)

    def test_zscore_last_above_mean_positive(self):
        fc = _make_forecaster()
        series = [10.0, 10.0, 10.0, 10.0, 20.0]
        z = fc._zscore(series)
        assert z > 1.0

    def test_zscore_single_value_returns_zero(self):
        fc = _make_forecaster()
        assert fc._zscore([5.0]) == 0.0


class TestFeatures:

    def test_features_dict_populated(self):
        fc = _make_forecaster()
        _feed_calm(fc, n=8)
        result = fc.get_forecast()
        assert "vix_z" in result.features
        assert "vix_roc_3bar" in result.features
        assert "vix_term_slope" in result.features
        assert "n_obs" in result.features

    def test_transition_prob_in_zero_one(self):
        fc = _make_forecaster()
        _feed_stressed(fc, n=12)
        assert 0.0 <= fc.get_forecast().transition_prob <= 1.0


class TestPersistence:

    def test_state_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            fc = RegimeTransitionForecaster(data_dir=Path(tmp))
            _feed_stressed(fc, n=12)
            path = Path(tmp) / "regime_forecaster.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert "last_forecast" in data

    def test_state_loaded_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            fc1 = RegimeTransitionForecaster(data_dir=Path(tmp))
            _feed_stressed(fc1, n=12)
            prob1 = fc1.get_forecast().transition_prob

            fc2 = RegimeTransitionForecaster(data_dir=Path(tmp))
            assert fc2.get_forecast().transition_prob == pytest.approx(prob1, abs=0.01)

    def test_no_dir_no_crash(self):
        fc = RegimeTransitionForecaster(data_dir=None)
        _feed_calm(fc, n=8)
        assert fc.get_forecast() is not None


class TestRegimeTransition:

    def test_calm_then_stress_raises_alert(self):
        fc = _make_forecaster(window=10, warn_prob=0.30, caution_prob=0.15)
        _feed_calm(fc, n=10)
        assert fc.get_forecast().signal == "clear"
        # Inject severe stress — VIX spiking, PCR surging, HYG collapsing
        for i in range(10):
            fc.update(vix=35.0 + i * 2, pcr=1.8 + i * 0.1,
                      hyg_price=72.0 - i * 0.5, spy_price=460.0 - i, vix3m=28.0)
        result = fc.get_forecast()
        assert result.signal in ("caution", "warning")

    def test_stress_then_calm_recovers(self):
        fc = _make_forecaster()
        _feed_stressed(fc, n=8)
        for _ in range(12):
            fc.update(vix=14.0, pcr=0.80, hyg_price=81.0, spy_price=510.0, vix3m=15.0)
        result = fc.get_forecast()
        # Should recover toward clear
        assert result.transition_prob < 0.65
