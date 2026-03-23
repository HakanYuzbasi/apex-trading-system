"""Tests for ModelDriftMonitor."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from monitoring.model_drift_monitor import (
    ModelDriftMonitor,
    DriftStatus,
    WindowStats,
    _IC_HEALTHY,
    _HIT_RATE_HEALTHY,
    _CONF_HEALTHY,
)


def _monitor(**kw) -> ModelDriftMonitor:
    defaults = dict(window_size=10, ic_retrain_threshold=0.01, consecutive_degrade_limit=2)
    defaults.update(kw)
    return ModelDriftMonitor(**defaults)


def _feed_correct(monitor: ModelDriftMonitor, n: int, conf: float = 0.70) -> None:
    """Feed n positively-correlated signal/return pairs (IC > 0, all directionally correct)."""
    for i in range(n):
        sig = 0.20 + 0.03 * (i % 5)   # varies 0.20–0.32 (positive, different magnitude)
        ret = sig * 0.05               # return proportional to signal → high positive IC
        sym = f"SYM{i}"
        monitor.record_signal(sym, sig, conf)
        monitor.record_outcome(sym, sig, ret)


def _feed_wrong(monitor: ModelDriftMonitor, n: int, conf: float = 0.40) -> None:
    """Feed n anti-correlated signal/return pairs (IC < 0, all directionally wrong)."""
    for i in range(n):
        sig = 0.20 + 0.03 * (i % 5)   # positive signal
        ret = -(sig * 0.05)            # opposite sign → negative IC
        sym = f"SYM{i}"
        monitor.record_signal(sym, sig, conf)
        monitor.record_outcome(sym, sig, ret)


def _feed_noise(monitor: ModelDriftMonitor, n: int) -> None:
    """Feed n random-signal, random-return pairs (no IC)."""
    import random
    rng = random.Random(42)
    for i in range(n):
        sig = rng.choice([-0.5, 0.5])
        ret = rng.choice([-0.01, 0.01])
        monitor.record_signal(f"X{i}", sig, 0.55)
        monitor.record_outcome(f"X{i}", sig, ret)


# ── Default / empty state ─────────────────────────────────────────────────────

class TestDefaultState:

    def test_get_status_returns_drift_status(self):
        m = _monitor()
        assert isinstance(m.get_status(), DriftStatus)

    def test_empty_health_is_healthy(self):
        m = _monitor()
        assert m.get_status().health == "healthy"

    def test_empty_should_not_retrain(self):
        m = _monitor()
        assert m.get_status().should_retrain is False

    def test_empty_total_windows_zero(self):
        m = _monitor()
        assert m.get_status().total_windows == 0

    def test_get_report_is_dict(self):
        m = _monitor()
        assert isinstance(m.get_report(), dict)


# ── record_signal / record_outcome ────────────────────────────────────────────

class TestRecording:

    def test_record_signal_stores_pending(self):
        m = _monitor()
        m.record_signal("AAPL", 0.5, 0.70)
        assert "AAPL" in m._pending

    def test_record_outcome_clears_pending(self):
        m = _monitor()
        m.record_signal("AAPL", 0.5, 0.70)
        m.record_outcome("AAPL", 0.5, 0.01)
        assert "AAPL" not in m._pending

    def test_record_outcome_without_prior_signal_ok(self):
        m = _monitor()
        # Should not crash even without a matching record_signal
        m.record_outcome("MSFT", 0.4, 0.01)

    def test_zero_signal_skipped(self):
        """Neutral signals (0.0) are not directionally meaningful, skip them."""
        m = _monitor(window_size=5)
        for i in range(5):
            m.record_signal(f"S{i}", 0.0, 0.55)
            m.record_outcome(f"S{i}", 0.0, 0.01)
        # Zero signal should be skipped → no window completed
        assert m.get_status().total_windows == 0

    def test_window_triggers_after_n_obs(self):
        m = _monitor(window_size=5)
        _feed_correct(m, 5)
        assert m.get_status().total_windows == 1

    def test_two_windows_counted(self):
        m = _monitor(window_size=5)
        _feed_correct(m, 10)
        assert m.get_status().total_windows == 2


# ── Health classification ─────────────────────────────────────────────────────

class TestHealthClassification:

    def test_correct_predictions_healthy(self):
        m = _monitor(window_size=10)
        _feed_correct(m, 10)
        assert m.get_status().health == "healthy"

    def test_wrong_predictions_degrading(self):
        m = _monitor(window_size=10)
        _feed_wrong(m, 10)
        status = m.get_status()
        assert status.health in ("degrading", "critical")

    def test_degrading_increments_consecutive(self):
        m = _monitor(window_size=10)
        _feed_wrong(m, 20)  # 2 windows
        assert m.get_status().consecutive_degraded >= 1

    def test_correct_resets_consecutive(self):
        m = _monitor(window_size=10)
        _feed_wrong(m, 10)   # 1 bad window
        _feed_correct(m, 10) # 1 good window
        assert m.get_status().consecutive_degraded == 0

    def test_should_retrain_after_consecutive_degrade(self):
        # window_size=10, consecutive_degrade_limit=2
        m = _monitor(window_size=10, consecutive_degrade_limit=2, ic_retrain_threshold=1.0)
        # IC threshold = 1.0 (always trigger when degraded)
        _feed_wrong(m, 20)  # 2 consecutive degraded windows
        assert m.get_status().should_retrain is True

    def test_single_bad_window_no_retrain(self):
        m = _monitor(window_size=10, consecutive_degrade_limit=2)
        _feed_wrong(m, 10)
        assert m.get_status().should_retrain is False  # only 1 window

    def test_retrain_flag_true_gives_true_should_retrain(self):
        """Verify should_retrain attribute is accessible and boolean."""
        m = _monitor(window_size=10, consecutive_degrade_limit=2, ic_retrain_threshold=1.0)
        _feed_wrong(m, 20)
        status = m.get_status()
        assert isinstance(status.should_retrain, bool)
        assert status.should_retrain is True

    def test_to_dict_has_should_retrain_key(self):
        m = _monitor()
        d = m.get_status().to_dict()
        assert "should_retrain" in d


# ── IC computation ────────────────────────────────────────────────────────────

class TestIC:

    def test_perfect_positive_ic(self):
        """Signals perfectly correlated with returns → IC ≈ 1."""
        m = _monitor(window_size=10)
        signals = [0.1 * i for i in range(1, 11)]
        returns = [0.01 * i for i in range(1, 11)]
        for i, (s, r) in enumerate(zip(signals, returns)):
            m.record_signal(f"S{i}", s, 0.65)
            m.record_outcome(f"S{i}", s, r)
        ic = m.get_status().ic_current
        assert ic > 0.90

    def test_negatively_correlated_ic_negative(self):
        """Signals anti-correlated with returns → IC < 0."""
        m = _monitor(window_size=10)
        signals = [0.1 * i for i in range(1, 11)]
        returns = [-0.01 * i for i in range(1, 11)]
        for i, (s, r) in enumerate(zip(signals, returns)):
            m.record_signal(f"S{i}", s, 0.65)
            m.record_outcome(f"S{i}", s, r)
        ic = m.get_status().ic_current
        assert ic < 0.0

    def test_random_ic_near_zero(self):
        m = _monitor(window_size=20)
        _feed_noise(m, 20)
        ic = m.get_status().ic_current
        assert -0.6 < ic < 0.6  # random ≈ 0 (loose bound for small n)

    def test_ic_trend_negative_when_decaying(self):
        m = _monitor(window_size=10)
        _feed_correct(m, 10)  # window 1: good
        _feed_wrong(m, 10)    # window 2: bad
        assert m.get_status().ic_trend < 0.0


# ── Pearson correlation helper ─────────────────────────────────────────────────

class TestPearsonCorr:

    def test_identical_returns_one(self):
        m = _monitor()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert m._pearson_corr(x, x) == pytest.approx(1.0)

    def test_opposite_returns_minus_one(self):
        m = _monitor()
        x = [1.0, 2.0, 3.0]
        y = [-1.0, -2.0, -3.0]
        assert m._pearson_corr(x, y) == pytest.approx(-1.0, abs=0.001)

    def test_constant_x_returns_zero(self):
        m = _monitor()
        x = [5.0, 5.0, 5.0, 5.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert m._pearson_corr(x, y) == 0.0

    def test_short_series_returns_zero(self):
        m = _monitor()
        assert m._pearson_corr([1.0], [1.0]) == 0.0
        assert m._pearson_corr([1.0, 2.0], [1.0, 2.0]) == 0.0


# ── Health classifier helper ───────────────────────────────────────────────────

class TestClassifyHealth:

    def test_all_healthy_signals(self):
        m = _monitor()
        h = m._classify_health(_IC_HEALTHY + 0.01, _HIT_RATE_HEALTHY + 0.01, _CONF_HEALTHY + 0.01)
        assert h == "healthy"

    def test_negative_ic_critical(self):
        m = _monitor()
        h = m._classify_health(-0.05, 0.40, 0.45)
        assert h == "critical"

    def test_two_degrading_signals(self):
        m = _monitor()
        # IC marginally degrading, hit rate degrading, conf borderline
        h = m._classify_health(0.015, 0.48, 0.58)
        assert h == "degrading"

    def test_majority_healthy(self):
        m = _monitor()
        # Only hit rate marginal, IC and conf healthy
        h = m._classify_health(0.08, 0.51, 0.65)
        assert h == "healthy"


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:

    def test_state_written_after_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = ModelDriftMonitor(window_size=5, data_dir=Path(tmp))
            _feed_correct(m, 5)
            path = Path(tmp) / "model_drift_monitor.json"
            assert path.exists()

    def test_state_loaded_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            m1 = ModelDriftMonitor(window_size=5, data_dir=Path(tmp))
            _feed_correct(m1, 5)
            ic1 = m1.get_status().ic_current

            m2 = ModelDriftMonitor(window_size=5, data_dir=Path(tmp))
            assert m2.get_status().ic_current == pytest.approx(ic1, abs=0.001)

    def test_no_dir_no_crash(self):
        m = ModelDriftMonitor(data_dir=None)
        _feed_correct(m, 10)
        assert m.get_status() is not None

    def test_report_has_window_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = ModelDriftMonitor(window_size=5, data_dir=Path(tmp))
            _feed_correct(m, 10)
            report = m.get_report()
            assert "window_history" in report
            assert len(report["window_history"]) == 2

    def test_report_contains_required_keys(self):
        m = _monitor()
        report = m.get_report()
        for key in ("health", "should_retrain", "ic_current", "hit_rate_current",
                    "med_confidence", "consecutive_degraded", "total_windows"):
            assert key in report
