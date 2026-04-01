"""tests/test_signal_drift_monitor.py — Signal drift detection tests."""
from __future__ import annotations

import pytest

from monitoring.signal_drift_monitor import (
    SignalDriftMonitor,
    DriftState,
    _SHORT_WINDOW,
    _LONG_WINDOW,
    _DRIFT_THRESHOLD,
    _MIN_TRADES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _monitor(**kwargs) -> SignalDriftMonitor:
    defaults = dict(
        short_window=10,
        long_window=30,
        drift_threshold=0.15,
        min_trades=5,
        recover_tolerance=0.04,
    )
    defaults.update(kwargs)
    return SignalDriftMonitor(**defaults)


def _record_n(monitor: SignalDriftMonitor, n: int, correct: bool) -> None:
    for _ in range(n):
        monitor.record_outcome("SYM", correct)


# ── Initial state ──────────────────────────────────────────────────────────────

class TestInitialState:
    def test_not_drifting_initially(self):
        m = _monitor()
        assert not m.is_drifting()

    def test_total_zero_initially(self):
        m = _monitor()
        assert m._total == 0

    def test_get_state_no_drift_initially(self):
        m = _monitor()
        s = m.get_state()
        assert not s.is_drifting
        assert s.total_trades == 0

    def test_rolling_win_rate_zero_when_empty(self):
        m = _monitor()
        assert m._rolling_win_rate() == 0.0

    def test_baseline_win_rate_zero_when_empty(self):
        m = _monitor()
        assert m._baseline_win_rate() == 0.0


# ── record_outcome ─────────────────────────────────────────────────────────────

class TestRecordOutcome:
    def test_total_increments(self):
        m = _monitor()
        m.record_outcome("AAPL", True)
        assert m._total == 1

    def test_multiple_records(self):
        m = _monitor()
        for _ in range(7):
            m.record_outcome("AAPL", True)
        assert m._total == 7

    def test_outcomes_stored(self):
        m = _monitor()
        m.record_outcome("X", True)
        m.record_outcome("Y", False)
        assert list(m._outcomes) == [True, False]


# ── Win-rate computation ───────────────────────────────────────────────────────

class TestWinRate:
    def test_all_correct(self):
        m = _monitor(short_window=5)
        _record_n(m, 5, True)
        assert abs(m._rolling_win_rate() - 1.0) < 1e-9

    def test_all_wrong(self):
        m = _monitor(short_window=5)
        _record_n(m, 5, False)
        assert abs(m._rolling_win_rate() - 0.0) < 1e-9

    def test_half_correct(self):
        m = _monitor(short_window=10)
        _record_n(m, 5, True)
        _record_n(m, 5, False)
        assert abs(m._rolling_win_rate() - 0.5) < 1e-9

    def test_rolling_uses_only_short_window(self):
        """Old outcomes outside short_window don't affect rolling rate."""
        m = _monitor(short_window=5, long_window=20)
        _record_n(m, 10, True)  # old - all correct
        _record_n(m, 5, False)  # new short window - all wrong
        assert m._rolling_win_rate() == 0.0

    def test_baseline_uses_full_buffer(self):
        m = _monitor(short_window=5, long_window=20)
        _record_n(m, 15, True)   # 15 correct
        _record_n(m, 5, False)   # 5 wrong
        # baseline = 15/20 = 0.75
        assert abs(m._baseline_win_rate() - 0.75) < 0.01


# ── Drift detection ────────────────────────────────────────────────────────────

class TestDriftDetection:
    def test_no_drift_before_min_trades(self):
        m = _monitor(min_trades=10, drift_threshold=0.10)
        _record_n(m, 5, False)  # only 5 trades, below min
        assert not m.is_drifting()

    def test_drift_triggers_on_accuracy_drop(self):
        m = _monitor(short_window=10, long_window=30, drift_threshold=0.15, min_trades=5)
        _record_n(m, 20, True)   # baseline: 100% correct
        _record_n(m, 10, False)  # rolling: 0% correct → drop = 1.0 > 0.15
        assert m.is_drifting()

    def test_no_drift_when_drop_below_threshold(self):
        m = _monitor(short_window=10, long_window=30, drift_threshold=0.30, min_trades=5)
        _record_n(m, 20, True)
        # Give 8/10 correct in rolling window: baseline≈1.0, rolling=0.8, drop=0.2 < 0.30
        _record_n(m, 8, True)
        _record_n(m, 2, False)
        assert not m.is_drifting()

    def test_drift_since_set_when_drifting(self):
        m = _monitor(short_window=5, long_window=20, drift_threshold=0.15, min_trades=5)
        _record_n(m, 15, True)
        _record_n(m, 5, False)
        assert m._drift_since is not None

    def test_drift_clears_on_recovery(self):
        m = _monitor(short_window=5, long_window=20, drift_threshold=0.15, min_trades=5, recover_tolerance=0.04)
        _record_n(m, 15, True)
        _record_n(m, 5, False)
        assert m.is_drifting()
        # Recover: add many correct trades to bring rolling back up
        _record_n(m, 10, True)
        assert not m.is_drifting()

    def test_drift_since_cleared_on_recovery(self):
        m = _monitor(short_window=5, long_window=20, drift_threshold=0.15, min_trades=5, recover_tolerance=0.04)
        _record_n(m, 15, True)
        _record_n(m, 5, False)
        _record_n(m, 10, True)
        assert m._drift_since is None

    def test_ring_buffer_capped_at_long_window(self):
        m = _monitor(long_window=10)
        _record_n(m, 50, True)
        assert len(m._outcomes) == 10


# ── DriftState dataclass ───────────────────────────────────────────────────────

class TestDriftState:
    def test_to_dict_has_all_keys(self):
        m = _monitor()
        s = m.get_state()
        d = s.to_dict()
        for key in (
            "is_drifting", "rolling_win_rate", "baseline_win_rate",
            "accuracy_drop", "total_trades", "drift_since", "last_evaluated",
        ):
            assert key in d, f"missing key: {key}"

    def test_accuracy_drop_non_negative(self):
        m = _monitor()
        _record_n(m, 5, True)
        s = m.get_state()
        assert s.accuracy_drop >= 0.0

    def test_total_trades_correct(self):
        m = _monitor()
        _record_n(m, 8, True)
        s = m.get_state()
        assert s.total_trades == 8


# ── Reset ──────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_outcomes(self):
        m = _monitor()
        _record_n(m, 10, True)
        m.reset()
        assert len(m._outcomes) == 0

    def test_reset_clears_drift(self):
        m = _monitor(short_window=5, long_window=20, drift_threshold=0.15, min_trades=5)
        _record_n(m, 15, True)
        _record_n(m, 5, False)
        assert m.is_drifting()
        m.reset()
        assert not m.is_drifting()

    def test_reset_zeroes_total(self):
        m = _monitor()
        _record_n(m, 5, True)
        m.reset()
        assert m._total == 0
