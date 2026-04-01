"""
tests/test_adaptive_weight_manager.py — Unit tests for AdaptiveWeightManager
"""
import math
import tempfile
from pathlib import Path

import pytest
from monitoring.adaptive_weight_manager import (
    AdaptiveWeightManager,
    _ic_to_multiplier,
    _BASE_WEIGHTS,
)


# ── _ic_to_multiplier ────────────────────────────────────────────────────────

class TestIcToMultiplier:
    def test_zero_ic_gives_one(self):
        assert _ic_to_multiplier(0.0, k=4.0) == pytest.approx(1.0, abs=1e-6)

    def test_positive_ic_gives_multiplier_above_one(self):
        assert _ic_to_multiplier(0.20, k=4.0) > 1.0

    def test_negative_ic_gives_multiplier_below_one(self):
        assert _ic_to_multiplier(-0.20, k=4.0) < 1.0

    def test_output_bounded_in_zero_two(self):
        for ic in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            m = _ic_to_multiplier(ic, k=4.0)
            assert 0.0 < m < 2.0 + 1e-9

    def test_higher_k_amplifies_effect(self):
        m_low_k  = _ic_to_multiplier(0.20, k=2.0)
        m_high_k = _ic_to_multiplier(0.20, k=8.0)
        assert m_high_k > m_low_k


# ── AdaptiveWeightManager ─────────────────────────────────────────────────────

def _make_mgr() -> AdaptiveWeightManager:
    tmp = tempfile.mkdtemp()
    return AdaptiveWeightManager(persist_path=str(Path(tmp) / "weights.json"))


class MockICResult:
    def __init__(self, name, ic, obs=20, reliable=True):
        self.signal_name = name
        self.ic = ic
        self.obs = obs
        self.is_reliable = reliable
        self.status = "active" if reliable and abs(ic) > 0.05 else "unreliable"


class MockICReport:
    def __init__(self, results):
        self.signals = results
        self.top_factors = [r.signal_name for r in results if r.is_reliable and r.ic > 0.05]
        self.weak_factors = []


class MockTracker:
    def __init__(self, results):
        self._results = results

    def get_report(self):
        return MockICReport(self._results)


class TestAdaptiveWeightManager:

    def test_initial_weights_match_base(self):
        mgr = _make_mgr()
        for name, base in _BASE_WEIGHTS.items():
            if name == "primary_signal":
                continue
            assert mgr.get_weight(name, base) == pytest.approx(base, abs=1e-9)

    def test_get_weight_returns_default_when_disabled(self, monkeypatch):
        import monitoring.adaptive_weight_manager as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "ADAPTIVE_WEIGHTS_ENABLED" else mod._DEF.get(k))
        mgr = _make_mgr()
        assert mgr.get_weight("god_level", 0.99) == pytest.approx(0.99)

    def test_high_ic_increases_weight(self):
        mgr = _make_mgr()
        tracker = MockTracker([MockICResult("god_level", ic=0.35)])
        mgr.update_from_tracker(tracker)
        base = _BASE_WEIGHTS["god_level"]
        assert mgr.get_weight("god_level", base) > base

    def test_negative_ic_decreases_weight(self):
        mgr = _make_mgr()
        tracker = MockTracker([MockICResult("god_level", ic=-0.30)])
        mgr.update_from_tracker(tracker)
        base = _BASE_WEIGHTS["god_level"]
        assert mgr.get_weight("god_level", base) < base

    def test_zero_ic_keeps_weight_near_base(self):
        mgr = _make_mgr()
        tracker = MockTracker([MockICResult("god_level", ic=0.0)])
        mgr.update_from_tracker(tracker)
        base = _BASE_WEIGHTS["god_level"]
        assert mgr.get_weight("god_level", base) == pytest.approx(base, abs=0.001)

    def test_weight_bounded_by_min_mult(self):
        mgr = _make_mgr()
        tracker = MockTracker([MockICResult("god_level", ic=-1.0)])
        for _ in range(20):  # run many times to push against floor
            mgr.update_from_tracker(tracker)
        base = _BASE_WEIGHTS["god_level"]
        floor = base * 0.30
        assert mgr.get_weight("god_level", base) >= floor - 1e-6

    def test_weight_bounded_by_max_mult(self):
        mgr = _make_mgr()
        tracker = MockTracker([MockICResult("god_level", ic=1.0)])
        for _ in range(20):  # run many times to push against ceiling
            mgr.update_from_tracker(tracker)
        base = _BASE_WEIGHTS["god_level"]
        ceil = base * 2.50
        assert mgr.get_weight("god_level", base) <= ceil + 1e-6

    def test_ema_smoothing_is_gradual(self):
        mgr = _make_mgr()
        base = _BASE_WEIGHTS["god_level"]
        tracker = MockTracker([MockICResult("god_level", ic=0.50)])
        mgr.update_from_tracker(tracker)
        w1 = mgr.get_weight("god_level", base)
        # Single update — should be between base and target, not jump all the way
        target = base * _ic_to_multiplier(0.50, 4.0)
        assert base < w1 < target + 1e-6

    def test_unreliable_ic_not_used(self):
        mgr = _make_mgr()
        base = _BASE_WEIGHTS["god_level"]
        tracker = MockTracker([MockICResult("god_level", ic=0.50, reliable=False)])
        updated = mgr.update_from_tracker(tracker)
        assert updated is False
        assert mgr.get_weight("god_level", base) == pytest.approx(base)

    def test_none_tracker_returns_false(self):
        mgr = _make_mgr()
        assert mgr.update_from_tracker(None) is False

    def test_empty_report_returns_false(self):
        mgr = _make_mgr()
        tracker = MockTracker([])
        assert mgr.update_from_tracker(tracker) is False

    def test_maybe_update_fires_on_interval(self):
        mgr = _make_mgr()
        base = _BASE_WEIGHTS["god_level"]
        tracker = MockTracker([MockICResult("god_level", ic=0.40)])
        # cycle=100 → interval=100 → should fire
        updated = mgr.maybe_update(tracker, cycle=100)
        assert updated is True
        assert mgr.get_weight("god_level", base) != pytest.approx(base)

    def test_maybe_update_skips_off_interval(self):
        mgr = _make_mgr()
        base = _BASE_WEIGHTS["god_level"]
        tracker = MockTracker([MockICResult("god_level", ic=0.40)])
        updated = mgr.maybe_update(tracker, cycle=101)
        assert updated is False

    def test_persistence_round_trip(self):
        tmp = tempfile.mkdtemp()
        path = str(Path(tmp) / "weights.json")
        m1 = AdaptiveWeightManager(persist_path=path)
        tracker = MockTracker([MockICResult("god_level", ic=0.40)])
        m1.update_from_tracker(tracker)
        w1 = m1.get_weight("god_level", 0.12)

        m2 = AdaptiveWeightManager(persist_path=path)
        assert m2.get_weight("god_level", 0.12) == pytest.approx(w1)

    def test_get_report_has_expected_keys(self):
        mgr = _make_mgr()
        report = mgr.get_report()
        assert "weights" in report
        assert "last_updated" in report
        assert "enabled" in report
        for name in ("god_level", "mean_reversion", "sector_rotation"):
            assert name in report["weights"]
            assert "current" in report["weights"][name]
            assert "base" in report["weights"][name]
            assert "mult" in report["weights"][name]

    def test_multiple_signals_updated_independently(self):
        mgr = _make_mgr()
        tracker = MockTracker([
            MockICResult("god_level", ic=0.40),
            MockICResult("mean_reversion", ic=-0.20),
        ])
        mgr.update_from_tracker(tracker)
        gl_base = _BASE_WEIGHTS["god_level"]
        mr_base = _BASE_WEIGHTS["mean_reversion"]
        assert mgr.get_weight("god_level", gl_base) > gl_base
        assert mgr.get_weight("mean_reversion", mr_base) < mr_base

    def test_unknown_signal_returns_default(self):
        mgr = _make_mgr()
        assert mgr.get_weight("nonexistent_signal", 0.42) == pytest.approx(0.42)

    def test_last_updated_set_after_update(self):
        mgr = _make_mgr()
        assert mgr._last_update_ts == ""
        tracker = MockTracker([MockICResult("god_level", ic=0.30)])
        mgr.update_from_tracker(tracker)
        assert mgr._last_update_ts != ""
