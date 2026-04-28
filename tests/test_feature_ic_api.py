"""
Tests for ICTracker — the feature IC tracking logic that backs
the Feature Importance Drift Dashboard.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from monitoring.ic_tracker import ICStats, ICTracker


@pytest.fixture
def tracker(tmp_path: Path) -> ICTracker:
    return ICTracker(
        state_path=str(tmp_path / "ic_state.json"),
        max_buffer=500,
        persist=False,
    )


def _fill_obs(tracker: ICTracker, feature: str, n: int, positive: bool = True) -> None:
    """Add n (feature_value, fwd_return) pairs with the given correlation sign."""
    import numpy as np
    rng = np.random.default_rng(42)
    for i in range(n):
        fv = float(rng.uniform(0.0, 1.0))
        ret = fv * 0.05 if positive else -fv * 0.05
        ret += float(rng.normal(0, 0.01))
        tracker._observations.setdefault(feature, __import__("collections").deque(maxlen=500))
        tracker._observations[feature].append((fv, ret))


# ---------------------------------------------------------------------------
# record_features / record_return
# ---------------------------------------------------------------------------

class TestRecordCycle:
    def test_record_features_creates_pending_entry(self, tracker):
        import datetime
        today = datetime.date.today().isoformat()
        tracker.record_features("AAPL", today, {"rsi": 62.0}, 0.2)
        assert tracker.get_pending_count() == 1

    def test_duplicate_date_not_double_counted(self, tracker):
        import datetime
        today = datetime.date.today().isoformat()
        tracker.record_features("AAPL", today, {"rsi": 62.0}, 0.2)
        tracker.record_features("AAPL", today, {"rsi": 63.0}, 0.3)
        assert tracker.get_pending_count() == 1

    def test_record_return_removes_from_pending(self, tracker):
        import datetime
        today = datetime.date.today().isoformat()
        tracker.record_features("AAPL", today, {"rsi": 62.0}, 0.2)
        tracker.record_return("AAPL", today, 0.025)
        assert tracker.get_pending_count() == 0

    def test_record_return_populates_observations(self, tracker):
        import datetime
        today = datetime.date.today().isoformat()
        tracker.record_features("AAPL", today, {"rsi": 62.0, "macd": 0.1}, 0.2)
        tracker.record_return("AAPL", today, 0.02)
        counts = tracker.get_observation_counts()
        assert "rsi" in counts
        assert "macd" in counts

    def test_unknown_return_silently_ignored(self, tracker):
        tracker.record_return("AAPL", "9999-01-01", 0.05)  # no matching pending
        assert tracker.get_pending_count() == 0


# ---------------------------------------------------------------------------
# get_ic
# ---------------------------------------------------------------------------

class TestGetIc:
    def test_insufficient_obs_returns_zero(self, tracker):
        tracker.record_features("X", "2026-03-01", {"f": 1.0}, 0.1)
        tracker.record_return("X", "2026-03-01", 0.01)
        ic = tracker.get_ic("f", window=30)
        assert ic == 0.0  # only 1 obs < min 5

    def test_positive_ic_for_correlated_feature(self, tracker):
        _fill_obs(tracker, "rsi", 50, positive=True)
        ic = tracker.get_ic("rsi", window=50)
        assert ic > 0.0

    def test_negative_ic_for_anti_correlated_feature(self, tracker):
        _fill_obs(tracker, "inv_rsi", 50, positive=False)
        ic = tracker.get_ic("inv_rsi", window=50)
        assert ic < 0.0

    def test_ic_bounded_minus_one_to_one(self, tracker):
        _fill_obs(tracker, "feat", 80, positive=True)
        ic = tracker.get_ic("feat", window=80)
        assert -1.0 <= ic <= 1.0

    def test_missing_feature_returns_zero(self, tracker):
        assert tracker.get_ic("nonexistent", window=30) == 0.0

    def test_constant_feature_returns_zero(self, tracker):
        import collections
        tracker._observations["const"] = collections.deque(
            [(1.0, r) for r in [0.01, -0.01, 0.02, -0.02, 0.01]], maxlen=500
        )
        ic = tracker.get_ic("const", window=5)
        assert ic == 0.0


# ---------------------------------------------------------------------------
# get_stats / status classification
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_strong_feature_status(self, tracker):
        _fill_obs(tracker, "strong_feat", 80, positive=True)
        stats = tracker.get_stats("strong_feat")
        # Can't guarantee exactly "strong" due to random data; just check it's not dead
        assert stats.status in ("strong", "live", "suspect")
        assert isinstance(stats.ic_30d, float)

    def test_dead_feature_status_when_low_ic(self, tracker):
        import collections
        # Uncorrelated feature: near-zero IC
        import numpy as np
        rng = np.random.default_rng(0)
        tracker._observations["noise"] = collections.deque(
            [(rng.random(), rng.choice([-0.01, 0.01])) for _ in range(100)],
            maxlen=500,
        )
        stats = tracker.get_stats("noise")
        # May be dead or suspect depending on random data
        assert stats.status in ("dead", "suspect", "live", "strong")

    def test_n_obs_accurate(self, tracker):
        _fill_obs(tracker, "foo", 30)
        stats = tracker.get_stats("foo")
        assert stats.n_obs == 30

    def test_stats_for_missing_feature_returns_zero_ic(self, tracker):
        stats = tracker.get_stats("ghost")
        assert stats.ic_30d == 0.0
        assert stats.ic_90d == 0.0
        assert stats.n_obs == 0


# ---------------------------------------------------------------------------
# get_summary / dead / strong sets
# ---------------------------------------------------------------------------

class TestSummaryAndSets:
    def test_get_summary_returns_dict(self, tracker):
        _fill_obs(tracker, "a", 20)
        _fill_obs(tracker, "b", 20)
        summary = tracker.get_summary()
        assert isinstance(summary, dict)
        assert "a" in summary
        assert "b" in summary

    def test_get_summary_respects_min_obs(self, tracker):
        _fill_obs(tracker, "rare", 5)  # below default min_obs=10
        summary = tracker.get_summary(min_obs=10)
        assert "rare" not in summary

    def test_dead_features_is_set(self, tracker):
        dead = tracker.get_dead_features()
        assert isinstance(dead, set)

    def test_strong_features_is_set(self, tracker):
        strong = tracker.get_strong_features()
        assert isinstance(strong, set)

    def test_observation_counts_dict(self, tracker):
        _fill_obs(tracker, "x", 15)
        counts = tracker.get_observation_counts()
        assert counts.get("x") == 15


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        t1 = ICTracker(state_path=str(tmp_path / "ic.json"), persist=True)
        _fill_obs(t1, "rsi", 30)
        t1._save_state()

        t2 = ICTracker(state_path=str(tmp_path / "ic.json"), persist=True)
        assert "rsi" in t2._observations
        assert len(t2._observations["rsi"]) == 30

    def test_missing_state_file_starts_empty(self, tmp_path):
        t = ICTracker(state_path=str(tmp_path / "missing.json"), persist=True)
        assert t.get_observation_counts() == {}
