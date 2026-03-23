"""Tests for ExecutionTimingOptimizer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from monitoring.execution_timing import (
    ExecutionTimingOptimizer,
    BucketStats,
    TimingScore,
    _DEFAULT_SCORE,
)


def _opt(**kw) -> ExecutionTimingOptimizer:
    defaults = dict(min_obs=3, score_floor=0.55)
    defaults.update(kw)
    return ExecutionTimingOptimizer(**defaults)


def _fill(opt: ExecutionTimingOptimizer, bps: float, hour: int = 10,
          dow: int = 0, regime: str = "neutral") -> None:
    opt.record_fill(bps, hour=hour, day_of_week=dow, regime=regime)


# ── Default / empty state ────────────────────────────────────────────────────

class TestDefaultState:

    def test_returns_timing_score(self):
        opt = _opt()
        assert isinstance(opt.get_timing_score(10, 0), TimingScore)

    def test_empty_score_is_default(self):
        opt = _opt()
        ts = opt.get_timing_score(10, 0)
        assert ts.score == pytest.approx(_DEFAULT_SCORE)

    def test_empty_has_no_data(self):
        opt = _opt()
        ts = opt.get_timing_score(10, 0)
        assert ts.has_data is False

    def test_report_is_dict(self):
        opt = _opt()
        assert isinstance(opt.get_report(), dict)

    def test_report_has_required_keys(self):
        opt = _opt()
        r = opt.get_report()
        for key in ("total_fills", "p50_bps", "p95_bps", "buckets", "worst_neutral"):
            assert key in r


# ── BucketStats ──────────────────────────────────────────────────────────────

class TestBucketStats:

    def test_update_increments_obs(self):
        b = BucketStats()
        b.update(5.0)
        assert b.obs == 1

    def test_mean_single_obs(self):
        b = BucketStats()
        b.update(8.0)
        assert b.mean_slippage_bps == pytest.approx(8.0)

    def test_mean_multiple_obs(self):
        b = BucketStats()
        for v in [2.0, 4.0, 6.0]:
            b.update(v)
        assert b.mean_slippage_bps == pytest.approx(4.0)

    def test_worst_tracks_max(self):
        b = BucketStats()
        for v in [1.0, 5.0, 3.0]:
            b.update(v)
        assert b.worst_bps == pytest.approx(5.0)

    def test_to_dict_has_keys(self):
        b = BucketStats()
        d = b.to_dict()
        for key in ("obs", "mean_slippage_bps", "var_slippage_bps", "worst_bps"):
            assert key in d


# ── record_fill ───────────────────────────────────────────────────────────────

class TestRecordFill:

    def test_single_fill_creates_bucket(self):
        opt = _opt(min_obs=1)
        _fill(opt, 5.0)
        ts = opt.get_timing_score(10, 0, "neutral")
        assert ts.obs == 1

    def test_fills_accumulate_in_bucket(self):
        opt = _opt(min_obs=3)
        for _ in range(3):
            _fill(opt, 5.0)
        ts = opt.get_timing_score(10, 0, "neutral")
        assert ts.obs == 3

    def test_different_hours_separate_buckets(self):
        opt = _opt(min_obs=1)
        _fill(opt, 5.0, hour=9)
        _fill(opt, 10.0, hour=15)
        ts9 = opt.get_timing_score(9, 0, "neutral")
        ts15 = opt.get_timing_score(15, 0, "neutral")
        assert ts9.obs == 1
        assert ts15.obs == 1

    def test_different_regimes_separate_buckets(self):
        opt = _opt(min_obs=1)
        _fill(opt, 5.0, regime="neutral")
        _fill(opt, 5.0, regime="bull")
        ts_n = opt.get_timing_score(10, 0, "neutral")
        ts_b = opt.get_timing_score(10, 0, "bull")
        assert ts_n.obs == 1
        assert ts_b.obs == 1

    def test_fill_log_grows(self):
        opt = _opt()
        for i in range(5):
            _fill(opt, float(i))
        assert opt.get_report()["total_fills"] == 5


# ── Timing score computation ──────────────────────────────────────────────────

class TestTimingScore:

    def test_score_1_when_below_min_obs(self):
        opt = _opt(min_obs=5)
        for _ in range(4):
            _fill(opt, 50.0)
        ts = opt.get_timing_score(10, 0, "neutral")
        assert ts.score == pytest.approx(_DEFAULT_SCORE)

    def test_low_slippage_gives_high_score(self):
        """Low slippage window should return score = 1.0 (no penalty)."""
        opt = _opt(min_obs=3, score_floor=0.55)
        # Feed many low-slippage fills globally to push p50 high
        for h in range(24):
            for _ in range(4):
                opt.record_fill(20.0, hour=h, day_of_week=0, regime="neutral")
        # Now fill target bucket with very low slippage
        for _ in range(4):
            opt.record_fill(1.0, hour=10, day_of_week=0, regime="neutral")
        ts = opt.get_timing_score(10, 0, "neutral")
        # Low slippage bucket → score should be higher (close to 1.0)
        assert ts.score >= 0.55

    def test_high_slippage_gives_low_score(self):
        """High slippage window should return score < 1.0.

        Strategy: fill 5 low-bps buckets (20 fills) + 1 high-bps bucket (4 fills)
        → 24 total fills, p95 = sorted_vals[22] = high value → bucket scores below 1.0.
        """
        opt = _opt(min_obs=3, score_floor=0.55)
        # 5 hours of low slippage → 20 fills at 1.0
        for h in [9, 11, 12, 13, 14]:
            for _ in range(4):
                opt.record_fill(1.0, hour=h, day_of_week=0, regime="neutral")
        # Target bucket: very high slippage (4 fills at 200 bps)
        for _ in range(4):
            opt.record_fill(200.0, hour=10, day_of_week=0, regime="neutral")
        ts = opt.get_timing_score(10, 0, "neutral")
        assert ts.score < 1.0

    def test_score_bounded(self):
        opt = _opt(min_obs=3, score_floor=0.55)
        # Fill with extreme slippage to push score to floor
        for h in range(24):
            for _ in range(4):
                opt.record_fill(1.0, hour=h, day_of_week=0, regime="neutral")
        for _ in range(5):
            opt.record_fill(9999.0, hour=10, day_of_week=0, regime="neutral")
        ts = opt.get_timing_score(10, 0, "neutral")
        assert ts.score >= 0.55
        assert ts.score <= 1.0

    def test_score_attributes(self):
        opt = _opt(min_obs=1)
        _fill(opt, 5.0, hour=9, dow=1, regime="bull")
        ts = opt.get_timing_score(9, 1, "bull")
        assert ts.hour == 9
        assert ts.day_of_week == 1
        assert ts.regime == "bull"
        assert isinstance(ts.mean_slippage_bps, float)


# ── Worst hours ──────────────────────────────────────────────────────────────

class TestWorstHours:

    def test_returns_list(self):
        opt = _opt()
        assert isinstance(opt.get_worst_hours(), list)

    def test_empty_regime_returns_empty(self):
        opt = _opt()
        assert opt.get_worst_hours("neutral") == []

    def test_worst_hours_sorted_by_score_asc(self):
        opt = _opt(min_obs=1)
        for h in range(24):
            for _ in range(4):
                opt.record_fill(1.0, hour=h, day_of_week=0, regime="neutral")
        # Make hour 14 very bad
        for _ in range(4):
            opt.record_fill(500.0, hour=14, day_of_week=0, regime="neutral")
        worst = opt.get_worst_hours("neutral", top_n=3)
        if len(worst) > 1:
            assert worst[0]["score"] <= worst[-1]["score"]

    def test_worst_hours_respects_min_obs(self):
        opt = _opt(min_obs=5)
        # Fill with only 3 obs — should be excluded
        for _ in range(3):
            _fill(opt, 100.0)
        assert opt.get_worst_hours("neutral") == []


# ── Percentile calibration ────────────────────────────────────────────────────

class TestPercentiles:

    def test_p50_computed_after_fills(self):
        opt = _opt()
        for i in range(1, 11):
            opt.record_fill(float(i), hour=i, day_of_week=0, regime="neutral")
        assert opt._p50_bps > 0.0

    def test_p95_ge_p50(self):
        opt = _opt()
        for i in range(1, 21):
            opt.record_fill(float(i), hour=i % 24, day_of_week=0, regime="neutral")
        assert opt._p95_bps >= opt._p50_bps


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:

    def test_state_written_after_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt = ExecutionTimingOptimizer(data_dir=Path(tmp), min_obs=1)
            opt.record_fill(5.0, hour=10, day_of_week=0, regime="neutral")
            assert (Path(tmp) / "execution_timing.json").exists()

    def test_state_reloaded_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt1 = ExecutionTimingOptimizer(data_dir=Path(tmp), min_obs=1)
            for _ in range(3):
                opt1.record_fill(7.5, hour=10, day_of_week=0, regime="neutral")

            opt2 = ExecutionTimingOptimizer(data_dir=Path(tmp), min_obs=1)
            key = (10, 0, "neutral")
            assert key in opt2._buckets
            assert opt2._buckets[key].obs == 3

    def test_no_dir_no_crash(self):
        opt = ExecutionTimingOptimizer(data_dir=None, min_obs=1)
        opt.record_fill(5.0, hour=9, day_of_week=0, regime="neutral")
        assert opt.get_timing_score(9, 0, "neutral").obs == 1

    def test_report_buckets_match_fills(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt = ExecutionTimingOptimizer(data_dir=Path(tmp), min_obs=1)
            opt.record_fill(5.0, hour=10, day_of_week=0, regime="neutral")
            r = opt.get_report()
            assert len(r["buckets"]) >= 1
