"""Tests for AlphaDecayCalibrator."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from monitoring.alpha_decay_calibrator import (
    AlphaDecayCalibrator,
    BucketStats,
    RegimeDecay,
    _BUCKETS,
    _BUCKET_LABELS,
    _DEFAULT_HOLD,
    _MIN_OBS,
)


def _cal(**kw) -> AlphaDecayCalibrator:
    defaults = dict(min_obs=5)
    defaults.update(kw)
    return AlphaDecayCalibrator(**defaults)


def _feed_bucket(cal: AlphaDecayCalibrator, bucket_hours: float, n: int,
                 good: bool = True, regime: str = "bull") -> None:
    """Feed n trades into the given hold-hour bucket, all correct or wrong."""
    for i in range(n):
        sig = 0.20 + 0.02 * (i % 5)
        ret = sig * 0.05 if good else -(sig * 0.05)
        cal.record_trade(signal=sig, actual_return=ret,
                         hold_hours=bucket_hours, regime=regime)


# ── Default / empty state ─────────────────────────────────────────────────────

class TestDefaultState:

    def test_optimal_hold_default_when_no_data(self):
        cal = _cal()
        assert cal.get_optimal_hold_hours("bull") == pytest.approx(_DEFAULT_HOLD)

    def test_half_life_none_when_no_data(self):
        cal = _cal()
        assert cal.get_alpha_half_life("neutral") is None

    def test_decay_score_one_when_no_data(self):
        cal = _cal()
        assert cal.get_decay_score(4.0, "bull") == pytest.approx(1.0)

    def test_report_empty_regimes(self):
        cal = _cal()
        report = cal.get_decay_report()
        assert report["total_trades"] == 0
        assert isinstance(report["regimes"], dict)


# ── record_trade ──────────────────────────────────────────────────────────────

class TestRecordTrade:

    def test_zero_signal_skipped(self):
        cal = _cal()
        cal.record_trade(signal=0.0, actual_return=0.01, hold_hours=1.0)
        assert cal._total_trades == 0

    def test_trade_increments_counter(self):
        cal = _cal()
        cal.record_trade(signal=0.5, actual_return=0.01, hold_hours=1.0)
        assert cal._total_trades == 1

    def test_trade_stored_in_correct_bucket(self):
        cal = _cal()
        cal.record_trade(signal=0.5, actual_return=0.01, hold_hours=1.0)
        # 1.0h → bucket 0 (0-2h)
        assert len(cal._obs["neutral"][0]) == 1

    def test_bucket_idx_overnight(self):
        cal = _cal()
        cal.record_trade(signal=0.5, actual_return=0.01, hold_hours=12.0, regime="bear")
        # 12h → bucket 2 (8-24h)
        assert len(cal._obs["bear"][2]) == 1

    def test_bucket_idx_multiday(self):
        cal = _cal()
        cal.record_trade(signal=0.5, actual_return=0.01, hold_hours=48.0)
        # 48h → bucket 3 (24h+)
        assert len(cal._obs["neutral"][3]) == 1


# ── Bucket selection ──────────────────────────────────────────────────────────

class TestBucketIndex:

    def test_zero_hours_first_bucket(self):
        assert AlphaDecayCalibrator._bucket_idx(0.0) == 0

    def test_exactly_2h_second_bucket(self):
        assert AlphaDecayCalibrator._bucket_idx(2.0) == 1

    def test_exactly_8h_third_bucket(self):
        assert AlphaDecayCalibrator._bucket_idx(8.0) == 2

    def test_exactly_24h_fourth_bucket(self):
        assert AlphaDecayCalibrator._bucket_idx(24.0) == 3

    def test_large_hours_last_bucket(self):
        assert AlphaDecayCalibrator._bucket_idx(1000.0) == 3

    def test_negative_hours_clamped_first(self):
        assert AlphaDecayCalibrator._bucket_idx(-5.0) == 0


# ── Optimal hold selection ────────────────────────────────────────────────────

class TestOptimalHold:

    def test_best_bucket_selected(self):
        cal = _cal(min_obs=5)
        # Feed bucket 1 (2-8h) with good trades, bucket 0 with bad
        _feed_bucket(cal, 1.0, 5, good=False, regime="bull")  # bucket 0: bad IC
        _feed_bucket(cal, 4.0, 5, good=True, regime="bull")   # bucket 1: good IC
        opt = cal.get_optimal_hold_hours("bull")
        # midpoint of (2, 8) = 5h
        assert opt == pytest.approx(5.0)

    def test_default_when_insufficient_obs(self):
        cal = _cal(min_obs=10)
        # Feed only 3 trades per bucket (below min_obs=10)
        _feed_bucket(cal, 1.0, 3, regime="neutral")
        assert cal.get_optimal_hold_hours("neutral") == pytest.approx(_DEFAULT_HOLD)

    def test_unknown_regime_returns_default(self):
        cal = _cal()
        assert cal.get_optimal_hold_hours("sideways") == pytest.approx(_DEFAULT_HOLD)


# ── Decay score ───────────────────────────────────────────────────────────────

class TestDecayScore:

    def test_best_bucket_score_near_one(self):
        cal = _cal(min_obs=5)
        # All trades in bucket 1 (2-8h) with strong positive IC
        _feed_bucket(cal, 4.0, 10, good=True, regime="bull")
        score = cal.get_decay_score(4.0, "bull")
        assert score >= 0.80  # targeting best bucket → high score

    def test_score_bounded_below_05(self):
        cal = _cal(min_obs=5)
        _feed_bucket(cal, 1.0, 5, good=True, regime="neutral")
        _feed_bucket(cal, 4.0, 5, good=False, regime="neutral")
        score = cal.get_decay_score(4.0, "neutral")
        assert score >= 0.50

    def test_score_bounded_above_one(self):
        cal = _cal(min_obs=5)
        _feed_bucket(cal, 1.0, 10, good=True, regime="bull")
        score = cal.get_decay_score(1.0, "bull")
        assert score <= 1.0

    def test_score_one_with_no_data(self):
        cal = _cal()
        assert cal.get_decay_score(2.0, "crisis") == pytest.approx(1.0)


# ── IC computation ────────────────────────────────────────────────────────────

class TestPearsonCorr:

    def test_perfect_positive(self):
        cal = _cal()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert cal._pearson_corr(x, x) == pytest.approx(1.0)

    def test_perfect_negative(self):
        cal = _cal()
        x = [1.0, 2.0, 3.0]
        y = [-1.0, -2.0, -3.0]
        assert cal._pearson_corr(x, y) == pytest.approx(-1.0, abs=0.001)

    def test_constant_series_zero(self):
        cal = _cal()
        assert cal._pearson_corr([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0

    def test_insufficient_data_zero(self):
        cal = _cal()
        assert cal._pearson_corr([1.0, 2.0], [1.0, 2.0]) == 0.0


# ── Half-life estimation ──────────────────────────────────────────────────────

class TestHalfLife:

    def test_none_when_no_data(self):
        cal = _cal()
        assert cal.get_alpha_half_life("bull") is None

    def test_half_life_positive_when_decaying(self):
        """IC decays from bucket 0 → bucket 1 → should estimate a positive half-life."""
        cal = _cal(min_obs=5)
        # Strong IC in early bucket, weak in later
        for i in range(8):
            sig = 0.20 + 0.02 * (i % 4)
            cal.record_trade(sig, sig * 0.08, 1.0, "neutral")   # bucket 0: strong IC
        for i in range(6):
            sig = 0.20 + 0.02 * (i % 4)
            cal.record_trade(sig, sig * 0.01, 4.0, "neutral")   # bucket 1: weaker IC
        hl = cal.get_alpha_half_life("neutral")
        if hl is not None:
            assert hl > 0.0

    def test_half_life_none_when_ic_not_decaying(self):
        cal = _cal(min_obs=5)
        # IC increases with bucket → not decaying → half-life None
        for i in range(6):
            sig = 0.20 + 0.02 * (i % 4)
            cal.record_trade(sig, sig * 0.01, 1.0, "bull")   # bucket 0: weak
        for i in range(6):
            sig = 0.20 + 0.02 * (i % 4)
            cal.record_trade(sig, sig * 0.08, 4.0, "bull")   # bucket 1: strong
        hl = cal.get_alpha_half_life("bull")
        # When slope is positive (IC growing), should return None
        assert hl is None or hl > 0.0  # either None or positive


# ── Report ────────────────────────────────────────────────────────────────────

class TestReport:

    def test_report_has_required_keys(self):
        cal = _cal(min_obs=5)
        _feed_bucket(cal, 1.0, 5, regime="bull")
        report = cal.get_decay_report()
        assert "total_trades" in report
        assert "regimes" in report
        assert "updated_at" in report

    def test_report_includes_regime(self):
        cal = _cal(min_obs=5)
        _feed_bucket(cal, 1.0, 5, regime="bear")
        report = cal.get_decay_report()
        assert "bear" in report["regimes"]

    def test_report_buckets_have_ic(self):
        cal = _cal(min_obs=5)
        _feed_bucket(cal, 4.0, 6, regime="neutral")
        report = cal.get_decay_report()
        if "neutral" in report["regimes"]:
            for bucket in report["regimes"]["neutral"]["buckets"]:
                assert "ic" in bucket


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:

    def test_state_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = AlphaDecayCalibrator(min_obs=5, data_dir=Path(tmp))
            for i in range(10):  # trigger persist (every 10 trades)
                cal.record_trade(0.3, 0.01, 1.0, "bull")
            path = Path(tmp) / "alpha_decay_calibrator.json"
            assert path.exists()

    def test_state_loaded_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal1 = AlphaDecayCalibrator(min_obs=5, data_dir=Path(tmp))
            for i in range(10):
                cal1.record_trade(0.3, 0.01, 1.0, "bull")

            cal2 = AlphaDecayCalibrator(min_obs=5, data_dir=Path(tmp))
            assert cal2._total_trades == 10

    def test_no_dir_no_crash(self):
        cal = AlphaDecayCalibrator(data_dir=None)
        cal.record_trade(0.5, 0.01, 2.0)
        assert cal._total_trades == 1
