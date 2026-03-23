"""Tests for SignalAutoTuner."""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from monitoring.signal_auto_tuner import (
    RegimeThresholdState,
    SignalAutoTuner,
)


def _make_report(by_regime: dict) -> dict:
    """Build a minimal EOD digest dict."""
    return {"report_date": date.today().isoformat(), "by_regime": by_regime}


def _regime_data(trades: int, win_rate: float) -> dict:
    return {"trades": trades, "win_rate": win_rate}


class TestRegimeThresholdState:

    def test_clamped_value_stays_in_hard_limits(self):
        s = RegimeThresholdState(regime="bull", base_value=0.17, current_value=0.50)
        assert s.clamped_value <= 0.40

    def test_clamped_value_below_hard_min(self):
        s = RegimeThresholdState(regime="bull", base_value=0.17, current_value=0.05)
        assert s.clamped_value >= 0.10

    def test_clamped_value_bounded_by_max_drift(self):
        # base=0.17, max_drift=0.03 → ceiling = 0.20
        s = RegimeThresholdState(regime="bull", base_value=0.17, current_value=0.25)
        assert s.clamped_value == pytest.approx(0.20, abs=0.001)

    def test_clamped_value_floor_by_max_drift(self):
        # base=0.17, max_drift=0.03 → floor = 0.14
        s = RegimeThresholdState(regime="bull", base_value=0.17, current_value=0.10)
        assert s.clamped_value == pytest.approx(0.14, abs=0.001)


class TestSignalAutoTunerCore:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.tuner = SignalAutoTuner(
            data_dir=self.tmp,
            min_samples=3,
            min_trades_per_regime=3,
            urgency_threshold=0.40,
            strong_threshold=0.62,
            step=0.01,
        )

    def test_insufficient_digests_returns_no_changes(self):
        result = self.tuner.run(eod_reports=[_make_report({})])
        assert len(result.changes) == 0
        assert len(result.insufficient_data) > 0

    def test_low_win_rate_raises_threshold_after_min_samples(self):
        # Each run() call needs ≥ min_samples=3 reports, and consecutive_low increments
        # once per run() call → call 3 times to accumulate 3 consecutive low days.
        regime = "bull"
        batch = [_make_report({regime: _regime_data(trades=10, win_rate=0.30)})] * 3
        result = None
        for _ in range(3):
            result = self.tuner.run(eod_reports=batch)
        changes = [c for c in result.changes if c["regime"] == regime]
        assert len(changes) == 1
        assert changes[0]["direction"] == "raise"
        assert changes[0]["new_value"] > changes[0]["old_value"]

    def test_high_win_rate_lowers_threshold_after_min_samples(self):
        regime = "neutral"
        batch = [_make_report({regime: _regime_data(trades=10, win_rate=0.70)})] * 3
        result = None
        for _ in range(3):
            result = self.tuner.run(eod_reports=batch)
        changes = [c for c in result.changes if c["regime"] == regime]
        assert len(changes) == 1
        assert changes[0]["direction"] == "lower"
        assert changes[0]["new_value"] < changes[0]["old_value"]

    def test_insufficient_trades_skips_regime(self):
        """Total trades across all reports < min_trades_per_regime → insufficient."""
        # Use a tuner with high min_trades_per_regime so 1 trade/report × 3 reports = 3 < 5
        tuner = SignalAutoTuner(
            data_dir=self.tmp, min_samples=3, min_trades_per_regime=5,
            urgency_threshold=0.40, strong_threshold=0.62, step=0.01,
        )
        regime = "bear"
        reports = [_make_report({regime: _regime_data(trades=1, win_rate=0.20)})] * 3
        result = tuner.run(eod_reports=reports)
        assert regime in result.insufficient_data

    def test_acceptable_win_rate_no_change(self):
        """WR in 0.40-0.62 range → no adjustment."""
        regime = "volatile"
        reports = [
            _make_report({regime: _regime_data(trades=10, win_rate=0.52)})
            for _ in range(3)
        ]
        result = self.tuner.run(eod_reports=reports)
        changes = [c for c in result.changes if c["regime"] == regime]
        assert len(changes) == 0

    def test_state_persisted_and_loaded(self):
        """Tuner state should survive a save+load cycle."""
        regime = "bull"
        reports = [
            _make_report({regime: _regime_data(trades=10, win_rate=0.30)})
            for _ in range(3)
        ]
        self.tuner.run(eod_reports=reports)

        tuner2 = SignalAutoTuner(data_dir=self.tmp)
        assert regime in tuner2._state
        assert tuner2._state[regime].consecutive_low >= 1

    def test_output_file_written(self):
        regime = "neutral"
        reports = [
            _make_report({regime: _regime_data(trades=10, win_rate=0.70)})
            for _ in range(3)
        ]
        self.tuner.run(eod_reports=reports)
        output_path = Path(self.tmp) / "auto_tuned_thresholds.json"
        assert output_path.exists()
        with open(output_path) as fh:
            data = json.load(fh)
        assert "thresholds" in data
        assert "generated_at" in data

    def test_get_thresholds_returns_all_regimes(self):
        thresholds = self.tuner.get_thresholds()
        for regime in self.tuner._default_bases:
            assert regime in thresholds

    def test_load_thresholds_from_disk_none_when_no_file(self):
        result = self.tuner.load_thresholds_from_disk()
        assert result is None

    def test_load_thresholds_from_disk_after_run(self):
        regime = "strong_bull"
        reports = [
            _make_report({regime: _regime_data(trades=10, win_rate=0.70)})
            for _ in range(3)
        ]
        self.tuner.run(eod_reports=reports)
        loaded = self.tuner.load_thresholds_from_disk()
        assert loaded is not None
        assert regime in loaded

    def test_reset_regime_restores_base(self):
        regime = "bull"
        state = self.tuner._get_or_init_state(regime, self.tuner._default_bases[regime])
        state.current_value = 0.25
        self.tuner.reset_regime(regime)
        assert self.tuner._state[regime].current_value == self.tuner._default_bases[regime]

    def test_reset_all_restores_base_values(self):
        # Adjust some thresholds, then reset
        for regime in list(self.tuner._default_bases.keys())[:3]:
            s = self.tuner._get_or_init_state(regime, self.tuner._default_bases[regime])
            s.current_value = 0.30
            s.consecutive_low = 5
        self.tuner.reset_all()
        # After reset_all(), _write_output() repopulates state via _get_or_init_state
        # All values must be at base with no streaks
        thresholds = self.tuner.get_thresholds()
        for regime, base in self.tuner._default_bases.items():
            assert thresholds[regime] == pytest.approx(base, abs=0.001)

    def test_max_drift_cap_prevents_runaway_raises(self):
        """After many cycles of low WR, threshold cannot exceed base + 0.03."""
        regime = "bull"
        reports = [
            _make_report({regime: _regime_data(trades=10, win_rate=0.20)})
            for _ in range(3)
        ]
        for _ in range(20):   # run many times
            self.tuner.run(eod_reports=reports)
        thresh = self.tuner.get_thresholds()[regime]
        base = self.tuner._default_bases[regime]
        assert thresh <= base + 0.03 + 1e-6

    def test_aggregate_win_rates_multiple_reports(self):
        """Aggregation correctly pools wins across reports."""
        reports = [
            _make_report({"bull": {"trades": 4, "win_rate": 0.50}}),  # 2 wins
            _make_report({"bull": {"trades": 6, "win_rate": 0.50}}),  # 3 wins
        ]
        agg = self.tuner._aggregate_regime_win_rates(reports)
        assert "bull" in agg
        assert agg["bull"]["total_trades"] == 10
        assert agg["bull"]["win_rate"] == pytest.approx(0.50, abs=0.01)

    def test_partial_revert_when_wr_recovers(self):
        """If threshold was raised (total_adj > step) and WR recovers, gentle revert."""
        state = self.tuner._get_or_init_state("bear", self.tuner._default_bases["bear"])
        state.total_adjustment = 0.02   # previously raised
        state.consecutive_low = 0
        direction, reason = self.tuner._evaluate(state, win_rate=0.52)  # WR in acceptable range
        assert direction == -1   # revert
        assert "revert" in reason
