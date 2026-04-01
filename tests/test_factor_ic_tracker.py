"""
tests/test_factor_ic_tracker.py — Unit tests for monitoring/factor_ic_tracker.py
"""
import tempfile
from pathlib import Path

import pytest
from monitoring.factor_ic_tracker import FactorICTracker, _spearman


# ── _spearman ─────────────────────────────────────────────────────────────────

class TestSpearman:
    def test_perfect_positive_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _spearman(x, y) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _spearman(x, y) == pytest.approx(-1.0)

    def test_zero_correlation_for_constant(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 2.0, 2.0, 2.0, 2.0]
        assert _spearman(x, y) == pytest.approx(0.0)

    def test_insufficient_data_returns_zero(self):
        assert _spearman([1.0], [1.0]) == pytest.approx(0.0)
        assert _spearman([1.0, 2.0], [2.0, 3.0]) == pytest.approx(0.0)

    def test_clipped_to_unit_range(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        val = _spearman(x, y)
        assert -1.0 <= val <= 1.0


# ── FactorICTracker ───────────────────────────────────────────────────────────

def _make_tracker() -> FactorICTracker:
    tmp = tempfile.mkdtemp()
    return FactorICTracker(persist_path=str(Path(tmp) / "factor_ic.json"))


class TestFactorICTracker:

    def test_empty_report_has_no_signals(self):
        t = _make_tracker()
        report = t.get_report()
        assert report.signals == []
        assert report.top_factors == []

    def test_record_entry_without_exit_does_not_affect_report(self):
        t = _make_tracker()
        t.record_entry("AAPL", {"god_level": 0.20})
        report = t.get_report()
        assert report.signals == []

    def test_record_entry_and_exit_adds_observation(self):
        t = _make_tracker()
        t.record_entry("AAPL", {"god_level": 0.20})
        t.record_exit("AAPL", realized_pnl_pct=0.03)
        assert len(t._observations["god_level"]) == 1

    def test_multiple_trades_compute_ic(self):
        t = _make_tracker()
        # Positive signal → positive P&L (predictive)
        for i in range(15):
            t.record_entry(f"SYM{i}", {"test_factor": float(i) / 10.0})
            t.record_exit(f"SYM{i}", realized_pnl_pct=float(i) * 0.002)
        report = t.get_report()
        factors = {r.signal_name: r for r in report.signals}
        assert "test_factor" in factors
        assert factors["test_factor"].ic > 0.5

    def test_unpredictive_signal_has_low_ic(self):
        t = _make_tracker()
        # Alternating signal direction but always profitable → IC near 0
        for i in range(20):
            sig = 0.20 if i % 2 == 0 else -0.20
            t.record_entry(f"SYM{i}", {"noise_factor": sig})
            t.record_exit(f"SYM{i}", realized_pnl_pct=0.01)  # always win
        report = t.get_report()
        factors = {r.signal_name: r for r in report.signals}
        assert "noise_factor" in factors
        assert abs(factors["noise_factor"].ic) < 0.3

    def test_report_sorted_by_ic_descending(self):
        t = _make_tracker()
        # Factor A: high IC; Factor B: low IC
        for i in range(20):
            t.record_entry(f"SYM{i}", {
                "factor_a": float(i),     # strongly predictive
                "factor_b": float(20 - i), # inversely predictive
            })
            t.record_exit(f"SYM{i}", realized_pnl_pct=float(i) * 0.001)
        report = t.get_report()
        ics = [r.ic for r in report.signals]
        assert ics == sorted(ics, reverse=True)

    def test_min_obs_gate(self):
        t = _make_tracker()
        # Only 3 observations (below default min_obs=10)
        for i in range(3):
            t.record_entry(f"SYM{i}", {"sparse_factor": float(i)})
            t.record_exit(f"SYM{i}", realized_pnl_pct=0.01)
        report = t.get_report()
        factors = {r.signal_name: r for r in report.signals}
        if "sparse_factor" in factors:
            assert factors["sparse_factor"].is_reliable is False
            assert factors["sparse_factor"].status == "unreliable"

    def test_window_limits_observations(self):
        t = _make_tracker()
        window = 50  # default
        for i in range(window + 20):
            t.record_entry(f"SYM{i}", {"rolling_factor": float(i % 5) / 10.0})
            t.record_exit(f"SYM{i}", realized_pnl_pct=0.01)
        assert len(t._observations["rolling_factor"]) == window

    def test_exit_without_entry_does_nothing(self):
        t = _make_tracker()
        t.record_exit("UNKNOWN", realized_pnl_pct=0.05)
        assert len(t._observations) == 0

    def test_report_dict_has_expected_keys(self):
        t = _make_tracker()
        d = t.get_report_dict()
        assert "signals" in d
        assert "top_factors" in d
        assert "weak_factors" in d
        assert "timestamp" in d

    def test_top_factors_are_active(self):
        t = _make_tracker()
        for i in range(15):
            t.record_entry(f"SYM{i}", {"good_factor": float(i) / 10.0})
            t.record_exit(f"SYM{i}", realized_pnl_pct=float(i) * 0.003)
        report = t.get_report()
        for name in report.top_factors:
            matching = [r for r in report.signals if r.signal_name == name]
            assert len(matching) == 1
            assert matching[0].status == "active"

    def test_persistence_round_trip(self):
        tmp = tempfile.mkdtemp()
        path = str(Path(tmp) / "ic.json")
        t1 = FactorICTracker(persist_path=path)
        for i in range(15):
            t1.record_entry(f"SYM{i}", {"persisted_factor": float(i) / 10.0})
            t1.record_exit(f"SYM{i}", realized_pnl_pct=float(i) * 0.002)

        t2 = FactorICTracker(persist_path=path)
        assert len(t2._observations["persisted_factor"]) == 15

    def test_win_rate_computed(self):
        t = _make_tracker()
        # 15 trades: positive signal + positive PnL
        for i in range(15):
            t.record_entry(f"SYM{i}", {"win_factor": 0.20})
            t.record_exit(f"SYM{i}", realized_pnl_pct=0.02)
        report = t.get_report()
        factors = {r.signal_name: r for r in report.signals}
        assert factors["win_factor"].win_rate == pytest.approx(1.0)

    def test_multiple_factors_per_trade(self):
        t = _make_tracker()
        for i in range(15):
            t.record_entry(f"SYM{i}", {
                "factor_x": float(i) / 15.0,
                "factor_y": 0.10,
            })
            t.record_exit(f"SYM{i}", realized_pnl_pct=float(i) * 0.001)
        report = t.get_report()
        names = [r.signal_name for r in report.signals]
        assert "factor_x" in names
        assert "factor_y" in names

    def test_obs_count_in_report(self):
        t = _make_tracker()
        for i in range(12):
            t.record_entry(f"SYM{i}", {"count_factor": 0.15})
            t.record_exit(f"SYM{i}", realized_pnl_pct=0.01)
        report = t.get_report()
        factors = {r.signal_name: r for r in report.signals}
        assert factors["count_factor"].obs == 12
