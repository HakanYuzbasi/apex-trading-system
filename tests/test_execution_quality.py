"""Tests for ExecutionQualityTracker."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from monitoring.execution_quality import ExecutionQualityTracker


def _make_tracker(**kwargs) -> ExecutionQualityTracker:
    defaults = dict(max_fills=500, penalty_p95_bps=30.0, penalty_floor=0.70, min_fills_for_penalty=3)
    defaults.update(kwargs)
    return ExecutionQualityTracker(**defaults)


def _fill(tracker, symbol="AAPL", side="BUY", expected=200.0, fill=201.0, qty=100.0, **kw):
    return tracker.record_fill(symbol, side, expected, fill, qty, **kw)


class TestRecordFill:

    def test_basic_fill_recorded(self):
        t = _make_tracker()
        rec = _fill(t)
        assert rec is not None
        assert len(t._fills) == 1

    def test_buy_adverse_slippage_positive_bps(self):
        """BUY fill > expected → adverse → positive bps."""
        t = _make_tracker()
        rec = _fill(t, expected=200.0, fill=201.0)
        assert rec.slippage_bps > 0
        assert pytest.approx(rec.slippage_bps, abs=0.5) == 50.0  # 1/200 × 10000 = 50 bps

    def test_sell_adverse_slippage_positive_bps(self):
        """SELL fill < expected → adverse → positive bps."""
        t = _make_tracker()
        rec = _fill(t, side="SELL", expected=200.0, fill=199.0)
        assert rec.slippage_bps > 0

    def test_favourable_fill_negative_bps(self):
        """BUY fill < expected → favourable → negative bps."""
        t = _make_tracker()
        rec = _fill(t, expected=200.0, fill=199.0)
        assert rec.slippage_bps < 0

    def test_invalid_fill_returns_none(self):
        t = _make_tracker()
        assert _fill(t, expected=0.0) is None
        assert _fill(t, fill=0.0) is None
        assert _fill(t, qty=0.0) is None

    def test_sym_slippage_populated(self):
        t = _make_tracker()
        _fill(t, symbol="MSFT")
        assert "MSFT" in t._sym_slippage
        assert len(t._sym_slippage["MSFT"]) == 1

    def test_daily_cost_tracked(self):
        t = _make_tracker()
        _fill(t, expected=200.0, fill=201.0, qty=100.0)
        total = sum(t._daily_slippage_cost.values())
        assert total > 0

    def test_fills_capped_at_max(self):
        t = _make_tracker(max_fills=5)
        for _ in range(10):
            _fill(t)
        assert len(t._fills) == 5


class TestSlippageStats:

    def test_no_fills_returns_zeros(self):
        t = _make_tracker()
        stats = t.get_symbol_slippage_bps("AAPL")
        assert stats["count"] == 0
        assert stats["p50"] == 0.0

    def test_stats_computed(self):
        t = _make_tracker()
        for _ in range(10):
            _fill(t, symbol="AAPL", expected=200.0, fill=201.0)
        stats = t.get_symbol_slippage_bps("AAPL")
        assert stats["count"] == 10
        assert stats["p50"] == pytest.approx(50.0, abs=1.0)
        assert stats["p95"] >= stats["p50"]

    def test_zero_slippage(self):
        t = _make_tracker()
        for _ in range(5):
            _fill(t, symbol="SPY", expected=500.0, fill=500.0)
        stats = t.get_symbol_slippage_bps("SPY")
        assert stats["p50"] == pytest.approx(0.0, abs=0.01)


class TestSizingPenalty:

    def test_no_penalty_below_min_fills(self):
        t = _make_tracker(min_fills_for_penalty=5)
        for _ in range(3):
            _fill(t, expected=200.0, fill=201.0)  # high slip
        assert t.get_sizing_penalty("AAPL") == 1.0

    def test_no_penalty_low_slippage(self):
        t = _make_tracker(penalty_p95_bps=30.0, min_fills_for_penalty=2)
        for _ in range(5):
            _fill(t, expected=200.0, fill=200.1)  # ~5bps, well below 30bps threshold
        assert t.get_sizing_penalty("AAPL") == pytest.approx(1.0, abs=0.01)

    def test_penalty_applied_high_slippage(self):
        t = _make_tracker(penalty_p95_bps=10.0, penalty_floor=0.70, min_fills_for_penalty=3)
        for _ in range(10):
            _fill(t, expected=200.0, fill=202.0)  # 100bps → well above 10bps threshold
        penalty = t.get_sizing_penalty("AAPL")
        assert penalty < 1.0
        assert penalty >= 0.70

    def test_penalty_floor_not_breached(self):
        t = _make_tracker(penalty_p95_bps=5.0, penalty_floor=0.70, min_fills_for_penalty=3)
        for _ in range(10):
            _fill(t, expected=200.0, fill=210.0)  # extreme slippage
        penalty = t.get_sizing_penalty("AAPL")
        assert penalty >= 0.70


class TestBrokerRegimeSummary:

    def test_broker_summary_groups_correctly(self):
        t = _make_tracker()
        for _ in range(5):
            _fill(t, broker="ibkr", expected=200.0, fill=200.5)
        for _ in range(5):
            _fill(t, broker="alpaca", expected=50000.0, fill=50200.0)
        summary = t.get_broker_summary()
        assert "ibkr" in summary
        assert "alpaca" in summary
        assert summary["alpaca"]["mean_bps"] > summary["ibkr"]["mean_bps"]

    def test_regime_summary_groups_correctly(self):
        t = _make_tracker()
        for _ in range(5):
            _fill(t, regime="bull", expected=200.0, fill=200.5)
        for _ in range(5):
            _fill(t, regime="bear", expected=200.0, fill=202.0)
        summary = t.get_regime_summary()
        assert "bull" in summary
        assert "bear" in summary
        assert summary["bear"]["mean_bps"] > summary["bull"]["mean_bps"]


class TestWorstSymbols:

    def test_worst_symbols_ordered_by_p95(self):
        t = _make_tracker(min_fills_for_penalty=3)
        for _ in range(5):
            _fill(t, symbol="BAD", expected=100.0, fill=101.0)   # 100bps
        for _ in range(5):
            _fill(t, symbol="OK", expected=100.0, fill=100.05)   # 5bps
        worst = t.get_worst_symbols(5)
        assert worst[0]["symbol"] == "BAD"
        assert worst[0]["p95_bps"] > worst[-1]["p95_bps"]

    def test_worst_symbols_excludes_low_fill_count(self):
        t = _make_tracker(min_fills_for_penalty=5)
        _fill(t, symbol="RARE")  # only 1 fill → excluded
        worst = t.get_worst_symbols(5)
        assert not any(w["symbol"] == "RARE" for w in worst)


class TestReport:

    def test_report_structure(self):
        t = _make_tracker()
        for _ in range(10):
            _fill(t)
        report = t.get_report()
        assert "global" in report
        assert "by_broker" in report
        assert "by_regime" in report
        assert "worst_symbols" in report
        assert "generated_at" in report

    def test_report_empty_when_no_fills(self):
        t = _make_tracker()
        report = t.get_report()
        assert report["global"] == {}

    def test_global_adverse_pct(self):
        t = _make_tracker()
        for _ in range(8):
            _fill(t, expected=200.0, fill=201.0)   # adverse
        for _ in range(2):
            _fill(t, expected=200.0, fill=199.0)   # favourable
        report = t.get_report()
        assert report["global"]["adverse_pct"] == pytest.approx(80.0, abs=1.0)


class TestPersistence:

    def test_persist_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = ExecutionQualityTracker(data_dir=Path(tmp), persist_interval_fills=5)
            for _ in range(5):
                _fill(t)
            files = list(Path(tmp).glob("execution_quality_*.json"))
            assert len(files) == 1
            data = json.loads(files[0].read_text())
            assert "global" in data

    def test_flush_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = ExecutionQualityTracker(data_dir=Path(tmp))
            for _ in range(3):
                _fill(t)
            t.flush()
            files = list(Path(tmp).glob("execution_quality_*.json"))
            assert len(files) == 1
