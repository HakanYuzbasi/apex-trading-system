"""Tests for TradeDiagnosticsTracker."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from monitoring.trade_diagnostics import (
    GateDecision,
    TradeDiagnosticsTracker,
    TradeDecisionRecord,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _gate(name="confidence_threshold", fired=True, blocked=True, value=0.60, threshold=0.68):
    return GateDecision(gate=name, fired=fired, blocked=blocked, value=value, threshold=threshold)


def _rec(
    symbol="AAPL",
    action="blocked",
    block_gate="confidence_threshold",
    regime="neutral",
    signal=0.18,
    confidence=0.60,
    gates=None,
    ts=None,
):
    return TradeDecisionRecord(
        symbol=symbol,
        ts=ts if ts is not None else time.time(),
        regime=regime,
        signal=signal,
        confidence=confidence,
        gates=gates if gates is not None else [_gate(name=block_gate)],
        action=action,
        block_gate=block_gate,
    )


def _tracker(max_records=500):
    return TradeDiagnosticsTracker(max_records=max_records)


# ── default state ─────────────────────────────────────────────────────────────

class TestDefaultState:
    def test_empty_report_returns_dict(self):
        t = _tracker()
        r = t.get_report()
        assert isinstance(r, dict)

    def test_total_records_zero(self):
        t = _tracker()
        assert t.get_report()["total_records"] == 0

    def test_gate_attribution_empty(self):
        t = _tracker()
        assert t.get_gate_attribution() == {}

    def test_blocked_analysis_zero_total(self):
        t = _tracker()
        ba = t.get_blocked_analysis()
        assert ba["total_decisions"] == 0
        assert ba["block_rate"] == 0.0

    def test_symbol_report_empty(self):
        t = _tracker()
        r = t.get_symbol_report("AAPL")
        assert r["total_decisions"] == 0
        assert r["win_rate"] is None


# ── record_decision ───────────────────────────────────────────────────────────

class TestRecordDecision:
    def test_blocked_record_added(self):
        t = _tracker()
        t.record_decision(_rec(action="blocked"))
        assert t.get_report()["total_records"] == 1

    def test_entered_record_pending(self):
        t = _tracker()
        t.record_decision(_rec(action="entered", symbol="MSFT"))
        # entered but no outcome yet — still in pending
        r = t.get_symbol_report("MSFT")
        assert r["entered"] == 1
        assert r["completed_trades"] == 0

    def test_max_records_capped(self):
        t = TradeDiagnosticsTracker(max_records=5)
        for _ in range(10):
            t.record_decision(_rec())
        assert t.get_report()["total_records"] == 5


# ── record_outcome ────────────────────────────────────────────────────────────

class TestRecordOutcome:
    def test_outcome_attached(self):
        t = _tracker()
        t.record_decision(_rec(action="entered", symbol="NVDA"))
        t.record_outcome("NVDA", pnl_pct=0.02, hold_hours=3.0, exit_reason="tp")
        r = t.get_symbol_report("NVDA")
        assert r["completed_trades"] == 1
        assert r["win_rate"] == pytest.approx(1.0)

    def test_losing_outcome_win_rate_zero(self):
        t = _tracker()
        t.record_decision(_rec(action="entered", symbol="GE"))
        t.record_outcome("GE", pnl_pct=-0.03, hold_hours=1.0, exit_reason="stop")
        r = t.get_symbol_report("GE")
        assert r["win_rate"] == pytest.approx(0.0)

    def test_outcome_missing_symbol_no_error(self):
        t = _tracker()
        t.record_outcome("UNKNOWN", pnl_pct=0.01, hold_hours=1.0, exit_reason="tp")
        # no crash

    def test_multiple_outcomes_mixed(self):
        t = _tracker()
        for i in range(4):
            t.record_decision(_rec(action="entered", symbol=f"X_{i}"))
            t.record_outcome(f"X_{i}", pnl_pct=0.01 if i < 3 else -0.01, hold_hours=2.0, exit_reason="tp")
        report = t.get_report()
        assert report["overall_win_rate"] == pytest.approx(0.75)


# ── gate attribution ──────────────────────────────────────────────────────────

class TestGateAttribution:
    def test_attribution_counts_correct_gate(self):
        t = _tracker()
        for _ in range(3):
            t.record_decision(_rec(block_gate="confidence_threshold"))
        for _ in range(2):
            t.record_decision(_rec(block_gate="drawdown_gate"))
        attr = t.get_gate_attribution()
        assert attr["confidence_threshold"]["blocks"] == 3
        assert attr["drawdown_gate"]["blocks"] == 2

    def test_attribution_sorted_by_blocks_descending(self):
        t = _tracker()
        for _ in range(5):
            t.record_decision(_rec(block_gate="signal_momentum"))
        for _ in range(2):
            t.record_decision(_rec(block_gate="consensus"))
        keys = list(t.get_gate_attribution().keys())
        assert keys[0] == "signal_momentum"

    def test_block_rate_range(self):
        t = _tracker()
        t.record_decision(_rec(action="blocked"))
        t.record_decision(_rec(action="entered", gates=[_gate(blocked=False)], block_gate=""))
        for gate, data in t.get_gate_attribution().items():
            assert 0.0 <= data["block_rate"] <= 1.0

    def test_symbol_filter(self):
        t = _tracker()
        t.record_decision(_rec(symbol="AAPL", block_gate="confidence_threshold"))
        t.record_decision(_rec(symbol="MSFT", block_gate="drawdown_gate"))
        aapl_attr = t.get_gate_attribution(symbol="AAPL")
        assert "confidence_threshold" in aapl_attr
        assert "drawdown_gate" not in aapl_attr

    def test_lookback_filter_excludes_old(self):
        t = _tracker()
        old_ts = time.time() - 30 * 86400  # 30 days ago
        t.record_decision(_rec(ts=old_ts))
        t.record_decision(_rec())  # recent
        attr = t.get_gate_attribution(lookback_days=7)
        total = list(attr.values())[0]["total_decisions"] if attr else 0
        assert total == 1


# ── blocked analysis ──────────────────────────────────────────────────────────

class TestBlockedAnalysis:
    def test_block_rate_all_blocked(self):
        t = _tracker()
        for _ in range(5):
            t.record_decision(_rec(action="blocked"))
        ba = t.get_blocked_analysis()
        assert ba["block_rate"] == pytest.approx(1.0)

    def test_block_rate_none_blocked(self):
        t = _tracker()
        for _ in range(5):
            t.record_decision(_rec(action="entered", gates=[], block_gate=""))
        ba = t.get_blocked_analysis()
        assert ba["block_rate"] == pytest.approx(0.0)

    def test_by_first_gate_populated(self):
        t = _tracker()
        t.record_decision(_rec(block_gate="tiered_confidence"))
        t.record_decision(_rec(block_gate="tiered_confidence"))
        t.record_decision(_rec(block_gate="drawdown_gate"))
        ba = t.get_blocked_analysis()
        assert ba["by_first_gate"]["tiered_confidence"] == 2
        assert ba["by_first_gate"]["drawdown_gate"] == 1


# ── symbol report ─────────────────────────────────────────────────────────────

class TestSymbolReport:
    def test_block_rate_correct(self):
        t = _tracker()
        t.record_decision(_rec(symbol="TSLA", action="blocked"))
        t.record_decision(_rec(symbol="TSLA", action="entered", gates=[], block_gate=""))
        r = t.get_symbol_report("TSLA")
        assert r["block_rate"] == pytest.approx(0.5)

    def test_top_blocking_gates_capped_at_5(self):
        t = _tracker()
        for i in range(7):
            t.record_decision(_rec(symbol="Z", block_gate=f"gate_{i}"))
        r = t.get_symbol_report("Z")
        assert len(r["top_blocking_gates"]) <= 5

    def test_avg_pnl_positive(self):
        t = _tracker()
        for _ in range(3):
            t.record_decision(_rec(action="entered", symbol="AMD", gates=[], block_gate=""))
            t.record_outcome("AMD", pnl_pct=0.02, hold_hours=2.0, exit_reason="tp")
        r = t.get_symbol_report("AMD")
        assert r["avg_pnl_pct"] == pytest.approx(0.02)


# ── persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_load_reload_preserves_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            t1 = TradeDiagnosticsTracker(data_dir=d)
            t1.record_decision(_rec(action="entered", symbol="SAVE"))
            t1.record_outcome("SAVE", pnl_pct=0.05, hold_hours=6.0, exit_reason="tp")
            t2 = TradeDiagnosticsTracker(data_dir=d)
            r = t2.get_symbol_report("SAVE")
            assert r["completed_trades"] == 1
            assert r["win_rate"] == pytest.approx(1.0)

    def test_persists_blocked_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            t1 = TradeDiagnosticsTracker(data_dir=d)
            for _ in range(3):
                t1.record_decision(_rec(block_gate="drawdown_gate"))
            t2 = TradeDiagnosticsTracker(data_dir=d)
            ba = t2.get_blocked_analysis()
            assert ba["total_blocked"] >= 1

    def test_gate_decisions_survive_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            t1 = TradeDiagnosticsTracker(data_dir=d)
            t1.record_decision(_rec(
                gates=[GateDecision("my_gate", True, True, 0.55, 0.68)],
                block_gate="my_gate",
            ))
            t2 = TradeDiagnosticsTracker(data_dir=d)
            attr = t2.get_gate_attribution()
            assert "my_gate" in attr
