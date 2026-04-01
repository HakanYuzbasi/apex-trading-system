"""
tests/test_backtest_gate.py — Unit tests for monitoring/backtest_gate.py
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from monitoring.backtest_gate import BacktestGate, PeriodMetrics, _sharpe


# ── _sharpe helper ────────────────────────────────────────────────────────────

class TestSharpe:
    def test_empty_list(self):
        assert _sharpe([]) == 0.0

    def test_too_few_values(self):
        assert _sharpe([0.01, 0.02]) == 0.0

    def test_positive_mean_positive_sharpe(self):
        pnls = [0.01] * 20 + [-0.002] * 5
        assert _sharpe(pnls) > 0.0

    def test_negative_mean_negative_sharpe(self):
        pnls = [-0.01] * 20 + [0.001] * 3
        assert _sharpe(pnls) < 0.0

    def test_all_same_positive_returns_sentinel(self):
        # Zero std, positive mean → +10.0 sentinel
        assert _sharpe([0.01] * 10) == pytest.approx(10.0)

    def test_all_same_negative_returns_sentinel(self):
        assert _sharpe([-0.01] * 10) == pytest.approx(-10.0)

    def test_all_zero_returns_zero(self):
        assert _sharpe([0.0] * 10) == pytest.approx(0.0)


# ── BacktestGate helpers ──────────────────────────────────────────────────────

def _make_gate(tmp: str) -> BacktestGate:
    return BacktestGate(
        audit_dir=tmp,
        state_path=str(Path(tmp) / "backtest_gate_state.json"),
    )


def _write_exits(audit_dir: str, pnls: list, days_ago: float = 0.5) -> None:
    """Write fake EXIT rows to a trade_audit JSONL file."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    path = Path(audit_dir) / "trade_audit_test.jsonl"
    with open(path, "a") as f:
        for i, pnl in enumerate(pnls):
            row = {"event": "EXIT", "symbol": f"SYM{i}", "pnl_pct": pnl, "ts": ts}
            f.write(json.dumps(row) + "\n")


# ── BacktestGate tests ────────────────────────────────────────────────────────

class TestBacktestGate:

    def test_initial_mode_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            assert gate.mode == "unknown"

    def test_is_live_false_when_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            assert gate.is_live is False

    def test_is_paper_false_when_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            assert gate.is_paper is False

    def test_empty_audit_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            record = gate.run_evaluation()
            assert record.mode == "unknown"
            assert gate.mode == "unknown"

    def test_healthy_trades_set_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Write enough healthy trades
            _write_exits(tmp, [0.01] * 20)
            gate = _make_gate(tmp)
            # Need 2 consecutive healthy runs to go live
            gate.run_evaluation()
            gate.run_evaluation()
            assert gate.mode == "live"

    def test_degraded_winrate_triggers_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Win rate far below 0.40 floor
            pnls = [-0.01] * 18 + [0.01] * 2  # 10% win rate
            _write_exits(tmp, pnls)
            gate = _make_gate(tmp)
            record = gate.run_evaluation()
            assert "low_winrate" in record.triggered_flags

    def test_two_degraded_runs_trigger_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            pnls = [-0.01] * 18 + [0.01] * 2  # 10% win rate
            _write_exits(tmp, pnls)
            gate = _make_gate(tmp)
            gate.run_evaluation()
            gate.run_evaluation()
            assert gate.mode == "paper"

    def test_recovery_from_paper_requires_consec_good(self):
        with tempfile.TemporaryDirectory() as tmp:
            # First push into paper
            bad_pnls = [-0.01] * 18 + [0.01] * 2
            _write_exits(tmp, bad_pnls)
            gate = _make_gate(tmp)
            gate.run_evaluation()
            gate.run_evaluation()
            assert gate.mode == "paper"

            # Replace with healthy trades
            audit_file = Path(tmp) / "trade_audit_test.jsonl"
            audit_file.write_text("")  # clear
            _write_exits(tmp, [0.01] * 20)

            gate.run_evaluation()
            gate.run_evaluation()
            assert gate.mode == "live"

    def test_force_live_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            gate.force_paper()
            assert gate.is_paper
            gate.force_live()
            assert gate.is_live

    def test_force_paper_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.01] * 20)
            gate = _make_gate(tmp)
            gate.run_evaluation()
            gate.run_evaluation()
            assert gate.is_live
            gate.force_paper()
            assert gate.is_paper

    def test_get_state_has_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            state = gate.get_state()
            assert "mode" in state
            assert "is_live" in state
            assert "last_eval_ts" in state
            assert "consec_bad" in state
            assert "consec_good" in state
            assert "history" in state

    def test_history_grows_with_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.01] * 20)
            gate = _make_gate(tmp)
            for _ in range(5):
                gate.run_evaluation()
            assert len(gate.get_history()) == 5

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = str(Path(tmp) / "gate_state.json")
            _write_exits(tmp, [-0.01] * 18 + [0.01] * 2)

            g1 = BacktestGate(audit_dir=tmp, state_path=state_path)
            g1.run_evaluation()
            g1.run_evaluation()
            assert g1.mode == "paper"

            g2 = BacktestGate(audit_dir=tmp, state_path=state_path)
            assert g2.mode == "paper"

    def test_last_eval_ts_set_after_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            assert gate._last_eval_ts == ""
            gate.run_evaluation()
            assert gate._last_eval_ts != ""

    def test_record_has_current_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.01] * 15)
            gate = _make_gate(tmp)
            record = gate.run_evaluation()
            assert record.current.trades == 15
            assert record.current.win_rate == pytest.approx(1.0)

    def test_record_sharpe_delta_when_no_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.01] * 15)
            gate = _make_gate(tmp)
            record = gate.run_evaluation()
            assert record.sharpe_delta == 0.0

    def test_disabled_gate_always_live(self, monkeypatch):
        import monitoring.backtest_gate as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "BACKTEST_GATE_ENABLED" else mod._DEF.get(k))
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            record = gate.run_evaluation()
            assert record.mode == "live"

    def test_period_metrics_is_healthy_true(self):
        m = PeriodMetrics(trades=20, win_rate=0.55, avg_pnl_pct=0.005, sharpe=0.8)
        assert m.is_healthy(win_floor=0.40) is True

    def test_period_metrics_is_healthy_false_low_winrate(self):
        m = PeriodMetrics(trades=20, win_rate=0.30, avg_pnl_pct=-0.002, sharpe=-0.1,
                          flags=["low_winrate"])
        assert m.is_healthy(win_floor=0.40) is False

    def test_get_history_returns_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _make_gate(tmp)
            assert isinstance(gate.get_history(), list)

    def test_sharpe_degrade_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Old (comparison window: 30-45 days ago) = profitable trades
            # New (current window: 0-30 days ago) = losing trades
            # sharpe_delta = sharpe(new) - sharpe(old) should be negative
            audit_file = Path(tmp) / "trade_audit_old.jsonl"
            ts_old = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
            ts_new = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            with open(audit_file, "w") as f:
                for i in range(20):
                    row = {"event": "EXIT", "symbol": f"OLD{i}", "pnl_pct": 0.015, "ts": ts_old}
                    f.write(json.dumps(row) + "\n")
                for i in range(15):
                    row = {"event": "EXIT", "symbol": f"NEW{i}", "pnl_pct": -0.02, "ts": ts_new}
                    f.write(json.dumps(row) + "\n")

            gate = _make_gate(tmp)
            record = gate.run_evaluation()
            assert record.sharpe_delta < 0
