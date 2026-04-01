"""
tests/test_daily_briefing.py — Unit tests for monitoring/daily_briefing.py
"""
from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from monitoring.daily_briefing import (
    DailyBriefingGenerator,
    DailyBriefing,
    TradeStats,
    SignalStats,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_gen(tmp: str) -> DailyBriefingGenerator:
    return DailyBriefingGenerator(data_dir=tmp)


def _write_exits(audit_dir: str, pnls: list, today: bool = True) -> None:
    ts_date = date.today() if today else (date.today() - timedelta(days=2))
    ts = datetime(ts_date.year, ts_date.month, ts_date.day,
                  15, 0, 0, tzinfo=timezone.utc).isoformat()
    path = Path(audit_dir) / "trade_audit_test.jsonl"
    with open(path, "a") as f:
        for i, pnl in enumerate(pnls):
            row = {"event": "EXIT", "symbol": f"SYM{i}", "pnl_pct": pnl, "ts": ts}
            f.write(json.dumps(row) + "\n")


# ── TradeStats ────────────────────────────────────────────────────────────────

class TestTradeStats:
    def test_zero_trades(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            stats = gen._compute_trade_stats(date.today())
            assert stats.total_trades == 0
            assert stats.win_rate == 0.0

    def test_all_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.01] * 10)
            gen = _make_gen(tmp)
            stats = gen._compute_trade_stats(date.today())
            assert stats.total_trades == 10
            assert stats.wins == 10
            assert stats.win_rate == pytest.approx(1.0)

    def test_mixed_wins_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.02] * 6 + [-0.01] * 4)
            gen = _make_gen(tmp)
            stats = gen._compute_trade_stats(date.today())
            assert stats.wins == 6
            assert stats.losses == 4
            assert stats.win_rate == pytest.approx(0.6)

    def test_best_and_worst_trade_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.05, -0.03, 0.01, 0.02])
            gen = _make_gen(tmp)
            stats = gen._compute_trade_stats(date.today())
            assert stats.best_trade is not None
            assert stats.worst_trade is not None
            assert float(stats.best_trade["pnl_pct"]) == pytest.approx(0.05)
            assert float(stats.worst_trade["pnl_pct"]) == pytest.approx(-0.03)

    def test_only_today_trades_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.01] * 5, today=True)
            _write_exits(tmp, [0.01] * 3, today=False)  # old trades — should be excluded
            gen = _make_gen(tmp)
            stats = gen._compute_trade_stats(date.today())
            assert stats.total_trades == 5

    def test_total_pnl_sums_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.01, 0.02, -0.01])
            gen = _make_gen(tmp)
            stats = gen._compute_trade_stats(date.today())
            assert stats.total_pnl_pct == pytest.approx(0.02, abs=1e-6)


# ── DailyBriefingGenerator ────────────────────────────────────────────────────

class TestDailyBriefingGenerator:

    def test_generate_returns_briefing(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            briefing = gen.generate(regime="neutral")
            assert isinstance(briefing, DailyBriefing)

    def test_briefing_has_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            briefing = gen.generate(regime="bull")
            assert briefing.date == str(date.today())
            assert briefing.regime == "bull"
            assert isinstance(briefing.recommendations, list)
            assert isinstance(briefing.adaptive_weights, dict)

    def test_generate_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            gen.generate(regime="neutral")
            output_dir = Path(tmp) / "daily_briefings"
            files = list(output_dir.glob("*.json"))
            assert len(files) == 1

    def test_get_latest_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            assert gen.get_latest() is None

    def test_get_latest_returns_most_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            gen.generate(regime="bear")
            result = gen.get_latest()
            assert result is not None
            assert result["regime"] == "bear"

    def test_get_history_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            assert gen.get_history() == []

    def test_get_history_returns_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            gen.generate(regime="neutral")
            history = gen.get_history(days=7)
            assert len(history) == 1

    def test_recommendations_for_low_winrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [-0.01] * 14 + [0.01] * 6)  # 30% win rate
            gen = _make_gen(tmp)
            briefing = gen.generate(regime="bear")
            recs_text = " ".join(briefing.recommendations)
            assert "40%" in recs_text or "win rate" in recs_text.lower()

    def test_recommendations_for_high_winrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_exits(tmp, [0.01] * 20)  # 100% win rate
            gen = _make_gen(tmp)
            briefing = gen.generate(regime="bull")
            recs_text = " ".join(briefing.recommendations)
            assert "60%" in recs_text or "strong" in recs_text.lower()

    def test_paper_gate_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            # Mock a paper-mode backtest gate
            mock_engine = MagicMock()
            mock_gate = MagicMock()
            mock_gate.mode = "paper"
            mock_engine._backtest_gate = mock_gate
            mock_shm = MagicMock()
            mock_shm.paper_only = False
            mock_engine._strategy_health = mock_shm
            mock_awm = MagicMock()
            mock_awm.get_report.return_value = {"weights": {}}
            mock_engine._adaptive_weights = mock_awm
            mock_fic = MagicMock()
            mock_fic.get_report.return_value = MagicMock(signals=[], weak_factors=[])
            mock_engine._factor_ic_tracker = mock_fic

            briefing = gen.generate(regime="bear", engine=mock_engine)
            recs_text = " ".join(briefing.recommendations)
            assert "paper" in recs_text.lower() or "PAPER" in recs_text

    def test_to_dict_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            briefing = gen.generate(regime="neutral")
            d = briefing.to_dict()
            # Should be serializable
            json.dumps(d)

    def test_to_text_contains_key_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            briefing = gen.generate(regime="bull")
            text = briefing.to_text()
            assert "PERFORMANCE" in text
            assert "SIGNAL" in text
            assert "SYSTEM STATE" in text
            assert "RECOMMENDATIONS" in text

    def test_generated_at_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            briefing = gen.generate()
            assert briefing.generated_at != ""
            # Should be parseable ISO timestamp
            datetime.fromisoformat(briefing.generated_at)

    def test_normal_operation_no_exceptions(self):
        """Full generate with no data should not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            briefing = gen.generate(regime="neutral")
            assert briefing is not None

    def test_backtest_mode_in_briefing(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            briefing = gen.generate(regime="neutral")
            assert briefing.backtest_gate_mode in ("live", "paper", "unknown")

    def test_history_respects_days_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_gen(tmp)
            # Generate one briefing
            gen.generate(regime="neutral")
            # Requesting 0 days should return empty
            history = gen.get_history(days=0)
            assert history == []
