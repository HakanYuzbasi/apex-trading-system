"""
tests/test_strategy_health_monitor.py — Unit tests for StrategyHealthMonitor
"""
import math
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from monitoring.strategy_health_monitor import StrategyHealthMonitor, HealthState


def _make_monitor(tmp_dir: str = None) -> StrategyHealthMonitor:
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    return StrategyHealthMonitor(persist_path=str(Path(tmp_dir) / "health.json"))


class TestStrategyHealthMonitor:

    def test_initial_state_is_live(self):
        m = _make_monitor()
        state = m.check_health()
        assert state.paper_only is False

    def test_insufficient_trades_stays_live(self):
        m = _make_monitor()
        m.record_trade(pnl_pct=-0.05)
        m.record_trade(pnl_pct=-0.03)
        state = m.check_health()
        assert state.paper_only is False
        assert "insufficient" in state.reason

    def test_healthy_positive_sharpe_stays_live(self):
        m = _make_monitor()
        for _ in range(10):
            m.record_trade(pnl_pct=0.02)
        state = m.check_health()
        assert state.paper_only is False
        assert state.rolling_sharpe > 0

    def test_negative_sharpe_triggers_paper_only(self):
        m = _make_monitor()
        for _ in range(10):
            m.record_trade(pnl_pct=-0.03)
        state = m.check_health()
        assert state.paper_only is True
        assert state.rolling_sharpe < 0

    def test_paper_only_persists_until_recovery_threshold(self):
        m = _make_monitor()
        # Push into paper-only
        for _ in range(10):
            m.record_trade(pnl_pct=-0.03)
        assert m.check_health().paper_only is True
        # Add a few mild wins — not enough to hit recover_threshold of 0.30
        for _ in range(3):
            m.record_trade(pnl_pct=0.005)
        # Still paper-only (recovery not confirmed)
        state = m.check_health()
        assert state.paper_only is True

    def test_recovery_exits_paper_only(self):
        m = _make_monitor()
        # Push into paper-only
        for _ in range(10):
            m.record_trade(pnl_pct=-0.05)
        assert m.check_health().paper_only is True
        # Flush old trades and add strongly positive ones
        m._trades.clear()
        for _ in range(15):
            m.record_trade(pnl_pct=0.04)
        state = m.check_health()
        assert state.paper_only is False
        assert "recovered" in state.reason.lower()

    def test_trade_count_reflects_lookback_window(self):
        m = _make_monitor()
        now = datetime.now(timezone.utc)
        # Add 5 old (35 days ago) and 5 recent trades
        old_ts = now - timedelta(days=35)
        for _ in range(5):
            from monitoring.strategy_health_monitor import _TradeRecord
            m._trades.append(_TradeRecord(pnl_pct=0.02, ts=old_ts.isoformat()))
        for _ in range(5):
            m.record_trade(pnl_pct=0.02)
        state = m.check_health()
        assert state.trade_count == 5  # old trades evicted

    def test_rolling_sharpe_is_annualised(self):
        m = _make_monitor()
        # Constant positive returns → finite positive Sharpe
        for _ in range(20):
            m.record_trade(pnl_pct=0.01)
        state = m.check_health()
        assert state.rolling_sharpe > 0

    def test_constant_zero_returns_gives_zero_sharpe(self):
        m = _make_monitor()
        for _ in range(10):
            m.record_trade(pnl_pct=0.0)
        state = m.check_health()
        assert state.rolling_sharpe == pytest.approx(0.0)

    def test_persistence_survives_reload(self):
        tmp = tempfile.mkdtemp()
        m1 = _make_monitor(tmp)
        for _ in range(10):
            m1.record_trade(pnl_pct=-0.04)
        m1.check_health()
        assert m1.paper_only is True

        # Re-load from same path
        m2 = _make_monitor(tmp)
        assert m2.paper_only is True

    def test_paper_only_property(self):
        m = _make_monitor()
        assert m.paper_only is False
        for _ in range(10):
            m.record_trade(pnl_pct=-0.05)
        m.check_health()
        assert m.paper_only is True

    def test_get_state_dict_has_expected_keys(self):
        m = _make_monitor()
        d = m.get_state_dict()
        for key in ("rolling_sharpe", "trade_count", "paper_only", "reason", "last_updated"):
            assert key in d

    def test_zero_trades_returns_live(self):
        m = _make_monitor()
        state = m.check_health()
        assert state.paper_only is False
        assert state.trade_count == 0

    def test_mixed_returns_sharpe_computed(self):
        m = _make_monitor()
        pnls = [0.03, -0.01, 0.02, -0.02, 0.04, 0.01, -0.03, 0.02, 0.01, -0.01]
        for p in pnls:
            m.record_trade(pnl_pct=p)
        state = m.check_health()
        mean = sum(pnls) / len(pnls)
        std = (sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1)) ** 0.5
        expected_sharpe = (mean / std) * math.sqrt(252)
        assert state.rolling_sharpe == pytest.approx(expected_sharpe, abs=1e-3)

    def test_old_trades_evicted_before_check(self):
        m = _make_monitor()
        # Record 10 very old losing trades
        old_ts = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        from monitoring.strategy_health_monitor import _TradeRecord
        for _ in range(10):
            m._trades.append(_TradeRecord(pnl_pct=-0.10, ts=old_ts))
        state = m.check_health()
        # Old trades evicted → insufficient data → live
        assert state.paper_only is False

    def test_record_trade_with_explicit_timestamp(self):
        m = _make_monitor()
        ts = datetime.now(timezone.utc)
        m.record_trade(pnl_pct=0.01, timestamp=ts)
        assert len(m._trades) == 1
        assert m._trades[0].pnl_pct == pytest.approx(0.01)

    def test_save_and_load_preserves_paper_only(self):
        tmp = tempfile.mkdtemp()
        m1 = _make_monitor(tmp)
        for _ in range(10):
            m1.record_trade(pnl_pct=-0.05)
        m1.check_health()

        m2 = StrategyHealthMonitor(persist_path=str(Path(tmp) / "health.json"))
        assert m2.paper_only is True
        assert len(m2._trades) >= 10

    def test_min_trades_exactly_met(self):
        m = _make_monitor()
        # Exactly 5 positive trades (=MIN_TRADES) → should evaluate
        for _ in range(5):
            m.record_trade(pnl_pct=0.05)
        state = m.check_health()
        assert state.trade_count == 5
        assert "insufficient" not in state.reason

    def test_no_sharpe_overflow_on_all_positive(self):
        m = _make_monitor()
        for _ in range(50):
            m.record_trade(pnl_pct=0.10)
        state = m.check_health()
        assert math.isfinite(state.rolling_sharpe)

    def test_disabled_config_always_live(self, monkeypatch):
        import monitoring.strategy_health_monitor as shm_mod
        monkeypatch.setattr(shm_mod, "_cfg", lambda k: False if k == "STRATEGY_HEALTH_ENABLED" else shm_mod._DEF.get(k))
        m = _make_monitor()
        for _ in range(10):
            m.record_trade(pnl_pct=-0.05)
        state = m.check_health()
        assert state.paper_only is False
        assert state.reason == "disabled"
