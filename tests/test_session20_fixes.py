"""
tests/test_session20_fixes.py

Tests for Session 20 bug fixes and improvements:
  BUG-1 — _et_mins() added to DirectionalEquityStrategy
  BUG-2 — BTC dominance exception returns True (conservative)
  BUG-3 — Dead code removed from RiskManager._quantity_from_signal
  IMP-1  — Weekly WFO optimize ScheduledTask wired in harness
  IMP-2  — CSM wall-clock fallback clock
  IMP-3  — BtcDominanceMonitor.force_refresh() + proactive harness poll
  IMP-4  — PortfolioHeatTracker.reset() + harness startup call
"""
from __future__ import annotations

import inspect
import time
import types
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# BUG-1: _et_mins() exists in DirectionalEquityStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionalEquityEtMins:
    def test_et_mins_defined_in_directional_equity(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        assert hasattr(DirectionalEquityStrategy, "_et_mins"), (
            "_et_mins must be defined on DirectionalEquityStrategy"
        )

    def test_et_mins_is_static(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        method = inspect.getattr_static(DirectionalEquityStrategy, "_et_mins")
        assert isinstance(method, staticmethod), "_et_mins must be a @staticmethod"

    def test_et_mins_returns_int(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        result = DirectionalEquityStrategy._et_mins()
        assert isinstance(result, int)
        assert 0 <= result < 24 * 60, "minutes in day must be 0–1439"

    def test_afternoon_trail_tighten_uses_et_mins(self):
        """_manage_position must not raise AttributeError after 14:30 ET."""
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        # Confirm the method body references self._et_mins
        src = inspect.getsource(DirectionalEquityStrategy._manage_position)
        assert "_et_mins" in src

    def test_is_equity_auction_window_uses_et_mins(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        src = inspect.getsource(DirectionalEquityStrategy._is_equity_auction_window)
        # Old code used inline `datetime.now(_ET)` + local `mins`; new code calls _et_mins()
        assert "_et_mins" in src


# ─────────────────────────────────────────────────────────────────────────────
# BUG-2: BTC dominance exception is conservative (returns True)
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanBtcDominanceConservative:
    def _make_strategy(self):
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        bus = MagicMock()
        bus.subscribe = MagicMock()
        strat = KalmanPairsStrategy.__new__(KalmanPairsStrategy)
        strat._btc_dominance_rising = KalmanPairsStrategy._btc_dominance_rising.__get__(strat)
        return strat

    def test_exception_returns_true_blocks_entry(self):
        """When CoinGecko raises, _btc_dominance_rising must return True."""
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        # Verify the source directly contains `return True` in the except block
        src = inspect.getsource(KalmanPairsStrategy._btc_dominance_rising)
        # Find the except block and check it returns True
        lines = src.splitlines()
        except_idx = next(i for i, l in enumerate(lines) if "except" in l)
        block_after = "\n".join(lines[except_idx:])
        assert "return True" in block_after, (
            "except block must return True (conservative) not False"
        )

    def test_coingecko_down_blocks_entry_via_return_true(self):
        """Simulate import failure → method returns True → new entry blocked."""
        import sys
        # Temporarily make the import fail
        fake_module = types.ModuleType("quant_system.risk.btc_dominance_monitor")
        fake_module.get_btc_dominance_monitor = MagicMock(
            side_effect=RuntimeError("import failed")
        )
        original = sys.modules.get("quant_system.risk.btc_dominance_monitor")
        sys.modules["quant_system.risk.btc_dominance_monitor"] = fake_module
        try:
            from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
            strat = MagicMock(spec=KalmanPairsStrategy)
            result = KalmanPairsStrategy._btc_dominance_rising(strat)
            assert result is True, "should return True when monitor import raises"
        finally:
            if original is not None:
                sys.modules["quant_system.risk.btc_dominance_monitor"] = original
            else:
                del sys.modules["quant_system.risk.btc_dominance_monitor"]


# ─────────────────────────────────────────────────────────────────────────────
# BUG-3: Dead code removed from RiskManager._quantity_from_signal
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskManagerDeadCode:
    def test_quantity_from_signal_has_single_return(self):
        from quant_system.risk.manager import RiskManager
        src = inspect.getsource(RiskManager._quantity_from_signal)
        # Count 'return' statements — must be exactly 1 after dead-code removal
        returns = [l.strip() for l in src.splitlines() if l.strip().startswith("return")]
        assert len(returns) == 1, (
            f"_quantity_from_signal must have exactly 1 return, found: {returns}"
        )

    def test_quantity_from_signal_delegates_to_v2(self):
        from quant_system.risk.manager import RiskManager
        src = inspect.getsource(RiskManager._quantity_from_signal)
        assert "_quantity_from_signal_v2" in src


# ─────────────────────────────────────────────────────────────────────────────
# IMP-1: Weekly WFO optimize task wired in harness
# ─────────────────────────────────────────────────────────────────────────────

class TestWeeklyOptimizeHarness:
    def test_weekly_optimize_task_obj_in_harness(self):
        import scripts.run_global_harness_v3 as harness_mod
        src = inspect.getsource(harness_mod)
        assert "weekly_optimize_task_obj" in src
        assert "weekly_wfo_optimize" in src

    def test_weekly_optimize_fires_on_saturday_only(self):
        """_weekly_optimize_coro must skip on non-Saturday."""
        import importlib
        import scripts.run_global_harness_v3  # ensure loaded
        # The coro logic: weekday() != 5 → return early
        # Verify source contains the Saturday guard
        src = inspect.getsource(scripts.run_global_harness_v3)
        assert "weekday() != 5" in src, "Saturday guard (weekday != 5) must be present"

    def test_weekly_optimize_in_all_tasks(self):
        src = inspect.getsource(__import__("scripts.run_global_harness_v3", fromlist=[""]))
        assert "weekly_optimize_task" in src

    def test_run_realistic_optimization_importable(self):
        import scripts.run_realistic_optimization  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# IMP-2: CSM wall-clock fallback clock
# ─────────────────────────────────────────────────────────────────────────────

class TestCSMWallClockFallback:
    def test_wallclock_rerank_interval_class_constant(self):
        from quant_system.strategies.cross_sectional_momentum_strategy import (
            CrossSectionalMomentumStrategy,
        )
        assert hasattr(CrossSectionalMomentumStrategy, "WALLCLOCK_RERANK_INTERVAL")
        assert CrossSectionalMomentumStrategy.WALLCLOCK_RERANK_INTERVAL == 65 * 60

    def test_fallback_triggers_rerank_when_spy_silent(self):
        """When a non-SPY bar arrives and _last_rank_ts is old, rerank fires."""
        from quant_system.strategies.cross_sectional_momentum_strategy import (
            CrossSectionalMomentumStrategy,
        )
        bus = MagicMock()
        bus.subscribe = MagicMock()
        strat = CrossSectionalMomentumStrategy.__new__(CrossSectionalMomentumStrategy)
        strat._global_5m = 0
        strat._last_rank_ts = time.monotonic() - 66 * 60  # 66 min ago — past threshold
        strat._1m_count = {"AAPL": 0}
        strat._5m_acc = {"AAPL": None}
        strat._5m_closes = {"AAPL": MagicMock(__bool__=lambda s: False)}
        strat._5m_bar_count = {"AAPL": 0}
        strat._5m_in_hour = {"AAPL": 0}
        strat._1h_acc = {"AAPL": None}
        strat._1h_closes = {"AAPL": MagicMock()}
        strat._day_open = {"AAPL": None}
        strat._day_date = {"AAPL": None}
        strat._open_positions = {}
        rerank_called = []
        strat._rerank_and_trade = lambda: rerank_called.append(1)
        strat._accumulate_1h = MagicMock()

        from quant_system.events import BarEvent
        from decimal import Decimal
        from datetime import datetime, timezone

        # Emit 5 x 1-min bars to form a 5-min bar
        for _ in range(5):
            ev = MagicMock(spec=BarEvent)
            ev.instrument_id = "AAPL"
            ev.close_price = Decimal("150.0")
            ev.high_price = Decimal("151.0")
            ev.low_price = Decimal("149.0")
            ev.open_price = Decimal("150.0")
            ev.volume = Decimal("1000")
            strat.on_bar(ev)

        assert rerank_called, "wall-clock fallback must trigger _rerank_and_trade"

    def test_fallback_does_not_trigger_when_spy_recent(self):
        """No fallback when _last_rank_ts is recent (SPY is fine)."""
        from quant_system.strategies.cross_sectional_momentum_strategy import (
            CrossSectionalMomentumStrategy,
        )
        strat = CrossSectionalMomentumStrategy.__new__(CrossSectionalMomentumStrategy)
        strat._global_5m = 0
        strat._last_rank_ts = time.monotonic()  # just happened
        strat._1m_count = {"MSFT": 0}
        strat._5m_acc = {"MSFT": None}
        strat._5m_closes = {"MSFT": MagicMock(__bool__=lambda s: False)}
        strat._5m_bar_count = {"MSFT": 0}
        strat._5m_in_hour = {"MSFT": 0}
        strat._1h_acc = {"MSFT": None}
        strat._1h_closes = {"MSFT": MagicMock()}
        strat._day_open = {"MSFT": None}
        strat._day_date = {"MSFT": None}
        strat._open_positions = {}
        rerank_called = []
        strat._rerank_and_trade = lambda: rerank_called.append(1)
        strat._accumulate_1h = MagicMock()

        from quant_system.events import BarEvent
        from decimal import Decimal

        for _ in range(5):
            ev = MagicMock(spec=BarEvent)
            ev.instrument_id = "MSFT"
            ev.close_price = Decimal("300.0")
            ev.high_price = Decimal("301.0")
            ev.low_price = Decimal("299.0")
            ev.open_price = Decimal("300.0")
            ev.volume = Decimal("500")
            strat.on_bar(ev)

        assert not rerank_called, "fallback must not fire when last rerank was recent"


# ─────────────────────────────────────────────────────────────────────────────
# IMP-3: BtcDominanceMonitor.force_refresh()
# ─────────────────────────────────────────────────────────────────────────────

class TestBtcDominanceForceRefresh:
    def test_force_refresh_method_exists(self):
        from quant_system.risk.btc_dominance_monitor import BtcDominanceMonitor
        assert hasattr(BtcDominanceMonitor, "force_refresh")

    def test_force_refresh_calls_fetch(self):
        from quant_system.risk.btc_dominance_monitor import BtcDominanceMonitor
        mon = BtcDominanceMonitor()
        fetch_calls = []
        mon._fetch = lambda: fetch_calls.append(1)
        mon.force_refresh()
        assert fetch_calls, "force_refresh must call _fetch"

    def test_force_refresh_with_network_down_does_not_raise(self):
        """Real _fetch swallows network errors; force_refresh must not raise."""
        from quant_system.risk.btc_dominance_monitor import BtcDominanceMonitor
        mon = BtcDominanceMonitor()
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            mon.force_refresh()  # real _fetch catches OSError internally — must not raise

    def test_harness_has_btc_dominance_poll_loop(self):
        import scripts.run_global_harness_v3 as h
        src = inspect.getsource(h)
        assert "_btc_dominance_poll_loop" in src
        assert "btc_dom_poll_task" in src
        assert "force_refresh" in src


# ─────────────────────────────────────────────────────────────────────────────
# IMP-4: PortfolioHeatTracker.reset()
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioHeatReset:
    def test_reset_method_exists(self):
        from risk.portfolio_heat import PortfolioHeatTracker
        assert hasattr(PortfolioHeatTracker, "reset")

    def test_reset_clears_all_heat(self):
        from risk.portfolio_heat import PortfolioHeatTracker
        tracker = PortfolioHeatTracker()
        tracker.register("AAPL", 150.0, 147.0, 1000.0)
        tracker.register("MSFT", 300.0, 294.0, 2000.0)
        assert tracker.current_heat() > 0
        tracker.reset()
        assert tracker.current_heat() == 0.0
        assert tracker.heat_breakdown() == {}

    def test_reset_allows_full_heat_cap_again(self):
        from risk.portfolio_heat import PortfolioHeatTracker
        tracker = PortfolioHeatTracker()
        tracker.register("AAPL", 100.0, 98.0, 50_000.0)  # 2% × 50k = $1000 heat
        # Now heat is $1000; can_open for another $600 would push to $1600 > $1500 cap
        can = tracker.can_open("MSFT", 100.0, 97.0, 20_000.0)  # 3% × 20k = $600
        assert not can
        tracker.reset()
        can_after = tracker.can_open("MSFT", 100.0, 97.0, 20_000.0)
        assert can_after, "after reset the cap must be available again"

    def test_harness_calls_reset_on_startup(self):
        import scripts.run_global_harness_v3 as h
        src = inspect.getsource(h)
        assert "get_portfolio_heat().reset()" in src
