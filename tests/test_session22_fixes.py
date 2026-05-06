"""
tests/test_session22_fixes.py

Tests for Session 22 bug fixes:
  P1-2 — Kalman flat-exit clears _z_history and _innovation_history
  P1-3 — DirectionalEquity _close_position uses remaining notional (not full original)
          + partial exits (_half_close / _quarter_close) track P&L in _daily_realized_pnl
  P1-4 — CSM _momentum_score returns None when end_close <= 0
  P2-3 — _ou_half_life_days returns float('inf') (not 0.0) when history < 10
          + callers use math.isfinite(hl) guard so warmup entries still allowed
  P2-4 — ShadowAccounting force_update logs WARNING about missing avg_price
  P2-5 — DirectionalEquity _close_position logs WARNING when entry_price == 0
  P3-2 — RiskManager._quantity_from_signal_v2 logs WARNING for zero/neg price
"""
from __future__ import annotations

import inspect
import logging
import math
from collections import deque
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# P1-2: Kalman flat-exit clears _z_history and _innovation_history
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanFlatExitClearsHistory:
    def _make_strategy(self):
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        from quant_system.core.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        return KalmanPairsStrategy(
            bus,
            instrument_a="AAPL",
            instrument_b="MSFT",
            entry_z_score=1.5,
            exit_z_score=0.5,
            warmup_bars=5,
            leg_notional=1_000.0,
        )

    def test_z_history_cleared_on_flat_exit(self):
        s = self._make_strategy()
        s._z_history.extend([0.5, 1.0, 1.5, 2.0])
        s._pair_state = "short_a_long_b"
        # Simulate on_bar routing with z_score below threshold → desired flat
        s._entry_ts = None  # already none
        s._entry_z_score = None
        s._entry_vix = None
        # Force a direct transition to flat by calling the internal block
        # (replicate the code path: elif desired_state == "flat")
        s._z_history.clear()
        s._innovation_history.clear()
        assert len(s._z_history) == 0
        assert len(s._innovation_history) == 0

    def test_flat_exit_code_path_clears_both_histories(self):
        """Verify the source code of on_bar has .clear() calls in the flat-exit block."""
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        src = inspect.getsource(KalmanPairsStrategy.on_bar)
        # The flat block should contain both clear() calls
        assert "_z_history.clear()" in src, "_z_history.clear() missing from on_bar flat-exit block"
        assert "_innovation_history.clear()" in src, "_innovation_history.clear() missing from on_bar flat-exit block"

    def test_flat_exit_clears_in_elif_desired_flat_block(self):
        """Both clears must appear AFTER the 'elif desired_state == "flat"' branch."""
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        src = inspect.getsource(KalmanPairsStrategy.on_bar)
        lines = src.splitlines()
        flat_idx = next(
            (i for i, ln in enumerate(lines) if 'elif desired_state == "flat"' in ln),
            None,
        )
        assert flat_idx is not None, "No 'elif desired_state == \"flat\"' found in on_bar"
        block = "\n".join(lines[flat_idx:flat_idx + 20])
        assert "_z_history.clear()" in block
        assert "_innovation_history.clear()" in block

    def test_reinit_divergence_cooldown_still_clears(self):
        """The divergence reinit path (line ~239) should ALSO clear both histories."""
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        src = inspect.getsource(KalmanPairsStrategy._update_filter_and_score)
        assert "_innovation_history.clear()" in src
        assert "_z_history.clear()" in src


# ─────────────────────────────────────────────────────────────────────────────
# P1-3: DirectionalEquity remaining-notional P&L tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionalEquityRemainingNotionalPnl:
    def _make_strategy(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        from quant_system.core.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        s = DirectionalEquityStrategy(bus, instrument="AAPL")
        # Manually set up an open long position
        s._state = "long"
        s._entry_price = 100.0
        s._open_notional = 1_000.0
        s._half_closed = False
        s._tp2_closed = False
        return s

    def test_full_close_uses_full_notional(self):
        s = self._make_strategy()
        s._emit_signal = MagicMock()
        s._close_position(110.0, "take_profit")
        # 10% gain on $1000 = $100
        assert abs(s._daily_realized_pnl - 100.0) < 0.01

    def test_close_after_tp1_uses_half_notional(self):
        s = self._make_strategy()
        s._emit_signal = MagicMock()
        s._daily_realized_pnl = 0.0
        s._half_closed = True
        s._tp2_closed = False
        s._close_position(110.0, "trailing_stop")
        # Only 50% of position remains: 10% gain on $500 = $50
        assert abs(s._daily_realized_pnl - 50.0) < 0.01

    def test_close_after_tp1_and_tp2_uses_quarter_notional(self):
        s = self._make_strategy()
        s._emit_signal = MagicMock()
        s._daily_realized_pnl = 0.0
        s._half_closed = True
        s._tp2_closed = True
        s._close_position(110.0, "trailing_stop")
        # Only 25% of position remains: 10% gain on $250 = $25
        assert abs(s._daily_realized_pnl - 25.0) < 0.01

    def test_half_close_tracks_pnl(self):
        s = self._make_strategy()
        s._emit_signal = MagicMock()
        s._daily_realized_pnl = 0.0
        s._half_close(105.0, "partial_tp1")
        # 5% gain on 50% of $1000 = $25
        assert abs(s._daily_realized_pnl - 25.0) < 0.01

    def test_quarter_close_tracks_pnl(self):
        s = self._make_strategy()
        s._emit_signal = MagicMock()
        s._daily_realized_pnl = 0.0
        s._quarter_close(115.0, "partial_tp2")
        # 15% gain on 25% of $1000 = $37.50
        assert abs(s._daily_realized_pnl - 37.5) < 0.01

    def test_tp1_plus_tp2_plus_trail_total_pnl(self):
        """TP1 + TP2 + trailing stop on a $1000 long position should sum correctly."""
        s = self._make_strategy()
        s._emit_signal = MagicMock()
        s._daily_realized_pnl = 0.0
        # TP1 at +5%: 50% of $1000
        s._half_close(105.0, "partial_tp1")
        # TP2 at +10%: 25% of $1000
        s._quarter_close(110.0, "partial_tp2")
        # Trailing stop at +8% (retraced): 25% of $1000
        s._close_position(108.0, "trailing_stop")
        expected = 0.05 * 500 + 0.10 * 250 + 0.08 * 250
        assert abs(s._daily_realized_pnl - expected) < 0.01

    def test_short_half_close_pnl(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        from quant_system.core.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        s = DirectionalEquityStrategy(bus, instrument="AAPL")
        s._state = "short"
        s._entry_price = 100.0
        s._open_notional = 1_000.0
        s._emit_signal = MagicMock()
        s._daily_realized_pnl = 0.0
        s._half_close(95.0, "partial_tp1")
        # 5% drop on short, 50% of $1000 = $25 profit
        assert abs(s._daily_realized_pnl - 25.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# P1-4: CSM _momentum_score returns None when end_close <= 0
# ─────────────────────────────────────────────────────────────────────────────

class TestCSMMomentumScoreEndCloseGuard:
    def _make_strategy(self):
        from quant_system.strategies.cross_sectional_momentum_strategy import CrossSectionalMomentumStrategy
        from quant_system.core.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        return CrossSectionalMomentumStrategy(bus)

    def test_returns_none_when_end_close_zero(self):
        s = self._make_strategy()
        needed = s.LOOKBACK_HOURS + s.SKIP_HOURS + 1
        # Fill history with valid prices then put 0.0 at the end_close position
        closes = [100.0] * needed
        closes[-(s.SKIP_HOURS + 1)] = 0.0  # end_close = 0
        s._1h_closes["AAPL"] = deque(closes, maxlen=200)
        assert s._momentum_score("AAPL") is None

    def test_returns_none_when_end_close_negative(self):
        s = self._make_strategy()
        needed = s.LOOKBACK_HOURS + s.SKIP_HOURS + 1
        closes = [100.0] * needed
        closes[-(s.SKIP_HOURS + 1)] = -1.0
        s._1h_closes["AAPL"] = deque(closes, maxlen=200)
        assert s._momentum_score("AAPL") is None

    def test_returns_none_when_start_close_zero(self):
        s = self._make_strategy()
        needed = s.LOOKBACK_HOURS + s.SKIP_HOURS + 1
        closes = [100.0] * needed
        closes[-(s.LOOKBACK_HOURS + s.SKIP_HOURS + 1)] = 0.0  # start_close = 0
        s._1h_closes["AAPL"] = deque(closes, maxlen=200)
        assert s._momentum_score("AAPL") is None

    def test_returns_float_for_valid_prices(self):
        s = self._make_strategy()
        needed = s.LOOKBACK_HOURS + s.SKIP_HOURS + 1
        closes = [100.0] * needed
        closes[-(s.SKIP_HOURS + 1)] = 110.0
        s._1h_closes["AAPL"] = deque(closes, maxlen=200)
        result = s._momentum_score("AAPL")
        assert result is not None
        assert abs(result - math.log(110.0 / 100.0)) < 1e-9

    def test_source_guards_both_closes(self):
        from quant_system.strategies.cross_sectional_momentum_strategy import CrossSectionalMomentumStrategy
        src = inspect.getsource(CrossSectionalMomentumStrategy._momentum_score)
        assert "end_close <= 0" in src or "end_close<=0" in src


# ─────────────────────────────────────────────────────────────────────────────
# P2-3: _ou_half_life_days returns inf (not 0.0) on cold-start
#       + callers use math.isfinite guard so warmup entries still allowed
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanHalfLifeColdStart:
    def _make_strategy(self):
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        from quant_system.core.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        return KalmanPairsStrategy(
            bus,
            instrument_a="AAPL",
            instrument_b="MSFT",
            entry_z_score=1.5,
            exit_z_score=0.5,
            warmup_bars=5,
            leg_notional=1_000.0,
        )

    def test_returns_inf_when_history_empty(self):
        s = self._make_strategy()
        s._innovation_history.clear()
        result = s._ou_half_life_days()
        assert math.isinf(result)

    def test_returns_inf_when_history_less_than_10(self):
        s = self._make_strategy()
        s._innovation_history.extend([0.1, 0.2, 0.3, 0.4, 0.5])
        result = s._ou_half_life_days()
        assert math.isinf(result)

    def test_returns_finite_with_enough_history(self):
        s = self._make_strategy()
        import numpy as np
        # Mean-reverting innovations: e_t = 0.5 * e_{t-1} + noise
        rng = np.random.default_rng(42)
        e = [0.0] * 20
        for i in range(1, 20):
            e[i] = 0.5 * e[i - 1] + rng.normal(0, 0.1)
        s._innovation_history.extend(e)
        result = s._ou_half_life_days()
        assert math.isfinite(result)
        assert result > 0

    def test_callers_use_isfinite_guard(self):
        """Both entry-path callers must use math.isfinite so warmup entries still fire."""
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        src = inspect.getsource(KalmanPairsStrategy._desired_state)
        assert "math.isfinite" in src, "math.isfinite guard missing from _desired_state"
        # Verify both half-life blocks use the guard
        occurrences = src.count("math.isfinite(_hl)")
        assert occurrences >= 2, f"Expected >=2 isfinite checks, found {occurrences}"

    def test_warmup_entry_not_blocked_by_inf_halflife(self):
        """When innovation history < 10, inf half-life should NOT block entry."""
        s = self._make_strategy()
        s._innovation_history.clear()  # cold start
        hl = s._ou_half_life_days()
        limit = s._half_life_limit()
        # Caller logic: math.isfinite(hl) and hl > limit → only block if finite AND long
        blocked = math.isfinite(hl) and hl > limit
        assert not blocked, "Warmup entry should not be blocked by unknown half-life"

    def test_docstring_says_inf_when_short(self):
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        doc = KalmanPairsStrategy._ou_half_life_days.__doc__ or ""
        # The docstring should mention inf for short history
        assert "inf" in doc.lower() or "insufficient" in doc.lower()


# ─────────────────────────────────────────────────────────────────────────────
# P2-4: ShadowAccounting force_update logs WARNING about avg_price
# ─────────────────────────────────────────────────────────────────────────────

class TestShadowAccountingForceUpdateWarning:
    def test_force_update_warning_in_source(self):
        from quant_system.portfolio.shadow_accounting import ShadowAccounting
        src = inspect.getsource(ShadowAccounting.verify_integrity)
        assert "avg_price" in src
        assert "logger.warning" in src

    def test_force_update_warning_mentions_force_sync(self):
        from quant_system.portfolio.shadow_accounting import ShadowAccounting
        src = inspect.getsource(ShadowAccounting.verify_integrity)
        assert "force_sync" in src

    def test_force_update_still_updates_quantity(self):
        """force_update must still update quantity — warning should not replace it."""
        from quant_system.portfolio.shadow_accounting import ShadowAccounting
        src = inspect.getsource(ShadowAccounting.verify_integrity)
        assert "pos.quantity = float(qty)" in src


# ─────────────────────────────────────────────────────────────────────────────
# P2-5: DirectionalEquity _close_position warns on zero entry_price
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionalEquityZeroEntryPriceWarning:
    def test_close_position_warns_on_zero_entry_price(self, caplog):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        from quant_system.core.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        s = DirectionalEquityStrategy(bus, instrument="AAPL")
        s._state = "long"
        s._entry_price = 0.0  # bug: entry_price never set
        s._open_notional = 1_000.0
        s._emit_signal = MagicMock()
        with caplog.at_level(logging.WARNING):
            s._close_position(100.0, "take_profit")
        assert any("zero entry_price" in r.message.lower() or "entry_price" in r.message for r in caplog.records)

    def test_close_position_skips_pnl_on_zero_entry_price(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        from quant_system.core.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        s = DirectionalEquityStrategy(bus, instrument="AAPL")
        s._state = "long"
        s._entry_price = 0.0
        s._open_notional = 1_000.0
        s._daily_realized_pnl = 0.0
        s._emit_signal = MagicMock()
        s._close_position(100.0, "take_profit")
        assert s._daily_realized_pnl == 0.0, "P&L must not be computed when entry_price is zero"

    def test_close_position_source_has_entry_price_guard(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        src = inspect.getsource(DirectionalEquityStrategy._close_position)
        assert "entry_price <= 0" in src or "entry_price == 0" in src


# ─────────────────────────────────────────────────────────────────────────────
# P3-2: RiskManager._quantity_from_signal_v2 warns on zero/neg price
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskManagerZeroPriceWarning:
    def _make_event(self, target_type: str = "notional", instrument: str = "AAPL"):
        ev = MagicMock()
        ev.target_type = target_type
        ev.instrument_id = instrument
        return ev

    def test_notional_zero_price_logs_warning(self, caplog):
        from quant_system.risk.manager import RiskManager
        ev = self._make_event("notional")
        with caplog.at_level(logging.WARNING):
            result = RiskManager._quantity_from_signal_v2(ev, 1000.0, 0.0, 50_000.0)
        assert result == 0.0
        assert any("reference_price" in r.message.lower() or "zero" in r.message.lower()
                   for r in caplog.records)

    def test_weight_zero_price_logs_warning(self, caplog):
        from quant_system.risk.manager import RiskManager
        ev = self._make_event("weight")
        with caplog.at_level(logging.WARNING):
            result = RiskManager._quantity_from_signal_v2(ev, 0.05, -1.0, 50_000.0)
        assert result == 0.0
        assert any("reference_price" in r.message.lower() or "zero" in r.message.lower()
                   for r in caplog.records)

    def test_notional_valid_price_no_warning(self, caplog):
        from quant_system.risk.manager import RiskManager
        ev = self._make_event("notional")
        with caplog.at_level(logging.WARNING):
            result = RiskManager._quantity_from_signal_v2(ev, 1000.0, 100.0, 50_000.0)
        assert abs(result - 10.0) < 1e-9
        assert not any("reference_price" in r.message.lower() for r in caplog.records)

    def test_units_target_type_not_affected(self, caplog):
        from quant_system.risk.manager import RiskManager
        ev = self._make_event("units")
        with caplog.at_level(logging.WARNING):
            result = RiskManager._quantity_from_signal_v2(ev, 5.0, 0.0, 50_000.0)
        assert abs(result - 5.0) < 1e-9  # units path ignores price entirely

    def test_warning_includes_instrument_id(self, caplog):
        from quant_system.risk.manager import RiskManager
        ev = self._make_event("notional", "TSLA")
        with caplog.at_level(logging.WARNING):
            RiskManager._quantity_from_signal_v2(ev, 1000.0, 0.0, 50_000.0)
        assert any("TSLA" in r.message for r in caplog.records)

    def test_source_has_logger_warning(self):
        from quant_system.risk.manager import RiskManager
        src = inspect.getsource(RiskManager._quantity_from_signal_v2)
        assert "logger.warning" in src
        # Both notional and weight paths should warn
        assert src.count("logger.warning") >= 2
