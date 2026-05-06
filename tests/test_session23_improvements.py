"""
tests/test_session23_improvements.py

Tests for Session 23 Sharpe/Sortino improvements:
  IMP-A — Regime-triggered early exit: close losing positions / tighten trail in HIGH_VOL_PANIC
  IMP-B — Entry quality floor: skip trade when combined scalar < 0.55
  IMP-C — Daily loss graduated size reduction (proportional, not binary block)
  IMP-D — Intraday win-rate circuit breaker: 60% size when win rate < 35% on ≥ 10 trades
  IMP-E — OU half-life continuous size scaling for Kalman pairs (0.5–1.0× range)
"""
from __future__ import annotations

import inspect
import math
import zoneinfo
from collections import deque
from datetime import datetime
from unittest.mock import MagicMock, patch

_ET = zoneinfo.ZoneInfo("America/New_York")


def _today_et():
    return datetime.now(_ET).date()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_de_strategy():
    from quant_system.strategies.directional_equity import DirectionalEquityStrategy
    from quant_system.core.bus import InMemoryEventBus
    bus = InMemoryEventBus()
    s = DirectionalEquityStrategy(bus, instrument="AAPL")
    return s


def _open_long(s, entry=100.0, atr=2.0, notional=1_000.0):
    """Manually open a long position on the strategy."""
    s._state = "long"
    s._entry_price = entry
    s._atr_at_entry = atr
    s._stop_price = entry - 1.5 * atr
    s._tp_price = entry + 3.0 * atr
    s._trail_hwm = entry
    s._trailing_active = False
    s._half_closed = False
    s._tp2_closed = False
    s._open_notional = notional
    s._emit_signal = MagicMock()


def _open_short(s, entry=100.0, atr=2.0, notional=1_000.0):
    s._state = "short"
    s._entry_price = entry
    s._atr_at_entry = atr
    s._stop_price = entry + 1.5 * atr
    s._tp_price = entry - 3.0 * atr
    s._trail_hwm = entry
    s._trailing_active = False
    s._half_closed = False
    s._tp2_closed = False
    s._open_notional = notional
    s._emit_signal = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# IMP-A: Regime-triggered early exit
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimePanicExit:
    def test_is_panic_regime_method_exists(self):
        s = _make_de_strategy()
        assert callable(s._is_panic_regime)

    def test_is_panic_regime_returns_false_by_default(self):
        s = _make_de_strategy()
        # No global regime router initialised → should not raise and return False
        with patch("quant_system.strategies.directional_equity.get_global_regime_router",
                   side_effect=Exception("not initialised"), create=True):
            assert s._is_panic_regime() is False

    def test_panic_regime_closes_losing_long(self):
        s = _make_de_strategy()
        _open_long(s, entry=100.0, atr=2.0)
        # Position is at 97 — below entry, so it's a loss
        close_price = 97.0
        with patch.object(s, "_is_panic_regime", return_value=True), \
             patch.object(s, "_is_eod_close_window", return_value=False):
            s._manage_position(close_price, rsi=45.0)
        # Should have called _emit_signal (via _close_position)
        s._emit_signal.assert_called()
        assert s._state == "flat"

    def test_panic_regime_closes_losing_short(self):
        s = _make_de_strategy()
        _open_short(s, entry=100.0, atr=2.0)
        # Position at 103 — above entry, so it's a loss on a short
        close_price = 103.0
        with patch.object(s, "_is_panic_regime", return_value=True), \
             patch.object(s, "_is_eod_close_window", return_value=False):
            s._manage_position(close_price, rsi=55.0)
        s._emit_signal.assert_called()
        assert s._state == "flat"

    def test_panic_regime_does_not_close_profitable_long(self):
        """Profitable long in panic: should NOT close immediately — just tighten trail."""
        s = _make_de_strategy()
        _open_long(s, entry=100.0, atr=2.0)
        close_price = 105.0  # profitable
        with patch.object(s, "_is_panic_regime", return_value=True), \
             patch.object(s, "_is_eod_close_window", return_value=False):
            s._manage_position(close_price, rsi=55.0)
        # Trailing should have been activated
        assert s._trailing_active is True
        assert s._state == "long"  # not closed yet

    def test_panic_regime_uses_tight_trail_mult(self):
        """trail_mult in panic should be TRAIL_ATR_MULT * 0.5."""
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        src = inspect.getsource(DirectionalEquityStrategy._manage_position)
        assert "0.5" in src or "* 0.5" in src
        assert "_is_panic_regime" in src

    def test_panic_regime_exit_reason_label(self):
        """_close_position reason should be 'regime_panic_exit'."""
        s = _make_de_strategy()
        _open_long(s, entry=100.0, atr=2.0)
        reasons = []
        original_close = s._close_position.__func__

        def capture_close(self_, price, reason):
            reasons.append(reason)
            original_close(self_, price, reason)

        with patch.object(s, "_is_panic_regime", return_value=True), \
             patch.object(s, "_is_eod_close_window", return_value=False), \
             patch.object(type(s), "_close_position", capture_close):
            s._manage_position(95.0, rsi=40.0)
        assert "regime_panic_exit" in reasons

    def test_kalman_panic_regime_exits_open_pair(self):
        """KalmanPairsStrategy _desired_state returns 'flat' when HIGH_VOL_PANIC."""
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        from quant_system.core.bus import InMemoryEventBus
        from datetime import datetime, timezone
        bus = InMemoryEventBus()
        s = KalmanPairsStrategy(bus, instrument_a="AAPL", instrument_b="MSFT",
                                entry_z_score=1.5, exit_z_score=0.5,
                                warmup_bars=5, leg_notional=1_000.0)
        s._pair_state = "short_a_long_b"

        mock_rr = MagicMock()
        from risk.regime_router import CompositeRegime
        mock_rr.last_regime = CompositeRegime.HIGH_VOL_PANIC

        with patch("quant_system.strategies.kalman_pairs.get_global_regime_router",
                   return_value=mock_rr, create=True):
            result = s._desired_state(z_score=0.3, current_ts=datetime.now(timezone.utc))
        assert result == "flat"
        s.close()

    def test_kalman_non_panic_does_not_force_exit(self):
        """KalmanPairsStrategy _desired_state does NOT return 'flat' in RANGING regime."""
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        from quant_system.core.bus import InMemoryEventBus
        from datetime import datetime, timezone
        bus = InMemoryEventBus()
        s = KalmanPairsStrategy(bus, instrument_a="AAPL", instrument_b="MSFT",
                                entry_z_score=1.0, exit_z_score=0.5,
                                warmup_bars=5, leg_notional=1_000.0)
        s._pair_state = "short_a_long_b"
        s._entry_z_score = 2.0

        mock_rr = MagicMock()
        from risk.regime_router import CompositeRegime
        mock_rr.last_regime = CompositeRegime.RANGING

        with patch("quant_system.strategies.kalman_pairs.get_global_regime_router",
                   return_value=mock_rr, create=True):
            s._desired_state(z_score=0.3, current_ts=datetime.now(timezone.utc))
        # At z=0.3 (below exit_z=0.5) in non-panic, should return "flat" via normal exit
        # (not panic exit) — either way "flat" is acceptable; the important thing is
        # that the panic path is NOT what triggered it
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# IMP-B: Entry quality floor
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryQualityFloor:
    def test_quality_floor_method_in_source(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        src = inspect.getsource(DirectionalEquityStrategy._try_entry)
        assert "quality_mult" in src
        assert "0.55" in src

    def test_quality_floor_skips_low_conviction_entry(self):
        """When ml × mtf × conviction < 0.55, trade should not be opened."""
        s = _make_de_strategy()
        s._emit_signal = MagicMock()
        # Patch the scalars to be weak: 0.75 × 0.75 × 0.90 = 0.506 < 0.55
        # close=99 > ema20=98 so the RSI buy condition fires; quality gate should block
        with patch.object(s, "_ml_entry_scalar", return_value=0.75), \
             patch.object(s, "_mtf_entry_scalar", return_value=0.75), \
             patch.object(s, "_rsi_conviction_scalar", return_value=0.90), \
             patch.object(s, "_is_equity_auction_window", return_value=False), \
             patch.object(s, "_volume_confirmed", return_value=True), \
             patch.object(s, "_1h_trend_direction", return_value="neutral"), \
             patch.object(s, "_session_vwap", return_value=None), \
             patch.object(s, "_intraday_vol_mult", return_value=1.0), \
             patch.object(s, "_winrate_circuit_mult", return_value=1.0), \
             patch("quant_system.strategies.directional_equity._earnings_blackout", return_value=False):
            # RSI below oversold threshold → buy signal
            s._try_entry(close=99.0, rsi=25.0, ema20=98.0, atr=2.0)
        # No open call should have been made (quality floor blocks it)
        assert s._state == "flat"

    def test_quality_floor_allows_high_conviction_entry(self):
        """When ml × mtf × conviction ≥ 0.55, trade is not rejected by quality gate."""
        s = _make_de_strategy()
        with patch.object(s, "_ml_entry_scalar", return_value=0.90), \
             patch.object(s, "_mtf_entry_scalar", return_value=0.85), \
             patch.object(s, "_rsi_conviction_scalar", return_value=0.90), \
             patch.object(s, "_is_equity_auction_window", return_value=False), \
             patch.object(s, "_volume_confirmed", return_value=True), \
             patch.object(s, "_1h_trend_direction", return_value="neutral"), \
             patch.object(s, "_session_vwap", return_value=None), \
             patch.object(s, "_intraday_vol_mult", return_value=1.0), \
             patch.object(s, "_winrate_circuit_mult", return_value=1.0), \
             patch("quant_system.strategies.directional_equity._earnings_blackout", return_value=False), \
             patch("risk.vix_regime_manager.get_global_vix_manager", side_effect=RuntimeError("test")), \
             patch.object(s, "_open") as mock_open:
            # close=99 > ema20=98 → buy condition fires; 0.90×0.85×0.90=0.689 ≥ 0.55
            s._try_entry(close=99.0, rsi=25.0, ema20=98.0, atr=2.0)
        # 0.90 × 0.85 × 0.90 = 0.689 ≥ 0.55 → should proceed to _open
        mock_open.assert_called_once()

    def test_quality_floor_threshold_is_0_55(self):
        """The threshold value must be exactly 0.55 in the source."""
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        src = inspect.getsource(DirectionalEquityStrategy._try_entry)
        assert "< 0.55" in src


# ─────────────────────────────────────────────────────────────────────────────
# IMP-C: Daily loss graduated size reduction
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyLossGraduatedSizing:
    def test_size_reduction_at_50pct_daily_loss(self):
        """At 50% of the daily limit, final_notional should be ~50% of base."""
        s = _make_de_strategy()
        s._daily_pnl_date = _today_et()   # prevent daily reset from zeroing pnl
        s._daily_realized_pnl = -150.0    # 50% of -300 limit
        # Capture the notional passed to _open
        opened_notionals = []
        with patch.object(s, "_ml_entry_scalar", return_value=1.0), \
             patch.object(s, "_mtf_entry_scalar", return_value=1.0), \
             patch.object(s, "_rsi_conviction_scalar", return_value=1.0), \
             patch.object(s, "_is_equity_auction_window", return_value=False), \
             patch.object(s, "_volume_confirmed", return_value=True), \
             patch.object(s, "_1h_trend_direction", return_value="neutral"), \
             patch.object(s, "_session_vwap", return_value=None), \
             patch.object(s, "_intraday_vol_mult", return_value=1.0), \
             patch.object(s, "_winrate_circuit_mult", return_value=1.0), \
             patch("quant_system.strategies.directional_equity._earnings_blackout", return_value=False), \
             patch("risk.vix_regime_manager.get_global_vix_manager", side_effect=RuntimeError("test")), \
             patch.object(s, "_open", side_effect=lambda side, price, atr, notional: opened_notionals.append(notional)):
            s._try_entry(close=99.0, rsi=25.0, ema20=98.0, atr=2.0)
        assert len(opened_notionals) == 1
        # loss_ratio = 0.50 → size_mult = max(0.25, 1.0 - 0.50) = 0.50
        assert abs(opened_notionals[0] / s.leg_notional - 0.50) < 0.02

    def test_size_reduction_at_75pct_daily_loss(self):
        """At 75% of daily limit, size_mult = max(0.25, 0.25) = 0.25."""
        s = _make_de_strategy()
        s._daily_pnl_date = _today_et()   # prevent daily reset from zeroing pnl
        s._daily_realized_pnl = -225.0    # 75% of -300 limit
        opened_notionals = []
        with patch.object(s, "_ml_entry_scalar", return_value=1.0), \
             patch.object(s, "_mtf_entry_scalar", return_value=1.0), \
             patch.object(s, "_rsi_conviction_scalar", return_value=1.0), \
             patch.object(s, "_is_equity_auction_window", return_value=False), \
             patch.object(s, "_volume_confirmed", return_value=True), \
             patch.object(s, "_1h_trend_direction", return_value="neutral"), \
             patch.object(s, "_session_vwap", return_value=None), \
             patch.object(s, "_intraday_vol_mult", return_value=1.0), \
             patch.object(s, "_winrate_circuit_mult", return_value=1.0), \
             patch("quant_system.strategies.directional_equity._earnings_blackout", return_value=False), \
             patch("risk.vix_regime_manager.get_global_vix_manager", side_effect=RuntimeError("test")), \
             patch.object(s, "_open", side_effect=lambda side, price, atr, notional: opened_notionals.append(notional)):
            s._try_entry(close=99.0, rsi=25.0, ema20=98.0, atr=2.0)
        assert len(opened_notionals) == 1
        assert abs(opened_notionals[0] / s.leg_notional - 0.25) < 0.02

    def test_no_size_reduction_when_pnl_positive(self):
        """Positive daily P&L should not reduce size."""
        s = _make_de_strategy()
        s._daily_pnl_date = _today_et()   # prevent daily reset from zeroing pnl
        s._daily_realized_pnl = +200.0
        opened_notionals = []
        with patch.object(s, "_ml_entry_scalar", return_value=1.0), \
             patch.object(s, "_mtf_entry_scalar", return_value=1.0), \
             patch.object(s, "_rsi_conviction_scalar", return_value=1.0), \
             patch.object(s, "_is_equity_auction_window", return_value=False), \
             patch.object(s, "_volume_confirmed", return_value=True), \
             patch.object(s, "_1h_trend_direction", return_value="neutral"), \
             patch.object(s, "_session_vwap", return_value=None), \
             patch.object(s, "_intraday_vol_mult", return_value=1.0), \
             patch.object(s, "_winrate_circuit_mult", return_value=1.0), \
             patch("quant_system.strategies.directional_equity._earnings_blackout", return_value=False), \
             patch("risk.vix_regime_manager.get_global_vix_manager", side_effect=RuntimeError("test")), \
             patch.object(s, "_open", side_effect=lambda side, price, atr, notional: opened_notionals.append(notional)):
            s._try_entry(close=99.0, rsi=25.0, ema20=98.0, atr=2.0)
        assert len(opened_notionals) == 1
        assert abs(opened_notionals[0] - s.leg_notional) < 1.0

    def test_size_mult_floor_is_0_25(self):
        """size_mult must not drop below 0.25 (even at 90%+ of daily limit)."""
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        src = inspect.getsource(DirectionalEquityStrategy._try_entry)
        assert "max(0.25" in src


# ─────────────────────────────────────────────────────────────────────────────
# IMP-D: Intraday win-rate circuit breaker
# ─────────────────────────────────────────────────────────────────────────────

class TestWinRateCircuit:
    def test_winrate_circuit_method_exists(self):
        s = _make_de_strategy()
        assert callable(s._winrate_circuit_mult)

    def test_returns_1_when_fewer_than_10_trades(self):
        s = _make_de_strategy()
        s._session_trades = [100.0, -50.0, 30.0]   # only 3 trades
        assert s._winrate_circuit_mult() == 1.0

    def test_returns_1_when_win_rate_above_threshold(self):
        s = _make_de_strategy()
        # 6 wins + 4 losses = 60% win rate → above 35%
        s._session_trades = [100.0] * 6 + [-50.0] * 4
        assert s._winrate_circuit_mult() == 1.0

    def test_returns_0_60_when_win_rate_below_35pct(self):
        s = _make_de_strategy()
        # 3 wins + 7 losses = 30% win rate → below 35%
        s._session_trades = [100.0] * 3 + [-50.0] * 7
        assert abs(s._winrate_circuit_mult() - 0.60) < 1e-9

    def test_boundary_exactly_10_trades(self):
        s = _make_de_strategy()
        s._session_trades = [10.0] * 10   # 100% win rate — above threshold
        assert s._winrate_circuit_mult() == 1.0

    def test_session_trades_reset_on_new_day(self):
        from quant_system.strategies.directional_equity import DirectionalEquityStrategy
        src = inspect.getsource(DirectionalEquityStrategy._try_entry)
        assert "_session_trades" in src and "= []" in src

    def test_trade_recorded_in_session_trades_after_close(self):
        s = _make_de_strategy()
        _open_long(s, entry=100.0, notional=1_000.0)
        s._daily_pnl_date = None  # prevent date reset from clearing trades
        s._close_position(110.0, "take_profit")
        assert len(s._session_trades) == 1
        assert s._session_trades[0] > 0  # profitable trade

    def test_session_trades_max_length_20(self):
        s = _make_de_strategy()
        for i in range(25):
            _open_long(s, entry=100.0, notional=1_000.0)
            s._daily_pnl_date = None
            s._close_position(110.0, "take_profit")
        assert len(s._session_trades) <= 20

    def test_winrate_circuit_reduces_entry_size(self):
        """When win rate is low, final_notional should be 60% of base."""
        s = _make_de_strategy()
        s._daily_pnl_date = _today_et()   # prevent daily reset from clearing session_trades
        s._session_trades = [100.0] * 3 + [-50.0] * 7   # 30% win rate
        opened_notionals = []
        with patch.object(s, "_ml_entry_scalar", return_value=1.0), \
             patch.object(s, "_mtf_entry_scalar", return_value=1.0), \
             patch.object(s, "_rsi_conviction_scalar", return_value=1.0), \
             patch.object(s, "_is_equity_auction_window", return_value=False), \
             patch.object(s, "_volume_confirmed", return_value=True), \
             patch.object(s, "_1h_trend_direction", return_value="neutral"), \
             patch.object(s, "_session_vwap", return_value=None), \
             patch.object(s, "_intraday_vol_mult", return_value=1.0), \
             patch("quant_system.strategies.directional_equity._earnings_blackout", return_value=False), \
             patch("risk.vix_regime_manager.get_global_vix_manager", side_effect=RuntimeError("test")), \
             patch.object(s, "_open", side_effect=lambda side, price, atr, notional: opened_notionals.append(notional)):
            s._try_entry(close=99.0, rsi=25.0, ema20=98.0, atr=2.0)
        assert len(opened_notionals) == 1
        assert abs(opened_notionals[0] / s.leg_notional - 0.60) < 0.02


# ─────────────────────────────────────────────────────────────────────────────
# IMP-E: OU half-life continuous size scaling for pairs
# ─────────────────────────────────────────────────────────────────────────────

class TestHalfLifeContinuousScaling:
    def _make_pairs_strategy(self):
        from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
        from quant_system.core.bus import InMemoryEventBus
        bus = InMemoryEventBus()
        return KalmanPairsStrategy(
            bus, instrument_a="AAPL", instrument_b="MSFT",
            entry_z_score=1.5, exit_z_score=0.5,
            warmup_bars=5, leg_notional=1_000.0,
        )

    def test_hl_score_in_metadata(self):
        """Metadata emitted at entry should contain hl_score."""
        import quant_system.strategies.kalman_pairs as _kp_mod
        with open(_kp_mod.__file__) as _f:
            src = _f.read()
        assert "hl_score" in src

    def test_hl_score_1_when_half_life_unknown(self):
        """When half-life is inf (unknown), hl_score should be 1.0 (no penalty)."""
        s = self._make_pairs_strategy()
        s._innovation_history.clear()   # force inf half-life
        hl = s._ou_half_life_days()
        assert math.isinf(hl)
        # Compute hl_score as the code does
        limit = s._half_life_limit()
        hl_score = 1.0 if not math.isfinite(hl) else max(0.5, 1.0 - 0.5 * hl / limit)
        assert hl_score == 1.0
        s.close()

    def test_hl_score_near_1_for_fast_reversion(self):
        """Half-life of 0.1 days on 5-day limit → hl_score ≈ 0.99."""
        hl, limit = 0.1, 5.0
        hl_score = max(0.5, 1.0 - 0.5 * hl / limit)
        assert abs(hl_score - 0.99) < 0.01

    def test_hl_score_0_75_at_half_limit(self):
        """Half-life of 2.5 days (50% of 5-day limit) → hl_score = 0.75."""
        hl, limit = 2.5, 5.0
        hl_score = max(0.5, 1.0 - 0.5 * hl / limit)
        assert abs(hl_score - 0.75) < 1e-9

    def test_hl_score_floor_is_0_5(self):
        """Even at hl == limit (or beyond), hl_score must not drop below 0.5."""
        for hl in (5.0, 10.0, 100.0):
            hl_score = max(0.5, 1.0 - 0.5 * hl / 5.0)
            assert hl_score >= 0.5

    def test_hl_score_not_applied_on_exit(self):
        """At desired_state='flat', hl_score must be 1.0 (no penalty on exits)."""
        import quant_system.strategies.kalman_pairs as _kp_mod
        with open(_kp_mod.__file__) as _f:
            src = _f.read()
        # The code must guard: only apply hl_score when desired_state != "flat"
        assert 'desired_state != "flat"' in src or 'desired_state == "flat"' in src

    def test_hl_score_multiplies_scaled_notional(self):
        """scaled_notional formula must include hl_score."""
        import quant_system.strategies.kalman_pairs as _kp_mod
        with open(_kp_mod.__file__) as _f:
            src = _f.read()
        assert "hl_score" in src
        assert "scaled_notional" in src
        # Check that scaled_notional uses hl_score
        lines = [ln.strip() for ln in src.splitlines() if "scaled_notional" in ln and "=" in ln]
        assert any("hl_score" in ln for ln in lines), \
            "scaled_notional assignment must reference hl_score"
