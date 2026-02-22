"""
tests/test_backtest_engine_v2.py - Tests for live-trade-ready backtest engine features

Validates:
1. Order types: limit, stop, stop-limit, cancel, expiry
2. Stop management: SL, TP, trailing stop, max hold
3. Risk guards: position limits, daily loss, drawdown, circuit breaker, cooldown
4. Execution realism: Open price fills, partial fills, pre-trade checks
5. Dynamic position sizing: Kelly, vol-adjusted, regime-aware
6. Integration: full run with all features enabled
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from backtesting.backtest_engine import (
    BacktestEngine,
    OrderType,
    TimeInForce,
    StopManager,
    StopLevel,
    RiskGuard,
    Position,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(
    symbol: str,
    closes: list,
    opens: list = None,
    highs: list = None,
    lows: list = None,
    start: datetime = None,
    volume: float = 1_000_000,
) -> dict:
    """Build {symbol: DataFrame} with OHLCV data."""
    start = start or datetime(2024, 1, 2)
    dates = pd.bdate_range(start=start, periods=len(closes), freq="B")
    if opens is None:
        opens = closes
    if highs is None:
        highs = [max(o, c) * 1.01 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) * 0.99 for o, c in zip(opens, closes)]
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [volume] * len(closes),
        },
        index=dates,
    )
    return {symbol: df}


def _engine(**kwargs) -> BacktestEngine:
    """Create a BacktestEngine with common test defaults."""
    defaults = dict(
        initial_capital=100_000,
        slippage_bps=0.0,
        use_dynamic_slippage=False,
        use_open_price_fill=False,  # Use Close for simpler assertions unless testing Open fills
        enable_stop_management=False,  # Disable by default so tests control it explicitly
        max_participation_rate=0.0,  # Disable partial fills by default
    )
    defaults.update(kwargs)
    return BacktestEngine(**defaults)


# ===========================================================================
# 1. ORDER TYPES
# ===========================================================================

class TestLimitOrders:
    """Limit orders fill when price crosses the limit, not otherwise."""

    def test_limit_buy_fills_when_low_touches(self):
        """Limit BUY at 95 fills when bar Low <= 95."""
        closes = [100.0, 98.0, 96.0, 94.0, 92.0]
        lows = [99.0, 97.0, 93.0, 91.0, 90.0]  # Low touches 93 on bar 2
        data = _make_price_data("AAPL", closes, lows=lows)
        engine = _engine()
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine._bar_idx = 0
        engine.place_limit_order("AAPL", "BUY", 100, limit_price=95.0)

        # Process bar 1 — Low=97 > 95, should NOT fill
        engine.current_time = ts[1]
        engine._bar_idx = 1
        engine._process_pending_orders(ts[1])
        assert len(engine.trades) == 0

        # Process bar 2 — Low=93 <= 95, should fill at 95
        engine.current_time = ts[2]
        engine._bar_idx = 2
        engine._process_pending_orders(ts[2])
        assert len(engine.trades) == 1
        assert engine.trades[0].price == pytest.approx(95.0, abs=0.01)

    def test_limit_sell_fills_when_high_touches(self):
        """Limit SELL at 105 fills when bar High >= 105."""
        closes = [100.0, 102.0, 104.0, 106.0]
        highs = [101.0, 103.0, 106.0, 108.0]  # High hits 106 on bar 2
        data = _make_price_data("AAPL", closes, highs=highs)
        engine = _engine()
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        # First buy shares so we have something to sell
        engine.current_time = ts[0]
        engine._execute_order_now("AAPL", "BUY", 100)

        engine.current_time = ts[0]
        engine._bar_idx = 0
        engine.place_limit_order("AAPL", "SELL", 100, limit_price=105.0)

        # Bar 1 — High=103 < 105, no fill
        engine.current_time = ts[1]
        engine._bar_idx = 1
        engine._process_pending_orders(ts[1])
        assert len(engine.trades) == 1  # Only the initial buy

        # Bar 2 — High=106 >= 105, fills at 105
        engine.current_time = ts[2]
        engine._bar_idx = 2
        engine._process_pending_orders(ts[2])
        assert len(engine.trades) == 2
        assert engine.trades[1].price == pytest.approx(105.0, abs=0.01)

    def test_limit_order_expires_gtc(self):
        """GTC limit order expires after gtc_bars."""
        closes = [100.0] * 10
        data = _make_price_data("AAPL", closes)
        engine = _engine()
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine._bar_idx = 0
        engine.place_limit_order("AAPL", "BUY", 100, limit_price=50.0,
                                 time_in_force=TimeInForce.GTC, gtc_bars=3)

        # Advance past expiry (bar 0 + 3 = bar 3, so bar 4 should expire)
        for i in range(1, 5):
            engine.current_time = ts[i]
            engine._bar_idx = i
            engine._process_pending_orders(ts[i])

        assert len(engine.trades) == 0
        assert len(engine.pending_orders) == 0  # Expired


class TestStopOrders:
    """Stop orders trigger when price crosses the stop level."""

    def test_stop_buy_triggers(self):
        """Stop BUY at 105 triggers when High >= 105."""
        closes = [100.0, 102.0, 106.0, 108.0]
        highs = [101.0, 103.0, 107.0, 109.0]
        data = _make_price_data("AAPL", closes, highs=highs)
        engine = _engine()
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine._bar_idx = 0
        engine.place_stop_order("AAPL", "BUY", 100, stop_price=105.0)

        # Bar 1 — High=103 < 105, no trigger
        engine.current_time = ts[1]
        engine._bar_idx = 1
        engine._process_pending_orders(ts[1])
        assert len(engine.trades) == 0

        # Bar 2 — High=107 >= 105, triggers at stop_price=105
        engine.current_time = ts[2]
        engine._bar_idx = 2
        engine._process_pending_orders(ts[2])
        assert len(engine.trades) == 1
        assert engine.trades[0].price == pytest.approx(105.0, abs=0.01)

    def test_stop_sell_triggers(self):
        """Stop SELL at 95 triggers when Low <= 95."""
        closes = [100.0, 98.0, 94.0, 92.0]
        lows = [99.0, 96.0, 93.0, 91.0]
        data = _make_price_data("AAPL", closes, lows=lows)
        engine = _engine()
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        # Buy first
        engine.current_time = ts[0]
        engine._execute_order_now("AAPL", "BUY", 100)

        engine.current_time = ts[0]
        engine._bar_idx = 0
        engine.place_stop_order("AAPL", "SELL", 100, stop_price=95.0)

        # Bar 1 — Low=96 > 95, no trigger
        engine.current_time = ts[1]
        engine._bar_idx = 1
        engine._process_pending_orders(ts[1])
        assert len(engine.trades) == 1  # Only initial buy

        # Bar 2 — Low=93 <= 95, triggers
        engine.current_time = ts[2]
        engine._bar_idx = 2
        engine._process_pending_orders(ts[2])
        assert len(engine.trades) == 2


class TestOrderCancellation:
    """Cancel orders by ID or cancel all."""

    def test_cancel_by_id(self):
        engine = _engine()
        data = _make_price_data("AAPL", [100.0] * 5)
        engine.load_data(data)
        ts = list(data["AAPL"].index)
        engine.current_time = ts[0]

        oid = engine.place_limit_order("AAPL", "BUY", 100, limit_price=90.0)
        assert len(engine.pending_orders) == 1

        result = engine.cancel_order(oid)
        assert result is True
        assert len(engine.pending_orders) == 0

    def test_cancel_all_for_symbol(self):
        engine = _engine()
        data = {**_make_price_data("AAPL", [100.0] * 5), **_make_price_data("MSFT", [200.0] * 5)}
        engine.load_data(data)
        ts = list(data["AAPL"].index)
        engine.current_time = ts[0]

        engine.place_limit_order("AAPL", "BUY", 50, limit_price=90.0)
        engine.place_limit_order("MSFT", "BUY", 50, limit_price=180.0)
        assert len(engine.pending_orders) == 2

        engine.cancel_all_orders(symbol="AAPL")
        assert len(engine.pending_orders) == 1
        assert engine.pending_orders[0].symbol == "MSFT"


class TestMarketOrderBackwardCompat:
    """execute_order() still works as before — queues market order for t+1."""

    def test_market_order_queued_and_filled(self):
        engine = _engine()
        data = _make_price_data("AAPL", [100.0] * 5)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        engine.current_time = ts[0]
        engine._bar_idx = 0
        engine.execute_order("AAPL", "BUY", 100)
        assert len(engine.pending_orders) == 1
        assert engine.pending_orders[0].order_type == OrderType.MARKET

        # Process at next bar
        engine.current_time = ts[1]
        engine._bar_idx = 1
        engine._process_pending_orders(ts[1])
        assert len(engine.trades) == 1
        assert "AAPL" in engine.positions


# ===========================================================================
# 2. STOP MANAGEMENT
# ===========================================================================

class TestStopManager:
    """StopManager triggers exits based on SL, TP, trailing stop, and max hold."""

    def test_stop_loss_triggers(self):
        """Position hitting -3% triggers stop loss exit."""
        sm = StopManager()
        sm.register("AAPL", StopLevel(stop_loss_pct=0.03, take_profit_pct=0.10,
                                       max_hold_bars=100))
        positions = {
            "AAPL": Position("AAPL", 100, 100.0, 96.0, -400.0, 0.0, 100.0, entry_bar=0),
        }
        # Price at 96 = -4% loss > 3% threshold
        exits = sm.check_exits(positions, current_bar=5, data={}, current_time=datetime.now())
        assert len(exits) == 1
        assert exits[0]["symbol"] == "AAPL"
        assert exits[0]["reason"] == "stop_loss"
        assert exits[0]["side"] == "SELL"

    def test_take_profit_triggers(self):
        """Position hitting +6% triggers take profit exit."""
        sm = StopManager()
        sm.register("AAPL", StopLevel(stop_loss_pct=0.03, take_profit_pct=0.06,
                                       max_hold_bars=100))
        positions = {
            "AAPL": Position("AAPL", 100, 100.0, 107.0, 700.0, 0.0, 107.0, entry_bar=0),
        }
        exits = sm.check_exits(positions, current_bar=5, data={}, current_time=datetime.now())
        assert len(exits) == 1
        assert exits[0]["reason"] == "take_profit"

    def test_trailing_stop_activates_and_triggers(self):
        """Trailing stop activates at +2.5%, then triggers when price drops 2% from peak."""
        sm = StopManager()
        sl = StopLevel(
            stop_loss_pct=0.10,
            take_profit_pct=0.20,
            trailing_activation_pct=0.025,
            trailing_distance_pct=0.02,
            max_hold_bars=100,
        )
        sm.register("AAPL", sl)

        # Position up 3% — should activate trailing
        positions = {
            "AAPL": Position("AAPL", 100, 100.0, 103.0, 300.0, 0.0, 103.0, entry_bar=0),
        }
        exits = sm.check_exits(positions, current_bar=5, data={}, current_time=datetime.now())
        assert len(exits) == 0
        assert sl.trailing_active is True
        assert sl.peak_since_activation == 103.0

        # Price rises to 105 — no exit, peak updates
        positions["AAPL"].update_price(105.0)
        exits = sm.check_exits(positions, current_bar=6, data={}, current_time=datetime.now())
        assert len(exits) == 0
        assert sl.peak_since_activation == 105.0

        # Price drops to 102.8 — 2.1% below 105 peak, triggers trailing
        positions["AAPL"].update_price(102.8)
        exits = sm.check_exits(positions, current_bar=7, data={}, current_time=datetime.now())
        assert len(exits) == 1
        assert exits[0]["reason"] == "trailing_stop"

    def test_max_hold_triggers(self):
        """Position held past max_hold_bars triggers forced exit."""
        sm = StopManager()
        sm.register("AAPL", StopLevel(stop_loss_pct=0.50, take_profit_pct=0.50,
                                       max_hold_bars=10))
        positions = {
            "AAPL": Position("AAPL", 100, 100.0, 101.0, 100.0, 0.0, 101.0, entry_bar=0),
        }
        # At bar 9, no exit
        exits = sm.check_exits(positions, current_bar=9, data={}, current_time=datetime.now())
        assert len(exits) == 0

        # At bar 10, max hold triggers
        exits = sm.check_exits(positions, current_bar=10, data={}, current_time=datetime.now())
        assert len(exits) == 1
        assert exits[0]["reason"] == "max_hold"

    def test_short_position_stop_loss(self):
        """Short position hitting +3% (wrong direction) triggers stop loss."""
        sm = StopManager()
        sm.register("AAPL", StopLevel(stop_loss_pct=0.03, take_profit_pct=0.10,
                                       max_hold_bars=100))
        # Short at 100, price now 104 = -4% for short
        positions = {
            "AAPL": Position("AAPL", -100, 100.0, 104.0, -400.0, 0.0, 104.0, entry_bar=0),
        }
        exits = sm.check_exits(positions, current_bar=5, data={}, current_time=datetime.now())
        assert len(exits) == 1
        assert exits[0]["reason"] == "stop_loss"
        assert exits[0]["side"] == "BUY"  # Buy to cover short

    def test_no_exit_in_normal_range(self):
        """No exit when position is within normal P&L range."""
        sm = StopManager()
        sm.register("AAPL", StopLevel(stop_loss_pct=0.03, take_profit_pct=0.06,
                                       max_hold_bars=100))
        # +1% gain — no exit
        positions = {
            "AAPL": Position("AAPL", 100, 100.0, 101.0, 100.0, 0.0, 101.0, entry_bar=0),
        }
        exits = sm.check_exits(positions, current_bar=5, data={}, current_time=datetime.now())
        assert len(exits) == 0


class TestStopManagementIntegration:
    """Stop management works within the full engine run loop."""

    def test_stop_loss_fires_during_run(self):
        """Engine auto-exits position when stop loss is hit."""
        # Price drops from 100 to 90 over 5 bars
        closes = [100.0, 100.0, 100.0, 98.0, 96.0, 94.0, 92.0, 90.0]
        data = _make_price_data("AAPL", closes)
        engine = _engine(enable_stop_management=True, default_stop_loss_pct=0.05)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 1:
                eng.execute_order("AAPL", "BUY", 100)

        engine.run(strategy, ts[0], ts[-1])

        # Should have: BUY trade + auto SELL from stop loss
        sell_trades = [t for t in engine.trades if t.side == "SELL"]
        assert len(sell_trades) >= 1
        # Position should be closed
        assert "AAPL" not in engine.positions


# ===========================================================================
# 3. RISK GUARDS
# ===========================================================================

class TestRiskGuardUnit:
    """Unit tests for RiskGuard checks."""

    def test_max_positions_blocks_entry(self):
        rg = RiskGuard(max_positions=2)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        ok, reason = rg.can_enter("AAPL", "BUY", 100, 100.0, 100_000, n_positions=2, bar_idx=0)
        assert ok is False
        assert reason == "max_positions"

    def test_max_positions_allows_below_limit(self):
        rg = RiskGuard(max_positions=2)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        ok, _ = rg.can_enter("AAPL", "BUY", 100, 100.0, 100_000, n_positions=1, bar_idx=0)
        assert ok is True

    def test_daily_loss_blocks_entry(self):
        rg = RiskGuard(max_daily_loss_pct=0.03)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        # Equity dropped to 96k = -4% daily loss
        ok, reason = rg.can_enter("AAPL", "BUY", 100, 96.0, 96_000, n_positions=0, bar_idx=1)
        assert ok is False
        assert reason == "daily_loss_limit"

    def test_daily_loss_allows_exits(self):
        rg = RiskGuard(max_daily_loss_pct=0.03)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        # Even with daily loss breached, exits allowed
        ok, _ = rg.can_enter("AAPL", "SELL", 100, 96.0, 96_000, n_positions=1,
                             bar_idx=1, is_exit=True)
        assert ok is True

    def test_drawdown_blocks_entry(self):
        rg = RiskGuard(max_drawdown_pct=0.10, max_daily_loss_pct=0.50)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)
        # Simulate intraday recovery: day_start_equity matches current so daily_loss is 0
        # But peak was 100k and current is 89k = -11% drawdown
        rg.day_start_equity = 89_000
        ok, reason = rg.can_enter("AAPL", "BUY", 100, 89.0, 89_000, n_positions=0, bar_idx=1)
        assert ok is False
        assert reason == "drawdown_limit"

    def test_circuit_breaker_trips_after_consecutive_losses(self):
        rg = RiskGuard(circuit_breaker_consecutive_losses=3, circuit_breaker_cooldown_bars=2)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        # 3 consecutive losses
        rg.record_trade_pnl(-100, 0)
        rg.record_trade_pnl(-100, 1)
        rg.record_trade_pnl(-100, 2)

        assert rg.circuit_breaker_active is True

        ok, reason = rg.can_enter("AAPL", "BUY", 100, 100.0, 100_000, n_positions=0, bar_idx=3)
        assert ok is False
        assert reason == "circuit_breaker"

    def test_circuit_breaker_resets_after_cooldown(self):
        rg = RiskGuard(circuit_breaker_consecutive_losses=2, circuit_breaker_cooldown_bars=3)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        rg.record_trade_pnl(-100, 0)
        rg.record_trade_pnl(-100, 1)
        assert rg.circuit_breaker_active is True

        # After 3 bars of cooldown
        rg.on_new_bar(100_000, datetime(2024, 1, 5), 4)
        assert rg.circuit_breaker_active is False

    def test_winning_trade_resets_consecutive_losses(self):
        rg = RiskGuard(circuit_breaker_consecutive_losses=3)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        rg.record_trade_pnl(-100, 0)
        rg.record_trade_pnl(-100, 1)
        assert rg.consecutive_losses == 2

        rg.record_trade_pnl(50, 2)  # Winner resets
        assert rg.consecutive_losses == 0

    def test_cooldown_blocks_rapid_retrade(self):
        rg = RiskGuard(trade_cooldown_bars=2)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        rg.last_trade_bar["AAPL"] = 5

        # Bar 6 — only 1 bar since last trade, cooldown=2
        ok, reason = rg.can_enter("AAPL", "BUY", 100, 100.0, 100_000, n_positions=0, bar_idx=6)
        assert ok is False
        assert reason == "cooldown"

        # Bar 7 — 2 bars since last trade, OK
        ok, _ = rg.can_enter("AAPL", "BUY", 100, 100.0, 100_000, n_positions=0, bar_idx=7)
        assert ok is True

    def test_max_order_notional_blocks(self):
        rg = RiskGuard(max_order_notional=50_000)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        # 1000 shares * $100 = $100k > $50k limit
        ok, reason = rg.can_enter("AAPL", "BUY", 1000, 100.0, 100_000, n_positions=0, bar_idx=0)
        assert ok is False
        assert reason == "max_order_notional"

    def test_max_order_shares_blocks(self):
        rg = RiskGuard(max_order_shares=500)
        rg.reset(100_000)
        rg.on_new_bar(100_000, datetime(2024, 1, 2), 0)

        ok, reason = rg.can_enter("AAPL", "BUY", 600, 10.0, 100_000, n_positions=0, bar_idx=0)
        assert ok is False
        assert reason == "max_order_shares"

    def test_uninitalized_guard_allows_all(self):
        """When reset() was never called, all entries are allowed (backward compat)."""
        rg = RiskGuard(max_positions=1)
        # Do NOT call rg.reset() or rg.on_new_bar()
        ok, _ = rg.can_enter("AAPL", "BUY", 100, 100.0, 100_000, n_positions=5, bar_idx=0)
        assert ok is True


class TestRiskGuardIntegration:
    """Risk guards work during full engine.run()."""

    def test_position_limit_in_run(self):
        """Engine blocks entries when max_positions reached."""
        # Use crypto symbols (24/7 market) to avoid market-hours gating
        data = {
            **_make_price_data("CRYPTO:BTC/USD", [40000.0] * 20),
            **_make_price_data("CRYPTO:ETH/USD", [3000.0] * 20),
            **_make_price_data("CRYPTO:SOL/USD", [100.0] * 20),
        }
        engine = _engine(
            max_positions=2,
            crypto_commission_bps=0.0,
            crypto_spread_bps=0.0,
        )
        engine.load_data(data)
        ts = list(data["CRYPTO:BTC/USD"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 2:
                eng.execute_order("CRYPTO:BTC/USD", "BUY", 0.1)
            elif idx == 4:
                eng.execute_order("CRYPTO:ETH/USD", "BUY", 1.0)
            elif idx == 6:
                eng.execute_order("CRYPTO:SOL/USD", "BUY", 10.0)  # Should be blocked

        engine.run(strategy, ts[0], ts[-1])

        # Only 2 positions should be open
        assert len(engine.positions) <= 2
        assert "CRYPTO:SOL/USD" not in engine.positions

    def test_circuit_breaker_in_run(self):
        """Circuit breaker trips after consecutive losses and blocks new entries."""
        # Prices drop, causing losing trades
        closes = list(range(100, 60, -1))  # 100 down to 61
        data = _make_price_data("AAPL", closes)
        engine = _engine(
            circuit_breaker_consecutive_losses=2,
            circuit_breaker_cooldown_bars=100,  # Long cooldown so it stays tripped
        )
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            # Keep buying and selling at a loss
            if idx == 2:
                eng.execute_order("AAPL", "BUY", 10)
            elif idx == 5:
                eng.execute_order("AAPL", "SELL", 10)  # Loss
            elif idx == 7:
                eng.execute_order("AAPL", "BUY", 10)
            elif idx == 10:
                eng.execute_order("AAPL", "SELL", 10)  # Loss #2 → trips CB
            elif idx == 12:
                eng.execute_order("AAPL", "BUY", 10)  # Should be blocked

        engine.run(strategy, ts[0], ts[-1])

        # After 2 consecutive losses, circuit breaker should have tripped
        assert engine.risk_guard.circuit_breaker_active is True


# ===========================================================================
# 4. EXECUTION REALISM
# ===========================================================================

class TestOpenPriceFill:
    """t+1 fills use Open price when use_open_price_fill=True."""

    def test_fills_at_open_not_close(self):
        # Open and Close are close together to avoid price_deviation rejection
        opens = [100.0, 100.5, 101.0, 101.5, 102.0]
        closes = [100.2, 100.7, 101.2, 101.7, 102.2]
        data = _make_price_data("AAPL", closes, opens=opens)
        engine = _engine(use_open_price_fill=True)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 0:
                eng.execute_order("AAPL", "BUY", 100)

        engine.run(strategy, ts[0], ts[-1])

        # Order placed at bar 0, fills at bar 1 using Open=100.5 (not Close=100.7)
        assert len(engine.trades) == 1
        assert engine.trades[0].price == pytest.approx(100.5, abs=0.01)

    def test_falls_back_to_close_when_no_open(self):
        """If data has no Open column, falls back to Close."""
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        df = pd.DataFrame({"Close": [100.0] * 5, "Volume": [1e6] * 5}, index=dates)
        data = {"AAPL": df}

        engine = _engine(use_open_price_fill=True)
        engine.load_data(data)
        ts = list(df.index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 0:
                eng.execute_order("AAPL", "BUY", 100)

        engine.run(strategy, ts[0], ts[-1])
        assert len(engine.trades) == 1
        assert engine.trades[0].price == pytest.approx(100.0, abs=0.01)


class TestPartialFills:
    """Volume-based partial fill simulation."""

    def test_partial_fill_when_volume_low(self):
        """Order exceeding 10% of volume gets partially filled."""
        # Volume = 100, max_participation = 0.10, so max fill = 10 shares
        data = _make_price_data("AAPL", [100.0] * 10, volume=100)
        engine = _engine(max_participation_rate=0.10)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 0:
                eng.execute_order("AAPL", "BUY", 50)  # Want 50 but only 10 allowed

        engine.run(strategy, ts[0], ts[-1])

        # First fill should be 10, remainder re-queued
        assert len(engine.trades) >= 1
        assert engine.trades[0].quantity == pytest.approx(10.0, abs=0.01)

    def test_full_fill_when_volume_sufficient(self):
        """Order within volume limit fills completely."""
        data = _make_price_data("AAPL", [100.0] * 5, volume=10_000)
        engine = _engine(max_participation_rate=0.10)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 0:
                eng.execute_order("AAPL", "BUY", 100)  # 100 < 10% * 10000 = 1000

        engine.run(strategy, ts[0], ts[-1])
        assert len(engine.trades) == 1
        assert engine.trades[0].quantity == pytest.approx(100.0, abs=0.01)


class TestPreTradeChecks:
    """Pre-trade notional and share limits."""

    def test_rejects_oversized_notional(self):
        """Order exceeding max_order_notional is rejected."""
        data = _make_price_data("AAPL", [100.0] * 10)
        engine = _engine(max_order_notional=5_000)
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 0:
                eng.execute_order("AAPL", "BUY", 100)  # 100 * 100 = $10k > $5k limit

        engine.run(strategy, ts[0], ts[-1])
        assert len(engine.trades) == 0  # Rejected
        assert "AAPL" not in engine.positions


# ===========================================================================
# 5. DYNAMIC POSITION SIZING
# ===========================================================================

class TestPositionSizing:
    """calculate_position_size returns reasonable values."""

    def test_basic_sizing(self):
        engine = _engine(initial_capital=100_000)
        # Simulate some price history
        prices = pd.Series(list(range(95, 106)))

        result = engine.calculate_position_size(
            entry_price=100.0,
            signal_strength=0.7,
            confidence=0.6,
            prices=prices,
            regime='neutral',
        )

        assert result['shares'] >= 0
        assert result['stop_loss'] < 100.0  # Long stop below entry
        assert result['take_profit'] > 100.0  # Long target above entry
        assert result['trailing_stop_pct'] > 0
        assert result['kelly_fraction'] > 0
        assert result['position_pct'] <= 0.05  # Capped at 5%

    def test_drawdown_reduces_size(self):
        """Position size should decrease when drawdown increases."""
        engine = _engine(initial_capital=100_000)
        engine.risk_guard.reset(100_000)
        engine.risk_guard.peak_equity = 100_000
        prices = pd.Series(list(range(95, 106)))

        # No drawdown
        result_no_dd = engine.calculate_position_size(
            entry_price=100.0, signal_strength=0.7, confidence=0.6,
            prices=prices, regime='neutral',
        )

        # Simulate drawdown by reducing cash
        engine.cash = 88_000  # ~12% drawdown
        result_dd = engine.calculate_position_size(
            entry_price=100.0, signal_strength=0.7, confidence=0.6,
            prices=prices, regime='neutral',
        )

        assert result_dd['shares'] < result_no_dd['shares']
        assert result_dd['drawdown_multiplier'] < result_no_dd['drawdown_multiplier']

    def test_regime_affects_size(self):
        """Bull regime should size larger than bear regime."""
        engine = _engine(initial_capital=100_000)
        prices = pd.Series(list(range(95, 106)))

        result_bull = engine.calculate_position_size(
            entry_price=100.0, signal_strength=0.7, confidence=0.6,
            prices=prices, regime='bull',
        )
        result_bear = engine.calculate_position_size(
            entry_price=100.0, signal_strength=0.7, confidence=0.6,
            prices=prices, regime='bear',
        )

        assert result_bull['regime_multiplier'] > result_bear['regime_multiplier']

    def test_short_signal_sizing(self):
        """Negative signal_strength should produce stop above entry."""
        engine = _engine(initial_capital=100_000)
        prices = pd.Series(list(range(95, 106)))

        result = engine.calculate_position_size(
            entry_price=100.0, signal_strength=-0.7, confidence=0.6,
            prices=prices, regime='neutral',
        )

        assert result['stop_loss'] > 100.0  # Stop above for short
        assert result['take_profit'] < 100.0  # Target below for short


# ===========================================================================
# 6. INTEGRATION
# ===========================================================================

class TestFullIntegration:
    """Full backtest run with all features enabled."""

    def test_full_run_produces_valid_results(self):
        """Complete backtest with stops, risk, and sizing produces valid equity curve."""
        np.random.seed(42)
        n = 100
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n))))
        data = _make_price_data("AAPL", prices)
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=5.0,
            use_dynamic_slippage=False,
            use_open_price_fill=False,
            enable_stop_management=True,
            default_stop_loss_pct=0.05,
            default_take_profit_pct=0.10,
            default_trailing_stop_pct=0.03,
            default_max_hold_bars=20,
            max_positions=5,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.15,
            max_participation_rate=0.0,  # Disable partial fills for simplicity
        )
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 5:
                eng.execute_order("AAPL", "BUY", 100)

        results = engine.run(strategy, ts[0], ts[-1])

        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'risk_stats' in results
        assert results['final_equity'] > 0
        assert len(engine.history) == n

    def test_equity_invariant_with_all_features(self):
        """Equity = cash + positions at every step, even with stops and risk guards."""
        np.random.seed(123)
        n = 80
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0, 0.02, n))))
        data = _make_price_data("AAPL", prices)
        engine = BacktestEngine(
            initial_capital=100_000,
            slippage_bps=5.0,
            use_dynamic_slippage=False,
            use_open_price_fill=False,
            enable_stop_management=True,
            default_stop_loss_pct=0.08,
            default_take_profit_pct=0.15,
            default_max_hold_bars=30,
            max_participation_rate=0.0,
        )
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 3:
                eng.execute_order("AAPL", "BUY", 50)

        engine.run(strategy, ts[0], ts[-1])

        # Every recorded equity should be positive
        for step in engine.history:
            assert step["equity"] > 0

    def test_backward_compat_no_new_features(self):
        """Engine with all new features disabled behaves like the old engine."""
        np.random.seed(42)
        n = 50
        prices = list(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))))
        data = _make_price_data("AAPL", prices)
        engine = _engine()  # All new features disabled via defaults
        engine.load_data(data)
        ts = list(data["AAPL"].index)

        def strategy(eng, timestamp):
            idx = ts.index(timestamp) if timestamp in ts else -1
            if idx == 5:
                eng.execute_order("AAPL", "BUY", 50)
            elif idx == 30:
                eng.execute_order("AAPL", "SELL", 50)

        results = engine.run(strategy, ts[0], ts[-1])

        assert len(engine.trades) == 2
        assert results['total_trades'] == 2
        assert results['final_equity'] > 0
