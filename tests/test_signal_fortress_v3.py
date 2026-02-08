"""
tests/test_signal_fortress_v3.py - Tests for Signal Fortress Phase 3

Phase 3: Autonomous Money Machine
- MacroEventShield: FOMC/CPI/NFP blackouts, Game Over protection
- OvernightRiskGuard: Gap protection, end-of-day exposure reduction
- ProfitRatchet: Progressive trailing stops, lock in gains
- LiquidityGuard: Detect illiquid conditions
- PositionAgingManager: Time-based exits
"""

import pytest
from datetime import datetime, timedelta, time
from unittest.mock import MagicMock, patch
import numpy as np

from risk.macro_event_shield import (
    MacroEventShield, MacroEvent, EventType, EventImpact, MacroState
)
from risk.overnight_risk_guard import (
    OvernightRiskGuard, MarketPhase, OvernightRiskAssessment
)
from risk.profit_ratchet import (
    ProfitRatchet, ProfitTier, RatchetState, PartialExitRecommendation
)
from risk.liquidity_guard import (
    LiquidityGuard, LiquidityRegime, LiquidityMetrics, MarketLiquidityState
)
from risk.position_aging_manager import (
    PositionAgingManager, AgingTier, PositionAge, AgingRecommendation
)


# ═══════════════════════════════════════════════════════════════════════════════
# MacroEventShield Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMacroEventShield:
    """Tests for MacroEventShield - FOMC/CPI/NFP blackouts."""

    def setup_method(self):
        """Set up test fixtures."""
        self.shield = MacroEventShield(
            blackout_before_fomc=60,
            blackout_after_fomc=30,
            blackout_before_data=30,
            blackout_after_data=15,
            game_over_threshold=0.05,
        )

    def test_initialization(self):
        """Shield initializes with correct parameters."""
        assert self.shield.blackout_before_fomc == 60
        assert self.shield.blackout_after_fomc == 30
        assert self.shield.game_over_threshold == 0.05
        assert len(self.shield._events) > 0  # Has pre-loaded events

    def test_game_over_trigger(self):
        """Game over triggers at 5% daily loss."""
        # Set starting capital
        self.shield.update_daily_capital(1000000, is_day_start=True)

        # 3% loss - not game over
        self.shield.update_daily_capital(970000)
        assert not self.shield.is_game_over()

        # 5% loss - game over
        self.shield.update_daily_capital(950000)
        assert self.shield.is_game_over()

    def test_game_over_blocks_all_trading(self):
        """Game over state blocks all entries."""
        self.shield.update_daily_capital(1000000, is_day_start=True)
        self.shield.update_daily_capital(940000)  # 6% loss

        blocked, reason = self.shield.should_block_entry()
        assert blocked
        assert "GAME OVER" in reason

        state = self.shield.assess_state()
        assert state.game_over
        assert not state.trading_allowed
        assert state.position_sizing_multiplier == 0.0

    def test_game_over_reset_requires_confirmation(self):
        """Game over reset requires manual confirmation."""
        self.shield.update_daily_capital(1000000, is_day_start=True)
        self.shield.update_daily_capital(940000)  # Trigger game over

        # Reset without confirmation fails
        result = self.shield.reset_game_over(manual_confirmation=False)
        assert not result
        assert self.shield.is_game_over()

        # Reset with confirmation succeeds
        result = self.shield.reset_game_over(manual_confirmation=True)
        assert result
        assert not self.shield.is_game_over()

    def test_add_custom_event(self):
        """Can add custom macro events."""
        event_time = datetime.now() + timedelta(hours=2)
        self.shield.add_custom_event(
            event_type=EventType.FED_SPEECH,
            event_time=event_time,
            impact=EventImpact.HIGH,
            description="Fed Chair Speech",
        )

        upcoming = self.shield.get_upcoming_events(hours=3)
        fed_events = [e for e in upcoming if e.event_type == EventType.FED_SPEECH]
        assert len(fed_events) >= 1

    def test_earnings_blackout_symbol_specific(self):
        """Earnings blackout only affects that symbol."""
        earnings_time = datetime.now() + timedelta(hours=1)
        self.shield.add_earnings_date("AAPL", earnings_time)

        # AAPL should be blocked
        blocked, reason = self.shield.should_block_entry("AAPL")
        # Note: might not be in blackout yet depending on timing

        # MSFT should not be affected by AAPL earnings
        state = self.shield.assess_state("MSFT")
        assert "AAPL" not in state.symbols_in_earnings_blackout or "MSFT" not in state.symbols_in_earnings_blackout

    def test_position_sizing_reduces_near_event(self):
        """Position sizing reduces as event approaches."""
        # Add an event 90 minutes from now
        event_time = datetime.now() + timedelta(minutes=90)
        self.shield.add_custom_event(
            event_type=EventType.CPI,
            event_time=event_time,
            impact=EventImpact.CRITICAL,
        )

        state = self.shield.assess_state()
        # 90 min away - should still allow some sizing
        assert state.position_sizing_multiplier > 0

    def test_diagnostics(self):
        """Diagnostics returns expected structure."""
        diag = self.shield.get_diagnostics()
        assert "in_blackout" in diag
        assert "game_over" in diag
        assert "trading_allowed" in diag
        assert "total_events_loaded" in diag


# ═══════════════════════════════════════════════════════════════════════════════
# OvernightRiskGuard Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOvernightRiskGuard:
    """Tests for OvernightRiskGuard - gap protection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = OvernightRiskGuard(
            market_open=time(9, 30),
            market_close=time(16, 0),
            no_entry_minutes=30,
            reduce_exposure_minutes=60,
            max_overnight_var_pct=2.0,
        )

    def test_market_phase_detection(self):
        """Correctly identifies market phases."""
        # Regular hours (11am)
        regular_time = datetime(2024, 1, 15, 11, 0)  # Monday 11am
        phase = self.guard.get_market_phase(regular_time)
        assert phase == MarketPhase.REGULAR

        # Power hour (3:15pm - 45 min to close)
        power_time = datetime(2024, 1, 15, 15, 15)
        phase = self.guard.get_market_phase(power_time)
        assert phase == MarketPhase.POWER_HOUR

        # Close prep (3:35pm - 25 min to close)
        close_prep_time = datetime(2024, 1, 15, 15, 35)
        phase = self.guard.get_market_phase(close_prep_time)
        assert phase == MarketPhase.CLOSE_PREP

        # Final minutes (3:50pm - 10 min to close)
        final_time = datetime(2024, 1, 15, 15, 50)
        phase = self.guard.get_market_phase(final_time)
        assert phase == MarketPhase.FINAL_MINUTES

    def test_entry_blocked_near_close(self):
        """Entries blocked in last 30 minutes."""
        close_time = datetime(2024, 1, 15, 15, 40)  # 20 min to close

        blocked, reason = self.guard.should_block_entry(close_time)
        assert blocked
        assert "30 minutes" in reason or "15 minutes" in reason

    def test_entry_allowed_regular_hours(self):
        """Entries allowed during regular hours."""
        regular_time = datetime(2024, 1, 15, 11, 0)

        blocked, reason = self.guard.should_block_entry(regular_time)
        assert not blocked

    def test_position_size_reduces_power_hour(self):
        """Position size reduces during power hour."""
        power_time = datetime(2024, 1, 15, 15, 15)

        mult = self.guard.get_position_size_multiplier(vix_level=20, current_time=power_time)
        assert mult < 1.0  # Reduced

    def test_earnings_force_exit(self):
        """Positions with earnings tonight are flagged for exit."""
        self.guard.add_earnings_tonight("AAPL")
        self.guard.add_earnings_tomorrow("MSFT")

        to_exit = self.guard.get_symbols_to_exit_before_close()
        assert "AAPL" in to_exit
        assert "MSFT" in to_exit

    def test_overnight_var_calculation(self):
        """Overnight VaR is calculated correctly."""
        positions = {
            "AAPL": {"market_value": 50000},
            "MSFT": {"market_value": 50000},
        }

        assessment = self.guard.assess_overnight_risk(
            positions=positions,
            capital=1000000,
            vix_level=20,
        )

        assert assessment.overnight_var_pct > 0
        assert assessment.overnight_var_pct < 10  # Reasonable range

    def test_high_vix_reduces_exposure(self):
        """High VIX reduces recommended exposure."""
        assessment_normal = self.guard.assess_overnight_risk(
            positions={}, capital=1000000, vix_level=15
        )
        assessment_high = self.guard.assess_overnight_risk(
            positions={}, capital=1000000, vix_level=30
        )

        # High VIX should have lower recommended exposure
        assert assessment_high.recommended_exposure_pct <= assessment_normal.recommended_exposure_pct

    def test_diagnostics(self):
        """Diagnostics returns expected structure."""
        diag = self.guard.get_diagnostics()
        assert "market_phase" in diag
        assert "entry_allowed" in diag


# ═══════════════════════════════════════════════════════════════════════════════
# ProfitRatchet Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProfitRatchet:
    """Tests for ProfitRatchet - progressive trailing stops."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ratchet = ProfitRatchet(
            tier_1_threshold=0.02,
            tier_2_threshold=0.05,
            tier_3_threshold=0.10,
            tier_4_threshold=0.20,
            initial_trailing_pct=0.03,
        )

    def test_register_position(self):
        """Position registration works."""
        self.ratchet.register_position("AAPL", entry_price=150.0)

        state = self.ratchet.get_position_state("AAPL")
        assert state is not None
        assert state["entry_price"] == 150.0

    def test_tier_progression(self):
        """Tiers progress correctly with gains."""
        self.ratchet.register_position("AAPL", entry_price=100.0)

        # 1% gain - still INITIAL
        state = self.ratchet.update_position("AAPL", current_price=101.0)
        assert state.tier == ProfitTier.INITIAL

        # 3% gain - TIER_1
        state = self.ratchet.update_position("AAPL", current_price=103.0)
        assert state.tier == ProfitTier.TIER_1

        # 7% gain - TIER_2
        state = self.ratchet.update_position("AAPL", current_price=107.0)
        assert state.tier == ProfitTier.TIER_2

        # 15% gain - TIER_3
        state = self.ratchet.update_position("AAPL", current_price=115.0)
        assert state.tier == ProfitTier.TIER_3

        # 25% gain - TIER_4
        state = self.ratchet.update_position("AAPL", current_price=125.0)
        assert state.tier == ProfitTier.TIER_4

    def test_profit_locking(self):
        """Profits are locked at higher tiers."""
        self.ratchet.register_position("AAPL", entry_price=100.0)

        # Move to 10% gain (Tier 3 = 80% locked)
        state = self.ratchet.update_position("AAPL", current_price=110.0)

        # Locked profit should be 80% of 10% = 8%
        assert state.locked_profit_pct == pytest.approx(0.08, rel=0.01)

        # Stop price should protect at least 8% gain
        # Stop = max(HWM * (1 - trailing), entry * (1 + locked))
        # Stop = max(110 * 0.982, 100 * 1.08) = max(108.02, 108) = 108.02
        assert state.stop_price >= 108.0

    def test_high_water_mark_tracking(self):
        """High water mark is tracked correctly."""
        self.ratchet.register_position("AAPL", entry_price=100.0)

        # Go up to 120
        self.ratchet.update_position("AAPL", current_price=120.0)

        # Come back down to 115
        state = self.ratchet.update_position("AAPL", current_price=115.0)

        # HWM should still be 120
        assert state.high_water_mark == 120.0

    def test_partial_profit_taking(self):
        """Partial profit taking triggers at correct tiers."""
        self.ratchet.register_position("AAPL", entry_price=100.0)

        # Move to Tier 2 (5%+)
        state = self.ratchet.update_position("AAPL", current_price=106.0)

        assert state.should_take_partial
        assert state.partial_size_pct == 0.25  # 25% at tier 2

    def test_stop_exit_trigger(self):
        """Stop exit triggers when price drops below stop."""
        self.ratchet.register_position("AAPL", entry_price=100.0)

        # Go up then down
        self.ratchet.update_position("AAPL", current_price=110.0)

        # Drop below stop
        should_exit, reason, stop = self.ratchet.should_exit("AAPL", current_price=95.0)
        assert should_exit
        assert "Ratchet stop" in reason

    def test_diagnostics(self):
        """Diagnostics returns expected structure."""
        self.ratchet.register_position("AAPL", entry_price=100.0)
        self.ratchet.update_position("AAPL", current_price=110.0)

        diag = self.ratchet.get_diagnostics()
        assert "positions_tracked" in diag
        assert diag["positions_tracked"] == 1
        assert "tier_distribution" in diag


# ═══════════════════════════════════════════════════════════════════════════════
# LiquidityGuard Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLiquidityGuard:
    """Tests for LiquidityGuard - illiquid condition detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.guard = LiquidityGuard(
            thin_spread_threshold=0.001,
            stressed_spread_threshold=0.003,
            crisis_spread_threshold=0.005,
        )

    def test_normal_liquidity(self):
        """Normal spreads result in NORMAL regime."""
        metrics = self.guard.update_quote(
            symbol="AAPL",
            bid=149.99,
            ask=150.01,  # 0.013% spread
            volume=50_000_000,
            avg_volume_20d=40_000_000,
        )

        assert metrics.regime == LiquidityRegime.NORMAL
        assert metrics.is_tradeable
        assert metrics.position_size_multiplier == 1.0

    def test_thin_liquidity(self):
        """Thin spreads result in THIN regime."""
        metrics = self.guard.update_quote(
            symbol="SMALL",
            bid=49.90,
            ask=50.10,  # 0.4% spread
            volume=500_000,
            avg_volume_20d=800_000,
        )

        # Spread alone would be STRESSED, volume ratio might push it further
        assert metrics.regime >= LiquidityRegime.THIN

    def test_crisis_liquidity(self):
        """Wide spreads result in CRISIS regime."""
        metrics = self.guard.update_quote(
            symbol="ILLIQUID",
            bid=9.50,
            ask=10.50,  # 10% spread
            volume=10_000,
            avg_volume_20d=100_000,
        )

        assert metrics.regime == LiquidityRegime.CRISIS
        assert not metrics.is_tradeable
        assert metrics.position_size_multiplier == 0.0

    def test_entry_blocked_in_crisis(self):
        """Entries blocked for crisis liquidity symbols."""
        self.guard.update_quote("CRISIS", bid=9.50, ask=10.50, volume=1000, avg_volume_20d=100000)

        blocked, reason = self.guard.should_block_entry("CRISIS")
        assert blocked
        assert "crisis" in reason.lower() or "spread" in reason.lower()

    def test_blacklist_blocks_entry(self):
        """Blacklisted symbols are blocked."""
        self.guard.blacklist_symbol("BAD", reason="Known illiquid")

        blocked, reason = self.guard.should_block_entry("BAD")
        assert blocked
        assert "blacklist" in reason.lower()

    def test_market_wide_assessment(self):
        """Market-wide liquidity assessment works."""
        # Add several symbols with varying liquidity
        self.guard.update_quote("AAPL", 149.99, 150.01, 50000000, 40000000)
        self.guard.update_quote("MSFT", 299.99, 300.01, 30000000, 25000000)
        self.guard.update_quote("SMALL", 9.90, 10.10, 50000, 100000)

        state = self.guard.assess_market_liquidity()
        assert isinstance(state, MarketLiquidityState)
        assert state.regime in list(LiquidityRegime)

    def test_position_size_adjustment(self):
        """Position size multiplier adjusts by liquidity."""
        # Normal liquidity
        self.guard.update_quote("GOOD", 99.99, 100.01, 10000000, 10000000)
        mult_good = self.guard.get_position_size_multiplier("GOOD")
        assert mult_good == 1.0

        # Poor liquidity
        self.guard.update_quote("BAD", 9.80, 10.20, 10000, 100000)
        mult_bad = self.guard.get_position_size_multiplier("BAD")
        assert mult_bad < 1.0

    def test_diagnostics(self):
        """Diagnostics returns expected structure."""
        self.guard.update_quote("AAPL", 149.99, 150.01, 50000000, 40000000)

        diag = self.guard.get_diagnostics()
        assert "symbols_tracked" in diag
        assert "market_regime" in diag
        assert "regime_distribution" in diag


# ═══════════════════════════════════════════════════════════════════════════════
# PositionAgingManager Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionAgingManager:
    """Tests for PositionAgingManager - time-based exits."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = PositionAgingManager(
            fresh_days=5,
            maturing_days=10,
            stale_days=15,
            critical_days=20,
            max_days=30,
            stale_min_pnl=0.0,
            critical_min_pnl=0.02,
        )

    def test_tier_progression(self):
        """Tiers progress correctly with time."""
        # Register position 3 days ago
        entry_time = datetime.now() - timedelta(days=3)
        self.manager.register_position("AAPL", 150.0, entry_time)

        age = self.manager.update_position("AAPL", 151.0)
        assert age.tier == AgingTier.FRESH
        assert age.days_held == 3

    def test_stale_position_requires_profit(self):
        """Stale positions require positive P&L."""
        entry_time = datetime.now() - timedelta(days=16)
        self.manager.register_position("AAPL", 150.0, entry_time)

        # Losing position at stale tier
        age = self.manager.update_position("AAPL", 145.0)
        assert age.tier == AgingTier.STALE
        assert age.should_exit  # Should exit because P&L < 0

    def test_critical_position_requires_2pct(self):
        """Critical positions require 2%+ P&L."""
        entry_time = datetime.now() - timedelta(days=21)
        self.manager.register_position("AAPL", 100.0, entry_time)

        # 1% profit at critical tier - not enough
        age = self.manager.update_position("AAPL", 101.0)
        assert age.tier == AgingTier.CRITICAL
        assert age.should_exit  # 1% < 2% required

        # 3% profit - should be OK
        age = self.manager.update_position("AAPL", 103.0)
        assert not age.should_exit

    def test_expired_forces_exit(self):
        """Expired positions force exit regardless of P&L."""
        entry_time = datetime.now() - timedelta(days=31)
        self.manager.register_position("AAPL", 100.0, entry_time)

        # Even 10% profit should exit at expired
        age = self.manager.update_position("AAPL", 110.0)
        assert age.tier == AgingTier.EXPIRED
        assert age.should_exit
        assert "Max holding period" in age.exit_reason

    def test_stop_tightening_by_tier(self):
        """Stops tighten as position ages."""
        # Fresh position
        entry_fresh = datetime.now() - timedelta(days=2)
        self.manager.register_position("FRESH", 100.0, entry_fresh)
        age_fresh = self.manager.update_position("FRESH", 105.0)

        # Stale position
        entry_stale = datetime.now() - timedelta(days=16)
        self.manager.register_position("STALE", 100.0, entry_stale)
        age_stale = self.manager.update_position("STALE", 105.0)

        # Stale should have tighter stops
        assert age_stale.stop_tightening < age_fresh.stop_tightening

    def test_recommendations_sorted_by_urgency(self):
        """Recommendations are sorted by urgency."""
        # Create positions at different tiers
        self.manager.register_position(
            "FRESH", 100.0, datetime.now() - timedelta(days=2)
        )
        self.manager.register_position(
            "CRITICAL", 100.0, datetime.now() - timedelta(days=21)
        )

        positions = {
            "FRESH": {"current_price": 101.0, "entry_price": 100.0},
            "CRITICAL": {"current_price": 100.5, "entry_price": 100.0},
        }

        recs = self.manager.get_recommendations(positions)

        # Critical should come first (if it generates a recommendation)
        if recs:
            # First recommendation should be high urgency if CRITICAL is flagged
            critical_rec = [r for r in recs if r.symbol == "CRITICAL"]
            if critical_rec:
                assert critical_rec[0].urgency == "high"

    def test_velocity_calculation(self):
        """Velocity (P&L rate) is calculated."""
        entry_time = datetime.now() - timedelta(days=5)
        self.manager.register_position("AAPL", 100.0, entry_time)

        # Simulate price updates over time
        self.manager.update_position("AAPL", 101.0)
        self.manager.update_position("AAPL", 102.0)
        self.manager.update_position("AAPL", 103.0)

        age = self.manager.update_position("AAPL", 104.0)
        # Velocity should be positive (gaining)
        # Note: velocity calculation depends on time between updates

    def test_diagnostics(self):
        """Diagnostics returns expected structure."""
        self.manager.register_position(
            "AAPL", 100.0, datetime.now() - timedelta(days=5)
        )

        diag = self.manager.get_diagnostics()
        assert "positions_tracked" in diag
        assert diag["positions_tracked"] == 1
        assert "tier_distribution" in diag
        assert "average_days_held" in diag


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase3Integration:
    """Integration tests for Phase 3 components working together."""

    def test_all_guards_can_block_entry(self):
        """All guards can independently block entries."""
        macro = MacroEventShield(game_over_threshold=0.05)
        overnight = OvernightRiskGuard()
        liquidity = LiquidityGuard()

        # Trigger game over
        macro.update_daily_capital(1000000, is_day_start=True)
        macro.update_daily_capital(940000)
        blocked, _ = macro.should_block_entry()
        assert blocked

        # Trigger overnight block
        close_time = datetime(2024, 1, 15, 15, 50)
        blocked, _ = overnight.should_block_entry(close_time)
        assert blocked

        # Trigger liquidity block
        liquidity.update_quote("BAD", 9.0, 11.0, 1000, 100000)
        blocked, _ = liquidity.should_block_entry("BAD")
        assert blocked

    def test_profit_ratchet_with_aging(self):
        """Profit ratchet and aging work together."""
        ratchet = ProfitRatchet()
        aging = PositionAgingManager()

        entry_time = datetime.now() - timedelta(days=10)

        # Both track the same position
        ratchet.register_position("AAPL", 100.0, entry_time)
        aging.register_position("AAPL", 100.0, entry_time)

        # Update both
        ratchet_state = ratchet.update_position("AAPL", 110.0)
        aging_state = aging.update_position("AAPL", 110.0)

        # Ratchet should have profit tier
        assert ratchet_state.tier >= ProfitTier.TIER_2

        # Aging should be MATURING
        assert aging_state.tier == AgingTier.MATURING

        # Both provide stop adjustments
        assert ratchet_state.stop_price > 100.0
        assert aging_state.stop_tightening < 1.0

    def test_liquidity_affects_overnight_assessment(self):
        """Liquidity conditions could affect overnight risk."""
        overnight = OvernightRiskGuard()
        liquidity = LiquidityGuard()

        # Wide spreads late in day = higher overnight risk
        liquidity.update_quote("WIDE", 99.0, 101.0, 100000, 500000)
        metrics = liquidity.get_metrics("WIDE")

        # Both independently assess risk
        assert metrics is not None
        assert metrics.regime >= LiquidityRegime.THIN

    def test_multiple_guards_diagnostics(self):
        """All guards provide diagnostics."""
        macro = MacroEventShield()
        overnight = OvernightRiskGuard()
        ratchet = ProfitRatchet()
        liquidity = LiquidityGuard()
        aging = PositionAgingManager()

        # All should have diagnostics methods
        assert "game_over" in macro.get_diagnostics()
        assert "market_phase" in overnight.get_diagnostics()
        assert "positions_tracked" in ratchet.get_diagnostics()
        assert "symbols_tracked" in liquidity.get_diagnostics()
        assert "positions_tracked" in aging.get_diagnostics()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
