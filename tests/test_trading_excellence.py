"""
tests/test_trading_excellence.py - Tests for Trading Excellence Module
"""

import pytest
from risk.trading_excellence import (
    TradingExcellenceManager,
    MismatchSeverity,
    ProfitAction,
    quick_mismatch_check
)


class TestSignalMismatchDetection:
    """Test signal-position mismatch detection."""

    def setup_method(self):
        self.manager = TradingExcellenceManager()

    def test_no_mismatch_long_bullish(self):
        """LONG position with bullish signal = no mismatch."""
        alert = self.manager.check_signal_mismatch(
            symbol="AAPL",
            position_side="LONG",
            position_qty=100,
            signal=0.5,  # Bullish
            confidence=0.8,
            entry_price=150.0,
            current_price=155.0  # +3.3%
        )
        assert alert is None

    def test_critical_mismatch_long_strong_bearish(self):
        """LONG position with strong bearish signal = CRITICAL."""
        alert = self.manager.check_signal_mismatch(
            symbol="AAPL",
            position_side="LONG",
            position_qty=100,
            signal=-0.5,  # Strong bearish
            confidence=0.85,  # High confidence
            entry_price=150.0,
            current_price=145.0  # -3.3%
        )
        assert alert is not None
        assert alert.severity == MismatchSeverity.CRITICAL
        assert "IMMEDIATELY" in alert.recommendation

    def test_strong_mismatch_long_bearish(self):
        """LONG position with bearish signal = STRONG mismatch."""
        alert = self.manager.check_signal_mismatch(
            symbol="AAPL",
            position_side="LONG",
            position_qty=100,
            signal=-0.20,  # Bearish
            confidence=0.6,
            entry_price=150.0,
            current_price=148.0
        )
        assert alert is not None
        assert alert.severity == MismatchSeverity.STRONG

    def test_moderate_mismatch_neutral_signal(self):
        """Position with neutral signal = MODERATE mismatch."""
        alert = self.manager.check_signal_mismatch(
            symbol="AAPL",
            position_side="LONG",
            position_qty=100,
            signal=0.02,  # Nearly neutral
            confidence=0.7,
            entry_price=150.0,
            current_price=155.0  # +3.3% profit
        )
        assert alert is not None
        assert alert.severity == MismatchSeverity.MODERATE
        assert "PROFITS" in alert.recommendation

    def test_short_mismatch_bullish_signal(self):
        """SHORT position with bullish signal = STRONG mismatch."""
        alert = self.manager.check_signal_mismatch(
            symbol="NFLX",
            position_side="SHORT",
            position_qty=-100,
            signal=0.25,  # Bullish
            confidence=0.75,
            entry_price=80.0,
            current_price=82.0  # -2.5% (losing)
        )
        assert alert is not None
        assert alert.severity == MismatchSeverity.STRONG


class TestProfitDecision:
    """Test profit-taking decisions."""

    def setup_method(self):
        self.manager = TradingExcellenceManager()

    def test_hold_small_profit(self):
        """Small profit (< 2%) = HOLD."""
        decision = self.manager.get_profit_decision(
            symbol="AAPL",
            position_side="LONG",
            entry_price=150.0,
            current_price=151.5,  # +1%
            peak_price=151.5,
            signal=0.5,
            confidence=0.8
        )
        assert decision.action == ProfitAction.HOLD

    def test_trail_at_tier_1(self):
        """2%+ profit = TRAIL."""
        decision = self.manager.get_profit_decision(
            symbol="AAPL",
            position_side="LONG",
            entry_price=150.0,
            current_price=153.5,  # +2.3%
            peak_price=153.5,
            signal=0.5,
            confidence=0.8
        )
        assert decision.action == ProfitAction.TRAIL
        assert decision.profit_tier == 1

    def test_partial_at_5pct_weak_signal(self):
        """5%+ profit with weak signal = PARTIAL."""
        decision = self.manager.get_profit_decision(
            symbol="AAPL",
            position_side="LONG",
            entry_price=150.0,
            current_price=158.0,  # +5.3%
            peak_price=158.0,
            signal=0.10,  # Weak
            confidence=0.6
        )
        assert decision.action == ProfitAction.PARTIAL

    def test_full_at_10pct_opposite_signal(self):
        """10%+ profit with opposite signal = FULL exit."""
        decision = self.manager.get_profit_decision(
            symbol="AAPL",
            position_side="LONG",
            entry_price=150.0,
            current_price=165.5,  # +10.3%
            peak_price=165.5,
            signal=-0.15,  # Bearish
            confidence=0.7
        )
        assert decision.action == ProfitAction.FULL

    def test_trailing_stop_triggered(self):
        """Drawdown from peak exceeds trailing = FULL exit."""
        decision = self.manager.get_profit_decision(
            symbol="AAPL",
            position_side="LONG",
            entry_price=150.0,
            current_price=157.0,  # +4.7% from entry
            peak_price=165.0,     # +10% was peak, now -4.8% drawdown
            signal=0.3,
            confidence=0.7
        )
        # Should trigger trailing stop (locked 80% at tier 3, trail at 2%)
        assert decision.action == ProfitAction.FULL
        assert "Trailing stop triggered" in decision.reason


class TestSizeScaling:
    """Test position size scaling."""

    def setup_method(self):
        self.manager = TradingExcellenceManager()

    def test_weak_signal_reduces_size(self):
        """Weak signal = reduced size."""
        rec = self.manager.calculate_size_scaling(
            symbol="AAPL",
            signal=0.15,  # Weak
            confidence=0.8,
            regime="neutral",
            base_shares=100
        )
        assert rec.scaling_factor < 1.0
        assert rec.recommended_shares < 100

    def test_strong_signal_increases_size(self):
        """Strong signal = increased size."""
        rec = self.manager.calculate_size_scaling(
            symbol="AAPL",
            signal=0.75,  # Strong
            confidence=0.85,
            regime="neutral",
            base_shares=100
        )
        assert rec.scaling_factor > 1.0
        assert rec.recommended_shares > 100

    def test_low_confidence_reduces_size(self):
        """Low confidence = reduced size."""
        rec = self.manager.calculate_size_scaling(
            symbol="AAPL",
            signal=0.5,  # Good signal
            confidence=0.4,  # Low confidence
            regime="neutral",
            base_shares=100
        )
        assert rec.scaling_factor < 1.0

    def test_volatile_regime_reduces_size(self):
        """Volatile regime = reduced size."""
        rec = self.manager.calculate_size_scaling(
            symbol="AAPL",
            signal=0.5,
            confidence=0.8,
            regime="high_volatility",
            base_shares=100
        )
        assert rec.scaling_factor < 1.0
        assert "Volatile regime" in " ".join(rec.reasons)


class TestQuickMismatchCheck:
    """Test the quick mismatch check function."""

    def test_long_strong_bearish_exits(self):
        """LONG + strong bearish = exit."""
        should_exit, reason = quick_mismatch_check(
            position_side="LONG",
            signal=-0.35,
            confidence=0.8,
            pnl_pct=2.0
        )
        assert should_exit
        assert "Strong bearish" in reason

    def test_long_bearish_exits(self):
        """LONG + bearish = exit."""
        should_exit, reason = quick_mismatch_check(
            position_side="LONG",
            signal=-0.20,
            confidence=0.7,
            pnl_pct=0.0
        )
        assert should_exit
        assert "bearish" in reason.lower()

    def test_long_weak_with_profit_exits(self):
        """LONG + weak signal + profit = exit (take profits)."""
        should_exit, reason = quick_mismatch_check(
            position_side="LONG",
            signal=0.03,
            confidence=0.7,
            pnl_pct=6.0
        )
        assert should_exit
        assert "profit" in reason.lower()

    def test_long_weak_with_loss_exits(self):
        """LONG + weak signal + loss = exit."""
        should_exit, reason = quick_mismatch_check(
            position_side="LONG",
            signal=0.03,
            confidence=0.7,
            pnl_pct=-2.0
        )
        assert should_exit
        assert "loss" in reason.lower()

    def test_long_aligned_holds(self):
        """LONG + bullish signal = hold."""
        should_exit, reason = quick_mismatch_check(
            position_side="LONG",
            signal=0.40,
            confidence=0.8,
            pnl_pct=3.0
        )
        assert not should_exit

    def test_short_bullish_exits(self):
        """SHORT + bullish signal = exit."""
        should_exit, reason = quick_mismatch_check(
            position_side="SHORT",
            signal=0.25,
            confidence=0.75,
            pnl_pct=-1.0
        )
        assert should_exit
        assert "bullish" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
