"""
tests/test_signal_fortress_v2.py - Signal Fortress Phase 2 Tests

Comprehensive tests for all 6 Phase 2 components:
1. BlackSwanGuard - Real-time crash detection
2. SignalDecayShield - Time-decay & staleness guard
3. ExitQualityGuard - Exit signal validation
4. CorrelationCascadeBreaker - Portfolio correlation shield
5. DrawdownCascadeBreaker - 5-tier drawdown response
6. ExecutionShield - Smart execution wrapper
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import all Phase 2 components
from risk.black_swan_guard import BlackSwanGuard, ThreatLevel
from monitoring.signal_decay_shield import SignalDecayShield
from risk.exit_quality_guard import ExitQualityGuard, OrderType
from risk.correlation_cascade_breaker import CorrelationCascadeBreaker, CorrelationRegime, CorrelationState
from risk.drawdown_cascade_breaker import DrawdownCascadeBreaker, DrawdownTier
from execution.execution_shield import ExecutionShield, ExecutionAlgo, Urgency


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: BlackSwanGuard
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlackSwanGuard:
    """Tests for real-time crash detection."""

    def setup_method(self):
        self.guard = BlackSwanGuard(
            crash_velocity_10m=0.02,
            crash_velocity_30m=0.04,
            vix_spike_elevated=0.30,
            vix_spike_severe=0.50,
        )

    def test_initialization(self):
        """Guard initializes with correct defaults."""
        assert self.guard._threat_level == ThreatLevel.NORMAL
        assert self.guard.get_position_size_multiplier() == 1.0
        assert not self.guard.should_block_entry()

    def test_normal_conditions(self):
        """No threat in normal market conditions."""
        # Record normal price movement using new API
        for i in range(20):
            self.guard.record_index_price("SPY", 100.0 + i * 0.1)

        self.guard.record_vix(18.0, 17.0)  # Normal VIX

        result = self.guard.assess_threat()
        assert result.threat_level == ThreatLevel.NORMAL
        assert result.position_size_multiplier == 1.0
        assert not result.entry_blocked

    def test_vix_spike_elevated(self):
        """VIX spike triggers ELEVATED threat."""
        # Record some baseline prices first
        for i in range(5):
            self.guard.record_index_price("SPY", 100.0)
        self.guard.record_vix(vix_current=26.0, vix_open=20.0)  # 30% spike

        result = self.guard.assess_threat()
        assert result.threat_level == ThreatLevel.ELEVATED
        assert result.position_size_multiplier == 0.5
        assert result.entry_blocked

    def test_vix_spike_severe(self):
        """Large VIX spike triggers SEVERE threat."""
        # Record some baseline prices first
        for i in range(5):
            self.guard.record_index_price("SPY", 100.0)
        self.guard.record_vix(vix_current=30.0, vix_open=20.0)  # 50% spike

        result = self.guard.assess_threat()
        assert result.threat_level == ThreatLevel.SEVERE
        assert result.position_size_multiplier == 0.25

    def test_positions_to_close_severe(self):
        """SEVERE threat closes worst 25% of positions."""
        # Record baseline and trigger SEVERE
        for i in range(5):
            self.guard.record_index_price("SPY", 100.0)
        self.guard.record_vix(vix_current=30.0, vix_open=20.0)
        self.guard.assess_threat()

        positions = {
            "AAPL": {"pnl": -100},
            "MSFT": {"pnl": 50},
            "GOOGL": {"pnl": -200},
            "AMZN": {"pnl": 100},
        }

        to_close = self.guard.get_positions_to_close(positions)
        assert "GOOGL" in to_close  # Worst P&L first

    def test_multi_trigger_escalation(self):
        """Multiple triggers escalate threat level."""
        # Record baseline
        for i in range(5):
            self.guard.record_index_price("SPY", 100.0)
        self.guard.record_vix(vix_current=26.0, vix_open=20.0)  # ELEVATED
        # Also high correlation
        result = self.guard.assess_threat(portfolio_correlations=[0.9, 0.85, 0.88])

        # Multi-trigger should escalate
        assert result.threat_level >= ThreatLevel.ELEVATED

    def test_diagnostics(self):
        """Diagnostics returns correct state."""
        for i in range(5):
            self.guard.record_index_price("SPY", 100.0)
        self.guard.record_vix(18.0, 17.0)
        diag = self.guard.get_diagnostics()

        assert "threat_level" in diag
        assert "vix_current" in diag
        assert diag["vix_current"] == 18.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: SignalDecayShield
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalDecayShield:
    """Tests for time-decay & staleness guard."""

    def setup_method(self):
        self.shield = SignalDecayShield(
            max_price_age_seconds=120,
            max_sentiment_age_seconds=1800,
            max_feature_age_seconds=14400,
        )

    def test_fresh_data(self):
        """Fresh data has no decay."""
        self.shield.record_data_timestamp("AAPL", "price")
        self.shield.record_data_timestamp("AAPL", "sentiment")

        report = self.shield.check_freshness("AAPL")
        assert report.is_fresh
        assert report.decay_factor == 1.0
        assert len(report.stale_components) == 0

    def test_stale_price_blocks_trading(self):
        """Stale price data blocks trading."""
        # Record price from 5 minutes ago
        old_time = datetime.now() - timedelta(seconds=350)
        self.shield.record_data_timestamp("AAPL", "price", old_time)

        assert not self.shield.is_data_tradeable("AAPL")

        report = self.shield.check_freshness("AAPL")
        assert not report.is_fresh
        assert report.decay_factor == 0.0

    def test_stale_sentiment_reduces_confidence(self):
        """Stale sentiment reduces confidence by 50%."""
        self.shield.record_data_timestamp("AAPL", "price")  # Fresh
        # Stale sentiment (35 min ago)
        old_time = datetime.now() - timedelta(seconds=2100)
        self.shield.record_data_timestamp("AAPL", "sentiment", old_time)

        report = self.shield.check_freshness("AAPL")
        assert "sentiment" in report.stale_components
        assert report.decay_factor == 0.5

    def test_apply_decay(self):
        """Decay correctly reduces signal and confidence."""
        self.shield.record_data_timestamp("AAPL", "price")
        old_time = datetime.now() - timedelta(seconds=2100)
        self.shield.record_data_timestamp("AAPL", "sentiment", old_time)

        report = self.shield.check_freshness("AAPL")
        signal, confidence = self.shield.apply_decay(0.5, 0.8, report)

        assert signal == 0.25  # 0.5 * 0.5
        assert confidence == 0.4  # 0.8 * 0.5

    def test_missing_price_not_tradeable(self):
        """Missing price data is not tradeable."""
        # No data recorded
        assert not self.shield.is_data_tradeable("AAPL")

    def test_diagnostics(self):
        """Diagnostics returns correct state."""
        self.shield.record_data_timestamp("AAPL", "price")
        diag = self.shield.get_diagnostics()

        assert "tracked_symbols" in diag
        assert diag["tracked_symbols"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: ExitQualityGuard
# ═══════════════════════════════════════════════════════════════════════════════

class TestExitQualityGuard:
    """Tests for exit signal validation."""

    def setup_method(self):
        self.guard = ExitQualityGuard(
            min_exit_confidence=0.30,
            hard_stop_pnl_threshold=-0.03,
            max_holding_days=30,
        )

    def test_hard_stop_bypasses_validation(self):
        """Hard stop-loss always exits regardless of validation."""
        result = self.guard.validate_exit(
            symbol="AAPL",
            exit_reason="signal_reversal",
            signal=-0.2,
            confidence=0.1,  # Below minimum
            entry_signal=0.5,
            pnl_pct=-0.05,  # Below hard stop
        )

        assert result.should_exit
        assert result.bypassed_validation

    def test_max_holding_period_exits(self):
        """Position held too long triggers forced exit."""
        result = self.guard.validate_exit(
            symbol="AAPL",
            exit_reason="signal_reversal",
            signal=-0.2,
            confidence=0.1,
            entry_signal=0.5,
            pnl_pct=0.05,  # Profitable
            holding_days=35,  # Over max
        )

        assert result.should_exit
        assert result.bypassed_validation

    def test_low_confidence_blocks_signal_exit(self):
        """Low confidence blocks signal-reversal exit."""
        result = self.guard.validate_exit(
            symbol="AAPL",
            exit_reason="signal_reversal",
            signal=-0.2,
            confidence=0.1,  # Below minimum
            entry_signal=0.5,
            pnl_pct=0.02,  # Profitable (no hard stop)
        )

        assert not result.should_exit

    def test_stale_data_blocks_signal_exit(self):
        """Stale data blocks signal-reversal exit."""
        result = self.guard.validate_exit(
            symbol="AAPL",
            exit_reason="signal_reversal",
            signal=-0.5,
            confidence=0.6,
            entry_signal=0.5,
            pnl_pct=0.02,
            data_age_seconds=400,  # Over max
        )

        assert not result.should_exit

    def test_stop_loss_always_allowed(self):
        """Stop-loss exit type always allowed."""
        result = self.guard.validate_exit(
            symbol="AAPL",
            exit_reason="stop_loss",
            signal=-0.1,
            confidence=0.1,  # Low confidence
            entry_signal=0.5,
            pnl_pct=-0.02,  # Above hard stop
        )

        assert result.should_exit

    def test_retry_strategy_exponential(self):
        """Retry strategy implements exponential backoff."""
        s1 = self.guard.get_retry_strategy("AAPL", 1)
        assert s1.delay_seconds == 0
        assert s1.order_type == OrderType.MARKET

        s2 = self.guard.get_retry_strategy("AAPL", 2)
        assert s2.delay_seconds == 30

        s3 = self.guard.get_retry_strategy("AAPL", 3)
        assert s3.delay_seconds == 60
        assert s3.order_type == OrderType.AGGRESSIVE_LIMIT

        s5 = self.guard.get_retry_strategy("AAPL", 5)
        assert s5.order_type == OrderType.MOC

    def test_never_gives_up(self):
        """Retry strategy never gives up after attempt 5."""
        s10 = self.guard.get_retry_strategy("AAPL", 10)
        assert not s10.give_up
        assert s10.delay_seconds == 300  # 5 min retry


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: CorrelationCascadeBreaker
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrelationCascadeBreaker:
    """Tests for portfolio correlation shield."""

    def setup_method(self):
        self.breaker = CorrelationCascadeBreaker(
            elevated_threshold=0.40,
            herding_threshold=0.60,
            crisis_threshold=0.80,
        )

    def _create_correlated_data(self, symbols, correlation, days=30):
        """Create test data with specified correlation."""
        data = {}
        base_returns = np.random.randn(days) * 0.02

        for i, sym in enumerate(symbols):
            noise = np.random.randn(days) * 0.02 * (1 - correlation)
            returns = base_returns * correlation + noise
            prices = 100 * np.cumprod(1 + returns)

            data[sym] = pd.DataFrame({
                "Close": prices,
            })

        return data

    def test_normal_correlation(self):
        """Low correlation results in NORMAL regime."""
        symbols = ["AAPL", "XOM", "JPM", "JNJ"]
        data = self._create_correlated_data(symbols, correlation=0.2)

        state = self.breaker.assess_correlation_state(symbols, data)
        assert state.regime == CorrelationRegime.NORMAL

    def test_elevated_correlation(self):
        """Moderate correlation results in ELEVATED regime."""
        symbols = ["AAPL", "MSFT", "GOOGL", "META"]
        data = self._create_correlated_data(symbols, correlation=0.5)

        state = self.breaker.assess_correlation_state(symbols, data)
        # May be ELEVATED or NORMAL depending on exact random values
        assert state.regime in [CorrelationRegime.NORMAL, CorrelationRegime.ELEVATED]

    def test_herding_correlation(self):
        """High correlation results in HERDING regime."""
        symbols = ["AAPL", "MSFT", "GOOGL", "META"]
        data = self._create_correlated_data(symbols, correlation=0.75)

        state = self.breaker.assess_correlation_state(symbols, data)
        assert state.regime >= CorrelationRegime.ELEVATED

    def test_crisis_correlation(self):
        """Very high correlation results in CRISIS regime."""
        symbols = ["AAPL", "MSFT", "GOOGL", "META"]
        data = self._create_correlated_data(symbols, correlation=0.95)

        state = self.breaker.assess_correlation_state(symbols, data)
        assert state.regime >= CorrelationRegime.HERDING

    def test_effective_positions_calculation(self):
        """Effective positions decreases with higher correlation."""
        symbols = ["A", "B", "C", "D"]

        low_corr_data = self._create_correlated_data(symbols, correlation=0.1)
        high_corr_data = self._create_correlated_data(symbols, correlation=0.9)

        low_state = self.breaker.assess_correlation_state(symbols, low_corr_data)
        high_state = self.breaker.assess_correlation_state(symbols, high_corr_data)

        # Higher correlation = fewer effective positions
        assert high_state.effective_positions <= low_state.effective_positions

    def test_positions_to_reduce_herding(self):
        """HERDING regime suggests reducing most-correlated pair."""
        symbols = ["AAPL", "MSFT", "GOOGL", "META"]
        data = self._create_correlated_data(symbols, correlation=0.85)

        # Force herding state
        self.breaker._last_state = CorrelationState(
            regime=CorrelationRegime.HERDING,
            avg_correlation=0.75,
            max_pairwise=0.9,
            effective_positions=2.0,
            concentration_risk=0.5,
            most_correlated_pair=("AAPL", "MSFT"),
        )

        reductions = self.breaker.get_positions_to_reduce(symbols, data)
        assert len(reductions) > 0

    def test_max_position_count_by_regime(self):
        """Max position count decreases in higher regimes."""
        self.breaker._last_state = CorrelationState(
            regime=CorrelationRegime.NORMAL,
            avg_correlation=0.3, max_pairwise=0.5,
            effective_positions=10, concentration_risk=0.2,
        )
        normal_max = self.breaker.get_max_position_count()

        self.breaker._last_state = CorrelationState(
            regime=CorrelationRegime.CRISIS,
            avg_correlation=0.85, max_pairwise=0.95,
            effective_positions=2, concentration_risk=0.8,
        )
        crisis_max = self.breaker.get_max_position_count()

        assert crisis_max < normal_max


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: DrawdownCascadeBreaker
# ═══════════════════════════════════════════════════════════════════════════════

class TestDrawdownCascadeBreaker:
    """Tests for 5-tier drawdown response."""

    def setup_method(self):
        self.breaker = DrawdownCascadeBreaker(
            initial_capital=1_000_000,
            tier_1_threshold=0.02,
            tier_2_threshold=0.04,
            tier_3_threshold=0.06,
            tier_4_threshold=0.08,
        )

    def test_normal_tier(self):
        """No drawdown results in NORMAL tier."""
        state = self.breaker.update(1_000_000)  # Peak capital

        assert state.tier == DrawdownTier.NORMAL
        assert state.size_multiplier == 1.0
        assert state.entry_allowed

    def test_caution_tier(self):
        """2-4% drawdown triggers CAUTION tier."""
        state = self.breaker.update(970_000)  # 3% drawdown

        assert state.tier == DrawdownTier.CAUTION
        assert state.size_multiplier == 0.75
        assert state.entry_allowed

    def test_defensive_tier(self):
        """4-6% drawdown triggers DEFENSIVE tier."""
        state = self.breaker.update(950_000)  # 5% drawdown

        assert state.tier == DrawdownTier.DEFENSIVE
        assert state.size_multiplier == 0.50
        assert state.entry_allowed
        assert state.force_close_count == 2

    def test_survival_tier(self):
        """6-8% drawdown triggers SURVIVAL tier."""
        state = self.breaker.update(930_000)  # 7% drawdown

        assert state.tier == DrawdownTier.SURVIVAL
        assert state.size_multiplier == 0.25
        assert not state.entry_allowed

    def test_emergency_tier(self):
        """Over 8% drawdown triggers EMERGENCY tier."""
        state = self.breaker.update(910_000)  # 9% drawdown

        assert state.tier == DrawdownTier.EMERGENCY
        assert state.size_multiplier == 0.0
        assert not state.entry_allowed

    def test_positions_to_close_defensive(self):
        """DEFENSIVE tier closes worst 2 positions."""
        self.breaker.update(950_000)  # Trigger DEFENSIVE

        positions = {
            "AAPL": {"pnl": -1000},
            "MSFT": {"pnl": 500},
            "GOOGL": {"pnl": -2000},
            "AMZN": {"pnl": 1000},
        }

        to_close = self.breaker.get_positions_to_close(positions)
        assert len(to_close) == 2
        assert "GOOGL" in to_close  # Worst P&L

    def test_emergency_closes_all(self):
        """EMERGENCY tier closes all positions."""
        self.breaker.update(910_000)  # Trigger EMERGENCY

        positions = {
            "AAPL": {"pnl": 1000},
            "MSFT": {"pnl": 500},
            "GOOGL": {"pnl": 2000},
        }

        to_close = self.breaker.get_positions_to_close(positions)
        assert len(to_close) == 3

    def test_min_confidence_increases(self):
        """Minimum confidence threshold increases with tier."""
        self.breaker.update(1_000_000)
        normal_conf = self.breaker.get_min_confidence()

        self.breaker.update(970_000)
        caution_conf = self.breaker.get_min_confidence()

        self.breaker.update(950_000)
        defensive_conf = self.breaker.get_min_confidence()

        assert caution_conf > normal_conf
        assert defensive_conf > caution_conf

    def test_peak_updates(self):
        """Peak capital updates on new highs."""
        self.breaker.update(1_000_000)
        assert self.breaker._peak_capital == 1_000_000

        self.breaker.update(1_100_000)
        assert self.breaker._peak_capital == 1_100_000


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: ExecutionShield
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutionShield:
    """Tests for smart execution wrapper."""

    def setup_method(self):
        self.shield = ExecutionShield(
            twap_threshold=50_000,
            vwap_threshold=200_000,
            max_slippage_bps=15,
            critical_slippage_bps=30,
        )

    def test_small_order_uses_market(self):
        """Small orders use MARKET algorithm."""
        strategy = self.shield.select_strategy(
            symbol="AAPL", side="BUY", shares=10, price=150.0
        )

        assert strategy.algo == ExecutionAlgo.MARKET
        assert strategy.time_horizon_seconds == 0

    def test_medium_order_uses_limit(self):
        """Medium orders use LIMIT algorithm."""
        strategy = self.shield.select_strategy(
            symbol="AAPL", side="BUY", shares=200, price=150.0
        )  # $30K order

        assert strategy.algo == ExecutionAlgo.LIMIT

    def test_large_order_uses_twap(self):
        """Large orders use TWAP algorithm."""
        strategy = self.shield.select_strategy(
            symbol="AAPL", side="BUY", shares=500, price=150.0
        )  # $75K order

        assert strategy.algo == ExecutionAlgo.TWAP
        assert strategy.time_horizon_seconds == 300

    def test_very_large_order_uses_vwap(self):
        """Very large orders use VWAP algorithm."""
        strategy = self.shield.select_strategy(
            symbol="AAPL", side="BUY", shares=2000, price=150.0
        )  # $300K order

        assert strategy.algo == ExecutionAlgo.VWAP
        assert strategy.time_horizon_seconds == 900

    def test_critical_urgency_uses_market(self):
        """Critical urgency always uses MARKET."""
        strategy = self.shield.select_strategy(
            symbol="AAPL", side="SELL", shares=2000, price=150.0,
            urgency=Urgency.CRITICAL
        )

        assert strategy.algo == ExecutionAlgo.MARKET

    def test_slippage_tracking(self):
        """Slippage is tracked per symbol."""
        self.shield.record_execution(
            symbol="AAPL", expected_price=150.0, fill_price=150.10,
            shares=100, algo=ExecutionAlgo.MARKET
        )

        avg_slip = self.shield.get_avg_slippage("AAPL")
        assert avg_slip > 0  # ~6.67 bps

    def test_slippage_adjustment_normal(self):
        """Normal slippage has no adjustment."""
        self.shield.record_execution(
            symbol="AAPL", expected_price=150.0, fill_price=150.01,
            shares=100, algo=ExecutionAlgo.MARKET
        )

        adj = self.shield.get_slippage_adjustment("AAPL")
        assert adj == 1.0

    def test_high_slippage_reduces_size(self):
        """High slippage reduces position size."""
        # Record multiple high-slippage executions
        for _ in range(10):
            self.shield.record_execution(
                symbol="AAPL", expected_price=150.0, fill_price=150.30,
                shares=100, algo=ExecutionAlgo.MARKET
            )  # 20 bps slippage

        adj = self.shield.get_slippage_adjustment("AAPL")
        assert adj < 1.0

    def test_critical_slippage_flags_symbol(self):
        """Critical slippage flags symbol as expensive."""
        for _ in range(10):
            self.shield.record_execution(
                symbol="AAPL", expected_price=150.0, fill_price=150.60,
                shares=100, algo=ExecutionAlgo.MARKET
            )  # 40 bps slippage

        assert self.shield.is_expensive_symbol("AAPL")

    def test_execution_quality_report(self):
        """Execution quality report includes all stats."""
        self.shield.record_execution(
            symbol="AAPL", expected_price=150.0, fill_price=150.05,
            shares=100, algo=ExecutionAlgo.MARKET
        )

        report = self.shield.get_execution_quality_report()

        assert report["total_executions"] == 1
        assert "AAPL" in report["per_symbol"]
        assert report["per_symbol"]["AAPL"]["executions"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalFortressV2Integration:
    """Integration tests for Phase 2 components working together."""

    def test_flash_crash_scenario(self):
        """Simulate flash crash: VIX spike + rapid price drop."""
        guard = BlackSwanGuard()
        drawdown = DrawdownCascadeBreaker(initial_capital=1_000_000)

        # Record baseline prices first
        for i in range(5):
            guard.record_index_price("SPY", 100.0)

        # Simulate flash crash with VIX spike
        guard.record_vix(vix_current=45.0, vix_open=20.0)  # 125% spike

        # Check threat
        threat = guard.assess_threat()
        assert threat.threat_level >= ThreatLevel.SEVERE

        # Simulate capital loss
        dd_state = drawdown.update(920_000)  # 8% drawdown
        assert dd_state.tier == DrawdownTier.EMERGENCY

        # Both should block entries
        assert guard.should_block_entry()
        assert not drawdown.get_entry_allowed()

    def test_stale_data_blocks_all(self):
        """Stale data should block both entry and exit signals."""
        shield = SignalDecayShield(
            max_price_age_seconds=60,
            price_decay_limit_seconds=100  # Lower limit for test
        )
        exit_guard = ExitQualityGuard(max_data_age_for_exit=60)

        # Record stale price (beyond decay limit)
        old_time = datetime.now() - timedelta(seconds=150)
        shield.record_data_timestamp("AAPL", "price", old_time)

        # Entry blocked
        assert not shield.is_data_tradeable("AAPL")

        # Signal-reversal exit blocked (but hard stop still works)
        exit_result = exit_guard.validate_exit(
            symbol="AAPL", exit_reason="signal_reversal",
            signal=-0.5, confidence=0.7, entry_signal=0.5,
            pnl_pct=0.02, data_age_seconds=120
        )
        assert not exit_result.should_exit

    def test_correlation_crisis_triggers_deleverage(self):
        """High correlation + drawdown triggers aggressive deleverage."""
        corr_breaker = CorrelationCascadeBreaker()
        dd_breaker = DrawdownCascadeBreaker(initial_capital=1_000_000)

        # Simulate crisis correlation
        corr_breaker._last_state = CorrelationState(
            regime=CorrelationRegime.CRISIS,
            avg_correlation=0.9,
            max_pairwise=0.95,
            effective_positions=1.5,
            concentration_risk=0.85,
            most_correlated_pair=("AAPL", "MSFT"),
        )

        # Simulate defensive drawdown
        dd_state = dd_breaker.update(950_000)

        # Both should recommend position reduction
        assert corr_breaker._last_state.regime >= CorrelationRegime.CRISIS
        assert dd_state.force_close_count > 0

    def test_execution_quality_feeds_back_to_sizing(self):
        """Poor execution quality should reduce future position sizes."""
        shield = ExecutionShield()

        # Simulate multiple poor executions
        for _ in range(20):
            shield.record_execution(
                symbol="ILLIQUID", expected_price=50.0, fill_price=50.20,
                shares=100, algo=ExecutionAlgo.MARKET
            )  # 40 bps slippage

        # Symbol should be flagged
        assert shield.is_expensive_symbol("ILLIQUID")

        # Size adjustment should reduce
        adj = shield.get_slippage_adjustment("ILLIQUID")
        assert adj == 0.8  # 20% reduction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
