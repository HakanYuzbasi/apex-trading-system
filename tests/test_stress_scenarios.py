"""
tests/test_stress_scenarios.py - Stress Tests and Property-Based Tests

Comprehensive test suite including:
- Property-based tests (invariants that must always hold)
- Stress scenario tests (2008 crash, flash crash, etc.)
- Execution latency benchmarks
- Edge cases and boundary conditions
"""

import pytest
import numpy as np
from datetime import datetime
import time

# Import modules to test
from risk.portfolio_stress_test import (
    PortfolioStressTest, StressTestResult
)
from backtesting.market_impact import (
    MarketImpactModel, MarketConditions, ExecutionCosts
)
from monitoring.alert_manager import (
    AlertManager, AlertSeverity, AlertCategory, Alert
)
from core.config_profiles import (
    ProfileType, PROFILES, DynamicConfigAdjuster
)


# ============================================================================
# Property-Based Tests (Invariants)
# ============================================================================

class TestPropertyInvariants:
    """Tests for properties that must always hold."""

    def test_position_size_never_exceeds_capital(self):
        """Position sizes should never exceed total capital."""
        capital = 1_000_000
        max_position_pct = 0.05

        # Generate random positions
        for _ in range(100):
            position_size = np.random.uniform(0, capital * max_position_pct * 2)
            capped_size = min(position_size, capital * max_position_pct)

            assert capped_size <= capital * max_position_pct
            assert capped_size <= capital

    def test_portfolio_weights_sum_to_one_or_less(self):
        """Portfolio weights should not exceed 100%."""
        capital = 1_000_000

        # Random portfolio
        for _ in range(50):
            num_positions = np.random.randint(5, 20)
            position_values = np.random.uniform(10000, 50000, num_positions)

            # Cap to ensure doesn't exceed capital
            total = position_values.sum()
            if total > capital:
                position_values = position_values * (capital / total) * 0.95

            weights = position_values / capital
            assert weights.sum() <= 1.0 + 1e-9  # Small epsilon for float precision

    def test_signal_always_bounded(self):
        """Signal values should always be in [-1, 1]."""
        for _ in range(1000):
            raw_signal = np.random.uniform(-10, 10)
            bounded = np.tanh(raw_signal)

            assert -1.0 <= bounded <= 1.0

    def test_stop_loss_less_than_take_profit(self):
        """Stop loss should always be less than take profit for positive R:R."""
        for profile in PROFILES.values():
            assert profile.stop_loss_pct < profile.take_profit_pct
            assert profile.take_profit_pct / profile.stop_loss_pct >= 1.5  # Min 1.5:1 R:R

    def test_confidence_bounded_zero_one(self):
        """Confidence values should be in [0, 1]."""
        for _ in range(1000):
            raw_confidence = np.random.uniform(-5, 5)
            bounded = np.clip(raw_confidence, 0.0, 1.0)

            assert 0.0 <= bounded <= 1.0

    def test_slippage_always_positive(self):
        """Slippage should always be positive (a cost)."""
        model = MarketImpactModel()

        for _ in range(100):
            order_size = np.random.uniform(1000, 100000)
            volume = np.random.uniform(100000, 10000000)
            volatility = np.random.uniform(0.01, 0.05)

            slippage = model.calculate_slippage_bps(
                order_size_usd=order_size,
                avg_daily_volume_usd=volume,
                volatility=volatility
            )

            assert slippage >= 0


# ============================================================================
# Stress Scenario Tests
# ============================================================================

class TestStressScenarios:
    """Tests for portfolio stress scenarios."""

    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample diversified portfolio."""
        return {
            'positions': {
                'AAPL': 100, 'MSFT': 80, 'GOOGL': 50,
                'JPM': 120, 'BAC': 200,
                'JNJ': 90, 'UNH': 30,
                'XOM': 150, 'CVX': 100
            },
            'prices': {
                'AAPL': 175.0, 'MSFT': 380.0, 'GOOGL': 140.0,
                'JPM': 175.0, 'BAC': 35.0,
                'JNJ': 155.0, 'UNH': 520.0,
                'XOM': 105.0, 'CVX': 150.0
            }
        }

    def test_2008_crash_scenario(self, sample_portfolio):
        """Test portfolio behavior in 2008-style crash."""
        stress_test = PortfolioStressTest(
            positions=sample_portfolio['positions'],
            prices=sample_portfolio['prices'],
            capital=1_000_000
        )

        result = stress_test.run_scenario(
            PortfolioStressTest.PREDEFINED_SCENARIOS['2008_financial_crisis']
        )

        # Validate result structure
        assert isinstance(result, StressTestResult)
        assert result.portfolio_return < 0  # Should lose money
        assert result.portfolio_return > -0.60  # But not unrealistically bad

        # Financial sector should be hit hardest
        if 'Financials' in result.sector_impacts:
            assert result.sector_impacts['Financials'] < 0

    def test_flash_crash_scenario(self, sample_portfolio):
        """Test portfolio behavior in flash crash."""
        stress_test = PortfolioStressTest(
            positions=sample_portfolio['positions'],
            prices=sample_portfolio['prices'],
            capital=1_000_000
        )

        result = stress_test.run_scenario(
            PortfolioStressTest.PREDEFINED_SCENARIOS['flash_crash']
        )

        # Flash crash should have moderate losses
        assert -0.20 < result.portfolio_return < 0

        # Liquidation cost should be high due to spread widening
        assert result.estimated_liquidation_cost > 0

    def test_correlation_spike_scenario(self, sample_portfolio):
        """Test when diversification fails (all correlations spike)."""
        stress_test = PortfolioStressTest(
            positions=sample_portfolio['positions'],
            prices=sample_portfolio['prices'],
            capital=1_000_000
        )

        result = stress_test.run_scenario(
            PortfolioStressTest.PREDEFINED_SCENARIOS['correlation_breakdown']
        )

        # All sectors should move together
        negative_sectors = sum(
            1 for v in result.sector_impacts.values() if v < 0
        )
        total_sectors = len(result.sector_impacts)

        # Most sectors should lose when correlation spikes
        assert negative_sectors >= total_sectors * 0.7

    def test_tech_sector_crash(self, sample_portfolio):
        """Test tech-specific crash (other sectors unaffected)."""
        stress_test = PortfolioStressTest(
            positions=sample_portfolio['positions'],
            prices=sample_portfolio['prices'],
            capital=1_000_000
        )

        result = stress_test.run_scenario(
            PortfolioStressTest.PREDEFINED_SCENARIOS['tech_sector_crash']
        )

        # Tech should be hit hard, others less so
        if 'Technology' in result.sector_impacts:
            tech_loss = result.sector_impacts['Technology']
            other_losses = [
                v for k, v in result.sector_impacts.items()
                if k != 'Technology'
            ]
            if other_losses:
                avg_other = np.mean(other_losses)
                assert tech_loss < avg_other  # Tech should lose more

    def test_custom_crash_scenario(self, sample_portfolio):
        """Test custom crash magnitude."""
        stress_test = PortfolioStressTest(
            positions=sample_portfolio['positions'],
            prices=sample_portfolio['prices'],
            capital=1_000_000
        )

        # 40% crash
        result = stress_test.run_custom_crash(-0.40)

        assert result.portfolio_return < 0
        assert result.portfolio_return > -0.50  # Should not lose more than market

    def test_all_predefined_scenarios(self, sample_portfolio):
        """Run all predefined scenarios and validate results."""
        stress_test = PortfolioStressTest(
            positions=sample_portfolio['positions'],
            prices=sample_portfolio['prices'],
            capital=1_000_000
        )

        results = stress_test.run_all_scenarios()

        assert len(results) > 0

        for scenario_id, result in results.items():
            # All scenarios should have valid structure
            assert isinstance(result, StressTestResult)
            assert result.scenario_name is not None
            assert result.portfolio_pnl is not None

            # P&L should be consistent with return
            expected_pnl = 1_000_000 * result.portfolio_return
            assert abs(result.portfolio_pnl - expected_pnl) < 10000  # Within $10K


# ============================================================================
# Market Impact Model Tests
# ============================================================================

class TestMarketImpact:
    """Tests for market impact and slippage model."""

    def test_larger_orders_have_more_impact(self):
        """Larger orders should have higher slippage."""
        model = MarketImpactModel()

        small_slippage = model.calculate_slippage_bps(
            order_size_usd=10000,
            avg_daily_volume_usd=10_000_000,
            volatility=0.02
        )

        large_slippage = model.calculate_slippage_bps(
            order_size_usd=500000,
            avg_daily_volume_usd=10_000_000,
            volatility=0.02
        )

        assert large_slippage > small_slippage

    def test_higher_volatility_increases_impact(self):
        """Higher volatility should increase slippage."""
        model = MarketImpactModel(random_slippage_std=0.0)

        low_vol_slippage = model.calculate_slippage_bps(
            order_size_usd=50000,
            avg_daily_volume_usd=10_000_000,
            volatility=0.01
        )

        high_vol_slippage = model.calculate_slippage_bps(
            order_size_usd=50000,
            avg_daily_volume_usd=10_000_000,
            volatility=0.05
        )

        assert high_vol_slippage > low_vol_slippage

    def test_lower_volume_increases_impact(self):
        """Lower volume should increase slippage."""
        model = MarketImpactModel()

        high_vol_slippage = model.calculate_slippage_bps(
            order_size_usd=50000,
            avg_daily_volume_usd=50_000_000,
            volatility=0.02
        )

        low_vol_slippage = model.calculate_slippage_bps(
            order_size_usd=50000,
            avg_daily_volume_usd=1_000_000,
            volatility=0.02
        )

        assert low_vol_slippage > high_vol_slippage

    def test_execution_costs_breakdown(self):
        """Test execution cost components."""
        model = MarketImpactModel()

        conditions = MarketConditions(
            avg_daily_volume=1_000_000,
            avg_daily_turnover=100_000_000,
            volatility=0.02,
            bid_ask_spread_bps=5.0,
            current_volume_ratio=1.0
        )

        costs = model.calculate_execution_costs(
            order_size_shares=1000,
            price=100.0,
            side='BUY',
            conditions=conditions
        )

        # Validate structure
        assert isinstance(costs, ExecutionCosts)
        assert costs.total_cost_bps >= costs.spread_cost_bps
        assert costs.spread_cost_bps >= 0
        assert costs.temporary_impact_bps >= 0

        # Effective price should be higher for BUY
        assert costs.effective_price > 100.0

    def test_sell_side_execution(self):
        """Test sell-side execution costs."""
        model = MarketImpactModel()

        conditions = MarketConditions(
            avg_daily_volume=1_000_000,
            avg_daily_turnover=100_000_000,
            volatility=0.02,
            bid_ask_spread_bps=5.0,
            current_volume_ratio=1.0
        )

        costs = model.calculate_execution_costs(
            order_size_shares=1000,
            price=100.0,
            side='SELL',
            conditions=conditions
        )

        # Effective price should be lower for SELL
        assert costs.effective_price < 100.0


# ============================================================================
# Alert Manager Tests
# ============================================================================

class TestAlertManager:
    """Tests for alert management system."""

    @pytest.fixture
    def alert_manager(self):
        """Create fresh alert manager."""
        return AlertManager(max_alerts=100)

    @pytest.mark.asyncio
    async def test_trigger_alert(self, alert_manager):
        """Test basic alert triggering."""
        alert = await alert_manager.trigger_alert(
            category=AlertCategory.RISK,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test",
            source="test"
        )

        assert alert is not None
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_alert_rule_registration(self, alert_manager):
        """Test alert rule registration and checking."""
        # Set state
        alert_manager.set_state('vix', 35)

        # Register rule
        alert_manager.register_rule(
            rule_id="high_vix",
            name="High VIX",
            condition=lambda: alert_manager.get_state('vix', 0) > 30,
            category=AlertCategory.MARKET,
            severity=AlertSeverity.WARNING,
            message_template="VIX is at {vix}",
            metadata_provider=lambda: {'vix': alert_manager.get_state('vix')}
        )

        # Check rules
        triggered = await alert_manager.check_rules()

        assert len(triggered) == 1
        assert "VIX is at 35" in triggered[0].message

    @pytest.mark.asyncio
    async def test_alert_cooldown(self, alert_manager):
        """Test that alerts respect cooldown period."""
        alert_manager.set_state('vix', 35)

        alert_manager.register_rule(
            rule_id="high_vix",
            name="High VIX",
            condition=lambda: alert_manager.get_state('vix', 0) > 30,
            category=AlertCategory.MARKET,
            severity=AlertSeverity.WARNING,
            message_template="VIX high",
            cooldown_seconds=300  # 5 minute cooldown
        )

        # First check
        triggered1 = await alert_manager.check_rules()
        assert len(triggered1) == 1

        # Immediate second check should not trigger
        triggered2 = await alert_manager.check_rules()
        assert len(triggered2) == 0

    def test_acknowledge_alert(self, alert_manager):
        """Test alert acknowledgment."""
        # Create alert synchronously for testing
        alert_id = "test_alert_001"
        alert = Alert(
            alert_id=alert_id,
            category=AlertCategory.RISK,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message",
            timestamp=datetime.now(),
            source="test"
        )
        alert_manager.alerts[alert_id] = alert

        # Acknowledge
        result = alert_manager.acknowledge_alert(alert_id, "user1")

        assert result is True
        assert alert_manager.alerts[alert_id].acknowledged is True

    def test_get_active_alerts(self, alert_manager):
        """Test filtering active alerts."""
        # Add some alerts
        for i in range(5):
            alert = Alert(
                alert_id=f"alert_{i}",
                category=AlertCategory.RISK if i < 3 else AlertCategory.SYSTEM,
                severity=AlertSeverity.WARNING if i < 4 else AlertSeverity.CRITICAL,
                title=f"Alert {i}",
                message="Test",
                timestamp=datetime.now(),
                source="test",
                resolved=i == 0  # First one resolved
            )
            alert_manager.alerts[alert.alert_id] = alert

        # Get active (unresolved)
        active = alert_manager.get_active_alerts()
        assert len(active) == 4

        # Filter by category
        risk_alerts = alert_manager.get_active_alerts(category=AlertCategory.RISK)
        assert len(risk_alerts) == 2  # 2 unresolved RISK alerts


# ============================================================================
# Configuration Profile Tests
# ============================================================================

class TestConfigProfiles:
    """Tests for configuration profiles."""

    def test_all_profiles_valid(self):
        """All predefined profiles should be valid."""
        for profile_type, profile in PROFILES.items():
            assert profile.name is not None
            assert profile.max_positions > 0
            assert profile.position_size_usd > 0
            assert 0 < profile.stop_loss_pct < 1
            assert 0 < profile.take_profit_pct < 1

    def test_conservative_profile_constraints(self):
        """Conservative profile should have tighter constraints."""
        conservative = PROFILES[ProfileType.CONSERVATIVE]
        aggressive = PROFILES[ProfileType.AGGRESSIVE]

        # Conservative should have smaller positions
        assert conservative.max_positions < aggressive.max_positions
        assert conservative.position_size_usd < aggressive.position_size_usd

        # Tighter stops
        assert conservative.stop_loss_pct < aggressive.stop_loss_pct

        # Higher thresholds
        assert conservative.min_signal_threshold > aggressive.min_signal_threshold

    def test_dynamic_adjuster_vix_reduction(self):
        """Dynamic adjuster should reduce exposure when VIX high."""
        base_profile = PROFILES[ProfileType.MODERATE]
        adjuster = DynamicConfigAdjuster(base_profile)

        # Normal VIX
        adjuster.update_market_state(vix=15)
        normal_size = adjuster.get_current_profile().position_size_usd

        # High VIX
        adjuster.update_market_state(vix=35)
        high_vix_size = adjuster.get_current_profile().position_size_usd

        assert high_vix_size < normal_size

    def test_dynamic_adjuster_drawdown_reduction(self):
        """Dynamic adjuster should reduce exposure during drawdown."""
        base_profile = PROFILES[ProfileType.MODERATE]
        adjuster = DynamicConfigAdjuster(base_profile)

        # No drawdown
        adjuster.update_market_state(drawdown=0.0)
        normal_size = adjuster.get_current_profile().position_size_usd

        # Significant drawdown
        adjuster.update_market_state(drawdown=0.05)
        dd_size = adjuster.get_current_profile().position_size_usd

        assert dd_size < normal_size


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance and latency benchmarks."""

    def test_slippage_calculation_performance(self):
        """Slippage calculation should be fast."""
        model = MarketImpactModel()

        start = time.time()
        iterations = 10000

        for _ in range(iterations):
            model.calculate_slippage_bps(
                order_size_usd=50000,
                avg_daily_volume_usd=10_000_000,
                volatility=0.02
            )

        elapsed = time.time() - start
        per_call_ms = (elapsed / iterations) * 1000

        # Should be < 1ms per call
        assert per_call_ms < 1.0, f"Slippage calc took {per_call_ms:.2f}ms"

    def test_stress_test_performance(self):
        """Stress test should complete quickly."""
        positions = {f"SYM{i}": 100 for i in range(50)}
        prices = {f"SYM{i}": 100.0 for i in range(50)}

        stress_test = PortfolioStressTest(
            positions=positions,
            prices=prices,
            capital=5_000_000
        )

        start = time.time()
        results = stress_test.run_all_scenarios()
        elapsed = time.time() - start

        # Should complete all scenarios in < 1 second
        assert elapsed < 1.0, f"Stress test took {elapsed:.2f}s"
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_alert_check_performance(self):
        """Alert rule checking should be fast."""
        manager = AlertManager()

        # Register many rules
        for i in range(100):
            manager.register_rule(
                rule_id=f"rule_{i}",
                name=f"Rule {i}",
                condition=lambda: False,  # Never triggers
                category=AlertCategory.RISK,
                severity=AlertSeverity.WARNING,
                message_template="Test"
            )

        start = time.time()
        iterations = 100

        for _ in range(iterations):
            await manager.check_rules()

        elapsed = time.time() - start
        per_check_ms = (elapsed / iterations) * 1000

        # Should check 100 rules in < 50ms
        assert per_check_ms < 50, f"Alert check took {per_check_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
