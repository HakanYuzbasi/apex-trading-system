# tests/test_circuit_breaker.py - Circuit breaker and stress test scenarios

import pytest
from models.risk_management import CircuitBreaker
import numpy as np


class TestCircuitBreaker:
    """Test circuit breaker under extreme market conditions."""

    @pytest.fixture
    def circuit_breaker(self, mock_ibkr):
        """Create circuit breaker with mock IBKR."""
        breaker = CircuitBreaker(
            ibkr_connector=mock_ibkr,
            max_daily_loss=50_000,      # $50k max daily loss
            max_position_loss=20_000,   # $20k max per position
            max_volatility=0.05,        # 5% max move per minute
        )
        return breaker

    @pytest.mark.asyncio
    async def test_circuit_breaker_normal_market(self, circuit_breaker, mock_ibkr):
        """Test circuit breaker allows trading in normal conditions."""
        # Setup - normal market conditions
        mock_ibkr.get_portfolio_value.return_value = 1_100_000.0
        
        # Execute
        can_trade = await circuit_breaker.check_if_can_trade()
        
        # Assert
        assert can_trade is True
        print("✅ test_circuit_breaker_normal_market passed")

    @pytest.mark.asyncio
    async def test_circuit_breaker_high_volatility_stop(self, circuit_breaker, mock_ibkr):
        """Test circuit breaker halts trading during volatility spike (2008 scenario)."""
        # Setup - extreme volatility (5% drop in 1 minute)
        circuit_breaker.prices = np.array([100.0] * 60 + [95.0])  # Last price is 5% down
        
        # Execute
        can_trade = await circuit_breaker.check_if_can_trade()
        
        # Assert
        assert can_trade is False
        print("✅ test_circuit_breaker_high_volatility_stop passed")

    @pytest.mark.asyncio
    async def test_circuit_breaker_daily_loss_limit(self, circuit_breaker, mock_ibkr):
        """Test circuit breaker triggers on daily loss limit reached."""
        # Setup - portfolio down $50k (at max loss limit)
        circuit_breaker.starting_value = 1_100_000.0
        mock_ibkr.get_portfolio_value.return_value = 1_050_000.0  # Down $50k
        
        # Execute
        can_trade = await circuit_breaker.check_if_can_trade()
        
        # Assert
        assert can_trade is False
        print("✅ test_circuit_breaker_daily_loss_limit passed")

    @pytest.mark.asyncio
    async def test_circuit_breaker_position_loss_limit(self, circuit_breaker, mock_ibkr):
        """Test circuit breaker closes position if loss > limit."""
        # Setup - single position down $25k (exceeds $20k limit)
        position = {
            "symbol": "AAPL",
            "quantity": 100,
            "entry_price": 150.0,
            "current_price": 150.0,
        }
        # Price drops to create $25k loss
        position["current_price"] = 100.0  # $50 loss per share = $5000... adjust
        # Actually: need 100 shares * $200 loss = $20k limit
        position["entry_price"] = 250.0
        position["current_price"] = 50.0  # $200 loss per share
        
        # Execute
        should_close = circuit_breaker.should_close_position(position)
        
        # Assert
        assert should_close is True
        print("✅ test_circuit_breaker_position_loss_limit passed")

    @pytest.mark.asyncio
    async def test_circuit_breaker_gap_down_detection(self, circuit_breaker):
        """Test circuit breaker detects gap-down opening (e.g., earnings miss)."""
        # Setup - gap down 8% at market open
        circuit_breaker.prices = np.array([100.0] * 60 + [92.0])  # 8% gap down
        
        # Execute
        gap_detected = circuit_breaker.detect_gap_down(threshold=0.05)  # 5% threshold
        
        # Assert
        assert gap_detected is True
        print("✅ test_circuit_breaker_gap_down_detection passed")

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_after_halt(self, circuit_breaker, mock_ibkr):
        """Test circuit breaker resumes trading after volatility decreases."""
        # Setup - initial halt
        circuit_breaker.prices = np.array([100.0] * 60 + [95.0])  # 5% drop
        can_trade_before = await circuit_breaker.check_if_can_trade()
        assert can_trade_before is False  # Halted
        
        # Now prices stabilize
        circuit_breaker.prices = np.array([95.0] * 30)  # Stable at 95
        
        # Execute
        can_trade_after = await circuit_breaker.check_if_can_trade()
        
        # Assert
        assert can_trade_after is True
        print("✅ test_circuit_breaker_recovery_after_halt passed")
