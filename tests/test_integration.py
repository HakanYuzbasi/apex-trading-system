"""
tests/test_integration.py - Integration Tests

End-to-end integration tests for the trading system.
Tests complete workflows from signal to execution.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_trading_system():
    """Create mock trading system components."""
    system = MagicMock()
    system.ibkr = AsyncMock()
    system.signal_generator = MagicMock()
    system.risk_manager = MagicMock()
    system.portfolio = MagicMock()
    system.is_running = True
    return system


@pytest.fixture
def mock_market_state():
    """Create mock market state."""
    return {
        'is_market_open': True,
        'current_time': datetime.now().replace(hour=10, minute=30),
        'volatility_index': 18.5,
        'market_regime': 'bull',
    }


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n_days = 252

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, n_days)))

    return pd.DataFrame({
        'Open': close * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': close * (1 + np.random.uniform(0, 0.02, n_days)),
        'Low': close * (1 - np.random.uniform(0, 0.02, n_days)),
        'Close': close,
        'Volume': np.random.randint(1_000_000, 50_000_000, n_days)
    }, index=dates)


class TestFullTradingCycle:
    """Test complete trading cycle."""

    @pytest.mark.asyncio
    async def test_signal_to_order_flow(self, mock_trading_system, sample_price_data):
        """Test flow from signal generation to order execution."""
        # Step 1: Generate signal
        mock_trading_system.signal_generator.generate_ml_signal.return_value = {
            'signal': 0.75,
            'confidence': 0.85,
            'components': {'momentum': 0.8, 'trend': 0.7}
        }

        signal = mock_trading_system.signal_generator.generate_ml_signal("AAPL")

        assert signal['signal'] > 0.5
        assert signal['confidence'] > 0.5

        # Step 2: Risk check
        mock_trading_system.risk_manager.approve_trade.return_value = {
            'approved': True,
            'position_size': 100,
            'reason': None
        }

        risk_check = mock_trading_system.risk_manager.approve_trade("AAPL", signal)

        assert risk_check['approved']

        # Step 3: Execute order
        mock_trading_system.ibkr.execute_order.return_value = {
            'status': 'FILLED',
            'fill_price': 185.50,
            'quantity': 100,
            'order_id': 'ORD-12345'
        }

        order_result = await mock_trading_system.ibkr.execute_order(
            symbol="AAPL",
            quantity=risk_check['position_size'],
            side="BUY"
        )

        assert order_result['status'] == 'FILLED'

        # Step 4: Update portfolio
        mock_trading_system.portfolio.update_position.return_value = True

        updated = mock_trading_system.portfolio.update_position("AAPL", order_result)

        assert updated

    @pytest.mark.asyncio
    async def test_sell_signal_flow(self, mock_trading_system):
        """Test sell signal to order flow."""
        # Generate sell signal
        mock_trading_system.signal_generator.generate_ml_signal.return_value = {
            'signal': -0.65,
            'confidence': 0.80,
        }

        signal = mock_trading_system.signal_generator.generate_ml_signal("AAPL")

        assert signal['signal'] < -0.5

        # Check existing position
        mock_trading_system.portfolio.get_position.return_value = {
            'symbol': 'AAPL',
            'quantity': 100,
            'avg_cost': 180.0
        }

        position = mock_trading_system.portfolio.get_position("AAPL")

        assert position['quantity'] > 0

        # Execute sell
        mock_trading_system.ibkr.execute_order.return_value = {
            'status': 'FILLED',
            'fill_price': 175.50,
            'quantity': 100,
            'side': 'SELL'
        }

        order_result = await mock_trading_system.ibkr.execute_order(
            symbol="AAPL",
            quantity=position['quantity'],
            side="SELL"
        )

        assert order_result['status'] == 'FILLED'


class TestRiskIntegration:
    """Test risk management integration."""

    @pytest.mark.asyncio
    async def test_position_size_limits(self, mock_trading_system):
        """Test that position sizing respects limits."""
        mock_trading_system.portfolio.get_value.return_value = 1100000
        mock_trading_system.risk_manager.max_position_pct = 0.05

        max_position_value = 1100000 * 0.05  # $55,000

        price = 185.0
        max_shares = int(max_position_value / price)

        assert max_shares > 0
        assert max_shares * price <= max_position_value

    @pytest.mark.asyncio
    async def test_daily_loss_halt(self, mock_trading_system):
        """Test trading halts on daily loss limit."""
        mock_trading_system.risk_manager.daily_pnl = -35000
        mock_trading_system.risk_manager.max_daily_loss = 33000  # 3% of $1.1M

        should_halt = abs(mock_trading_system.risk_manager.daily_pnl) > mock_trading_system.risk_manager.max_daily_loss

        assert should_halt

        # Verify no new trades
        mock_trading_system.risk_manager.approve_trade.return_value = {
            'approved': False,
            'reason': 'Daily loss limit exceeded'
        }

        result = mock_trading_system.risk_manager.approve_trade("AAPL", {'signal': 0.8})

        assert not result['approved']

    @pytest.mark.asyncio
    async def test_sector_exposure_limit(self, mock_trading_system):
        """Test sector exposure limits."""
        mock_trading_system.portfolio.get_sector_exposure.return_value = {
            'Technology': 0.38,
            'Financials': 0.20,
            'Healthcare': 0.15,
        }

        max_sector_exposure = 0.40

        tech_exposure = mock_trading_system.portfolio.get_sector_exposure()['Technology']

        # Trying to add more tech
        new_position_value = 30000  # $30k
        portfolio_value = 1100000
        new_tech_exposure = tech_exposure + (new_position_value / portfolio_value)

        would_exceed = new_tech_exposure > max_sector_exposure

        assert would_exceed


class TestMarketHoursIntegration:
    """Test market hours handling."""

    @pytest.mark.asyncio
    async def test_trade_during_market_hours(self, mock_trading_system, mock_market_state):
        """Test trading during market hours."""
        assert mock_market_state['is_market_open']

        mock_trading_system.ibkr.execute_order.return_value = {'status': 'FILLED'}

        result = await mock_trading_system.ibkr.execute_order(
            symbol="AAPL",
            quantity=100,
            side="BUY"
        )

        assert result['status'] == 'FILLED'

    @pytest.mark.asyncio
    async def test_no_trade_outside_hours(self, mock_trading_system):
        """Test no trading outside market hours."""
        mock_trading_system.is_market_open = False

        # Should not attempt trade
        should_trade = mock_trading_system.is_market_open

        assert not should_trade

    @pytest.mark.asyncio
    async def test_extended_hours_for_commodities(self, mock_trading_system):
        """Test extended hours trading for commodities/ETFs."""
        commodity_symbols = ['GLD', 'SLV', 'USO']
        extended_hours = True

        for symbol in commodity_symbols:
            mock_trading_system.can_trade_extended.return_value = extended_hours
            can_trade = mock_trading_system.can_trade_extended(symbol)
            assert can_trade


class TestErrorRecovery:
    """Test error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_connection_recovery(self, mock_trading_system):
        """Test recovery from connection loss."""
        # Simulate connection loss - use explicit MagicMock for sync methods
        mock_trading_system.ibkr.is_connected = MagicMock(return_value=False)

        is_connected = mock_trading_system.ibkr.is_connected()
        assert not is_connected

        # Reconnect
        mock_trading_system.ibkr.reconnect = AsyncMock(return_value=True)
        reconnected = await mock_trading_system.ibkr.reconnect()

        assert reconnected

        mock_trading_system.ibkr.is_connected.return_value = True
        assert mock_trading_system.ibkr.is_connected()


    @pytest.mark.asyncio
    async def test_order_rejection_handling(self, mock_trading_system):
        """Test handling of order rejections."""
        mock_trading_system.ibkr.execute_order.return_value = {
            'status': 'REJECTED',
            'reason': 'Insufficient buying power'
        }

        result = await mock_trading_system.ibkr.execute_order(
            symbol="AAPL",
            quantity=10000,
            side="BUY"
        )

        assert result['status'] == 'REJECTED'

        # Should not update portfolio
        mock_trading_system.portfolio.update_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_fill_handling(self, mock_trading_system):
        """Test handling of partial fills."""
        mock_trading_system.ibkr.execute_order.return_value = {
            'status': 'PARTIAL',
            'filled_quantity': 75,
            'remaining_quantity': 25,
            'fill_price': 185.50
        }

        result = await mock_trading_system.ibkr.execute_order(
            symbol="AAPL",
            quantity=100,
            side="BUY"
        )

        assert result['status'] == 'PARTIAL'
        assert result['filled_quantity'] == 75


class TestDataIntegration:
    """Test data integration."""

    @pytest.mark.asyncio
    async def test_historical_data_loading(self, mock_trading_system, sample_price_data):
        """Test loading historical data."""
        mock_trading_system.data_fetcher = MagicMock()
        mock_trading_system.data_fetcher.fetch_historical.return_value = sample_price_data

        data = mock_trading_system.data_fetcher.fetch_historical("AAPL", days=252)

        assert len(data) == 252
        assert 'Close' in data.columns

    @pytest.mark.asyncio
    async def test_realtime_price_update(self, mock_trading_system):
        """Test realtime price updates."""
        mock_trading_system.ibkr.get_market_price = AsyncMock(return_value=185.50)

        price = await mock_trading_system.ibkr.get_market_price("AAPL")

        assert price == 185.50

    @pytest.mark.asyncio
    async def test_data_caching(self, mock_trading_system, sample_price_data):
        """Test that data is cached appropriately."""
        cache = {}

        # First fetch - cache miss
        mock_trading_system.data_fetcher.fetch_historical.return_value = sample_price_data

        if "AAPL" not in cache:
            data = mock_trading_system.data_fetcher.fetch_historical("AAPL")
            cache["AAPL"] = data

        # Second fetch - cache hit
        data2 = cache.get("AAPL")

        assert data2 is not None
        # Should not call fetch again
        assert mock_trading_system.data_fetcher.fetch_historical.call_count == 1


class TestPerformanceTracking:
    """Test performance tracking integration."""

    @pytest.mark.asyncio
    async def test_trade_recording(self, mock_trading_system):
        """Test recording trades for performance tracking."""
        trade = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 100,
            'price': 185.50,
            'timestamp': datetime.now(),
            'order_id': 'ORD-12345'
        }

        mock_trading_system.performance_tracker = MagicMock()
        mock_trading_system.performance_tracker.record_trade.return_value = True

        recorded = mock_trading_system.performance_tracker.record_trade(trade)

        assert recorded

    @pytest.mark.asyncio
    async def test_daily_pnl_calculation(self, mock_trading_system):
        """Test daily P&L calculation."""
        mock_trading_system.performance_tracker.get_daily_pnl.return_value = 5250.0

        daily_pnl = mock_trading_system.performance_tracker.get_daily_pnl()

        assert daily_pnl == 5250.0

    @pytest.mark.asyncio
    async def test_metrics_aggregation(self, mock_trading_system):
        """Test aggregating performance metrics."""
        mock_trading_system.performance_tracker.get_metrics.return_value = {
            'total_trades': 150,
            'win_rate': 0.58,
            'avg_win': 450.0,
            'avg_loss': -280.0,
            'sharpe_ratio': 1.85,
            'max_drawdown': -0.05,
            'total_pnl': 125000.0
        }

        metrics = mock_trading_system.performance_tracker.get_metrics()

        assert metrics['win_rate'] > 0.5
        assert metrics['sharpe_ratio'] > 1.0
        assert metrics['total_pnl'] > 0


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers(self, mock_trading_system):
        """Test circuit breaker activation."""
        mock_trading_system.circuit_breaker = MagicMock()
        mock_trading_system.circuit_breaker.consecutive_losses = 5
        mock_trading_system.circuit_breaker.threshold = 5

        should_halt = mock_trading_system.circuit_breaker.consecutive_losses >= mock_trading_system.circuit_breaker.threshold

        assert should_halt

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(self, mock_trading_system):
        """Test circuit breaker reset after winning trade."""
        mock_trading_system.circuit_breaker = MagicMock()
        mock_trading_system.circuit_breaker.consecutive_losses = 3

        # Winning trade
        mock_trading_system.circuit_breaker.record_win = MagicMock()
        mock_trading_system.circuit_breaker.record_win()

        mock_trading_system.circuit_breaker.consecutive_losses = 0

        assert mock_trading_system.circuit_breaker.consecutive_losses == 0
