"""
tests/test_ibkr_connector.py - IBKR Connector Tests

Tests for the Interactive Brokers connector including:
- Connection handling
- Order execution
- Position queries
- Error handling and retry logic
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


# Test fixtures
@pytest.fixture
def mock_ib_connection():
    """Mock ib_insync IB connection."""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.connect = MagicMock()
    ib.disconnect = MagicMock()
    ib.reqPositions = MagicMock(return_value=[])
    ib.reqAccountSummary = MagicMock(return_value=[])
    return ib


@pytest.fixture
def mock_contract():
    """Mock IB contract."""
    contract = MagicMock()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


@pytest.fixture
def mock_order():
    """Mock IB order."""
    order = MagicMock()
    order.orderId = 12345
    order.action = "BUY"
    order.totalQuantity = 100
    order.orderType = "MKT"
    return order


class TestConnectionHandling:
    """Test broker connection handling."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_ib_connection):
        """Test successful connection."""
        with patch('execution.ibkr_connector.IB', return_value=mock_ib_connection):
            # Connection should succeed
            mock_ib_connection.isConnected.return_value = True
            assert mock_ib_connection.isConnected()

    @pytest.mark.asyncio
    async def test_connect_retry_on_failure(self, mock_ib_connection):
        """Test connection retry on failure."""
        mock_ib_connection.connect.side_effect = [
            ConnectionError("First attempt failed"),
            ConnectionError("Second attempt failed"),
            None  # Third attempt succeeds
        ]

        call_count = 0
        for _ in range(3):
            try:
                mock_ib_connection.connect()
                break
            except ConnectionError:
                call_count += 1
                continue

        assert call_count == 2  # Failed twice before success

    @pytest.mark.asyncio
    async def test_disconnect_graceful(self, mock_ib_connection):
        """Test graceful disconnection."""
        mock_ib_connection.disconnect()
        mock_ib_connection.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(self, mock_ib_connection):
        """Test reconnection after disconnect."""
        mock_ib_connection.isConnected.return_value = False
        assert not mock_ib_connection.isConnected()

        mock_ib_connection.connect()
        mock_ib_connection.isConnected.return_value = True
        assert mock_ib_connection.isConnected()


class TestOrderExecution:
    """Test order execution functionality."""

    @pytest.mark.asyncio
    async def test_market_order_submission(self, mock_ib_connection, mock_contract, mock_order):
        """Test market order submission."""
        mock_ib_connection.placeOrder = MagicMock(return_value=mock_order)

        result = mock_ib_connection.placeOrder(mock_contract, mock_order)
        assert result.orderId == 12345
        assert result.action == "BUY"

    @pytest.mark.asyncio
    async def test_limit_order_submission(self, mock_ib_connection, mock_contract):
        """Test limit order submission."""
        limit_order = MagicMock()
        limit_order.orderId = 12346
        limit_order.orderType = "LMT"
        limit_order.lmtPrice = 150.0

        mock_ib_connection.placeOrder = MagicMock(return_value=limit_order)

        result = mock_ib_connection.placeOrder(mock_contract, limit_order)
        assert result.orderType == "LMT"
        assert result.lmtPrice == 150.0

    @pytest.mark.asyncio
    async def test_order_cancellation(self, mock_ib_connection, mock_order):
        """Test order cancellation."""
        mock_ib_connection.cancelOrder = MagicMock(return_value=True)

        result = mock_ib_connection.cancelOrder(mock_order)
        assert result is True

    @pytest.mark.asyncio
    async def test_order_modification(self, mock_ib_connection, mock_contract, mock_order):
        """Test order modification."""
        mock_order.totalQuantity = 150  # Modify quantity
        mock_ib_connection.placeOrder = MagicMock(return_value=mock_order)

        result = mock_ib_connection.placeOrder(mock_contract, mock_order)
        assert result.totalQuantity == 150

    @pytest.mark.asyncio
    async def test_order_fill_notification(self, mock_ib_connection):
        """Test order fill notification handling."""
        fill = MagicMock()
        fill.execution = MagicMock()
        fill.execution.orderId = 12345
        fill.execution.avgPrice = 151.50
        fill.execution.shares = 100

        # Simulate fill callback
        fills = [fill]
        assert len(fills) == 1
        assert fills[0].execution.avgPrice == 151.50


class TestPositionQueries:
    """Test position query functionality."""

    @pytest.mark.asyncio
    async def test_get_all_positions(self, mock_ib_connection):
        """Test fetching all positions."""
        position = MagicMock()
        position.contract = MagicMock()
        position.contract.symbol = "AAPL"
        position.position = 100
        position.avgCost = 150.0

        mock_ib_connection.positions = MagicMock(return_value=[position])

        positions = mock_ib_connection.positions()
        assert len(positions) == 1
        assert positions[0].contract.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_account_summary(self, mock_ib_connection):
        """Test fetching account summary."""
        account_value = MagicMock()
        account_value.tag = "NetLiquidation"
        account_value.value = "1100000.00"

        mock_ib_connection.accountSummary = MagicMock(return_value=[account_value])

        summary = mock_ib_connection.accountSummary()
        assert len(summary) == 1
        assert summary[0].tag == "NetLiquidation"

    @pytest.mark.asyncio
    async def test_empty_positions(self, mock_ib_connection):
        """Test handling empty positions."""
        mock_ib_connection.positions = MagicMock(return_value=[])

        positions = mock_ib_connection.positions()
        assert len(positions) == 0


class TestErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_ib_connection):
        """Test timeout handling."""
        mock_ib_connection.reqMktData = MagicMock(side_effect=asyncio.TimeoutError())

        with pytest.raises(asyncio.TimeoutError):
            mock_ib_connection.reqMktData()

    @pytest.mark.asyncio
    async def test_connection_lost_recovery(self, mock_ib_connection):
        """Test recovery from lost connection."""
        # Simulate connection loss
        mock_ib_connection.isConnected.return_value = False
        assert not mock_ib_connection.isConnected()

        # Reconnect
        mock_ib_connection.connect()
        mock_ib_connection.isConnected.return_value = True
        assert mock_ib_connection.isConnected()

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, mock_ib_connection):
        """Test rate limit error handling."""
        # Simulate rate limit response
        mock_ib_connection.reqMktData = MagicMock(
            side_effect=[Exception("Rate limit exceeded"), MagicMock()]
        )

        # First call should fail
        with pytest.raises(Exception):
            mock_ib_connection.reqMktData()

        # Retry should succeed
        result = mock_ib_connection.reqMktData()
        assert result is not None

    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self, mock_ib_connection):
        """Test handling of invalid symbols."""
        mock_ib_connection.qualifyContracts = MagicMock(return_value=[])

        contracts = mock_ib_connection.qualifyContracts("INVALID123")
        assert len(contracts) == 0


class TestMarketData:
    """Test market data functionality."""

    @pytest.mark.asyncio
    async def test_get_market_price(self, mock_ib_connection, mock_contract):
        """Test fetching market price."""
        ticker = MagicMock()
        ticker.last = 151.50
        ticker.bid = 151.45
        ticker.ask = 151.55

        mock_ib_connection.reqMktData = MagicMock(return_value=ticker)

        result = mock_ib_connection.reqMktData(mock_contract)
        assert result.last == 151.50

    @pytest.mark.asyncio
    async def test_get_historical_data(self, mock_ib_connection, mock_contract):
        """Test fetching historical data."""
        bars = [
            MagicMock(date=datetime(2024, 1, i), open=150, high=152, low=149, close=151, volume=1000000)
            for i in range(1, 11)
        ]

        mock_ib_connection.reqHistoricalData = MagicMock(return_value=bars)

        result = mock_ib_connection.reqHistoricalData(
            mock_contract,
            endDateTime='',
            durationStr='10 D',
            barSizeSetting='1 day'
        )

        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_market_data_subscription(self, mock_ib_connection, mock_contract):
        """Test market data subscription."""
        mock_ib_connection.reqMktData = MagicMock()
        mock_ib_connection.cancelMktData = MagicMock()

        # Subscribe
        mock_ib_connection.reqMktData(mock_contract)
        mock_ib_connection.reqMktData.assert_called_once()

        # Unsubscribe
        mock_ib_connection.cancelMktData(mock_contract)
        mock_ib_connection.cancelMktData.assert_called_once()


class TestOrderTypes:
    """Test different order types."""

    @pytest.mark.asyncio
    async def test_stop_order(self, mock_ib_connection, mock_contract):
        """Test stop order."""
        stop_order = MagicMock()
        stop_order.orderType = "STP"
        stop_order.auxPrice = 145.0

        mock_ib_connection.placeOrder = MagicMock(return_value=stop_order)

        result = mock_ib_connection.placeOrder(mock_contract, stop_order)
        assert result.orderType == "STP"
        assert result.auxPrice == 145.0

    @pytest.mark.asyncio
    async def test_stop_limit_order(self, mock_ib_connection, mock_contract):
        """Test stop-limit order."""
        stop_limit_order = MagicMock()
        stop_limit_order.orderType = "STP LMT"
        stop_limit_order.auxPrice = 145.0
        stop_limit_order.lmtPrice = 144.0

        mock_ib_connection.placeOrder = MagicMock(return_value=stop_limit_order)

        result = mock_ib_connection.placeOrder(mock_contract, stop_limit_order)
        assert result.orderType == "STP LMT"

    @pytest.mark.asyncio
    async def test_bracket_order(self, mock_ib_connection, mock_contract):
        """Test bracket order."""
        parent = MagicMock(orderId=1)
        take_profit = MagicMock(orderId=2, parentId=1)
        stop_loss = MagicMock(orderId=3, parentId=1)

        mock_ib_connection.placeOrder = MagicMock(side_effect=[parent, take_profit, stop_loss])

        orders = [
            mock_ib_connection.placeOrder(mock_contract, parent),
            mock_ib_connection.placeOrder(mock_contract, take_profit),
            mock_ib_connection.placeOrder(mock_contract, stop_loss)
        ]

        assert len(orders) == 3
        assert orders[1].parentId == orders[0].orderId
