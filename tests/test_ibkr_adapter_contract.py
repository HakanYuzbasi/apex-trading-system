import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch

from execution.ibkr_adapter import IBKRAdapter, IBKRAdapterException, CircuitBreakerOpenException

class MockConnector:
    def __init__(self):
        self.connect = AsyncMock()
        self.disconnect = AsyncMock()
        self.is_connected = lambda: True
        
        # Mock wrapped methods
        self.get_market_price = AsyncMock(return_value=150.0)
        self.get_position = AsyncMock(return_value=10.0)
        self.get_all_positions = AsyncMock(return_value={"AAPL": 10.0})
        self.get_portfolio_value = AsyncMock(return_value=100000.0)

@pytest.fixture
def adapter():
    connector = MockConnector()
    adapter = IBKRAdapter(connector)
    # Fast timeouts for testing
    adapter._failure_threshold = 3
    adapter._recovery_timeout_sec = 0.5
    adapter._stale_max_age = 5.0
    return adapter

@pytest.mark.asyncio
async def test_successful_call_caches_value(adapter):
    price = await adapter.get_market_price("AAPL")
    assert price == 150.0
    assert "get_market_price::symbol=AAPL" in adapter._value_cache

@pytest.mark.asyncio
async def test_timeout_triggers_stale_fallback(adapter):
    # Seed cache
    await adapter.get_market_price("AAPL")
    
    # Simulate timeout natively in asyncio
    adapter.connector.get_market_price = AsyncMock(side_effect=asyncio.TimeoutError("Simulated timeout breaker"))
    
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        # Should fallback to 150.0 instead of crashing explicitly
        price = await adapter.get_market_price("AAPL")
        
    assert price == 150.0
    assert adapter._error_count == 1

@pytest.mark.asyncio
async def test_circuit_breaker_trips_and_recovers(adapter):
    adapter.connector.get_all_positions = AsyncMock(side_effect=Exception("Simulated strict connection drop"))
    
    # 1. Trip the breaker (threshold = 3) via global block failure (no fallback allowed)
    for _ in range(3):
        with pytest.raises(IBKRAdapterException):
            await adapter.get_all_positions()
            
    assert adapter._circuit_open is True
    assert adapter._error_count == 3
    
    # 2. Assert CircuitBreakerOpenException cuts calls immediately
    with pytest.raises(CircuitBreakerOpenException):
        await adapter.get_all_positions()
        
    # 3. Wait for recovery timeout (0.5s)
    await asyncio.sleep(0.6)
    
    # 4. Breaker enters HALF-OPEN, try call (still failing)
    with pytest.raises(IBKRAdapterException):
        await adapter.get_all_positions()
        
    assert adapter._circuit_open is True # Tripped again immediately
    
    # 5. Wait for recovery, then simulate broker recovery (SUCCESS)
    await asyncio.sleep(0.6)
    adapter.connector.get_all_positions = AsyncMock(return_value={"AAPL": 10.0})
    
    positions = await adapter.get_all_positions()
    assert positions == {"AAPL": 10.0}
    assert adapter._circuit_open is False
    assert adapter._error_count == 0

@pytest.mark.asyncio
async def test_stale_fallback_expires(adapter):
    # Seed cache
    await adapter.get_position("TSLA")
    
    # Manually expire cache
    cache_key = "get_position::symbol=TSLA"
    val, _ = adapter._value_cache[cache_key]
    adapter._value_cache[cache_key] = (val, time.time() - 10.0) # Older than max age 5.0
    
    adapter.connector.get_position = AsyncMock(side_effect=Exception("Simulated strict connection drop"))
    
    # Fallback should fail (too old), bubbling the Exception
    with pytest.raises(IBKRAdapterException):
        await adapter.get_position("TSLA")
