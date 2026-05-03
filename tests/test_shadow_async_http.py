import pytest
import httpx
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from quant_system.portfolio.shadow_accounting import ShadowAccounting

@pytest.fixture
def mock_dependencies():
    event_bus = MagicMock()
    ledger = MagicMock()
    ledger.positions = {}
    reconciler = MagicMock()
    return event_bus, ledger, reconciler

@pytest.mark.asyncio
async def test_http_client_is_async_instance(mock_dependencies):
    event_bus, ledger, reconciler = mock_dependencies
    shadow = ShadowAccounting(event_bus, ledger, reconciler)
    
    try:
        assert isinstance(shadow._http, httpx.AsyncClient)
    finally:
        await shadow.close()

@pytest.mark.asyncio
async def test_force_sync_uses_async_client(mock_dependencies):
    event_bus, ledger, reconciler = mock_dependencies
    shadow = ShadowAccounting(event_bus, ledger, reconciler)
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.json.return_value = [{"symbol": "AAPL", "qty": "10"}]
    mock_response.raise_for_status = MagicMock()
    
    with patch.object(shadow._http, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        result = await shadow.force_sync(reason="test")
        
        # Verify the mock was awaited
        mock_get.assert_awaited_once()
        assert shadow.shadow_positions["AAPL"] == 10.0
        assert result["reason"] == "test"
    
    await shadow.close()

@pytest.mark.asyncio
async def test_close_awaits_http_client_aclose(mock_dependencies):
    event_bus, ledger, reconciler = mock_dependencies
    shadow = ShadowAccounting(event_bus, ledger, reconciler)
    
    with patch.object(shadow._http, "aclose", new_callable=AsyncMock) as mock_aclose:
        await shadow.close()
        mock_aclose.assert_awaited_once()
