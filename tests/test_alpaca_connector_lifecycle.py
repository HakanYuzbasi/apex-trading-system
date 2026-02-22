import asyncio

import pytest

from execution.alpaca_connector import AlpacaConnector


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_disconnect_clears_client_and_get_market_price_uses_fallback() -> None:
    connector = AlpacaConnector(api_key="k", secret_key="s")
    fake_client = _FakeClient()
    connector._client = fake_client
    connector._connected = True

    connector.disconnect()
    await asyncio.sleep(0)

    assert connector._client is None
    assert connector.is_connected() is False
    assert fake_client.closed is True

    connector._fallback_price = lambda normalized: 123.45  # type: ignore[method-assign]
    price = await connector.get_market_price("ETH/USD")
    assert price == 123.45
