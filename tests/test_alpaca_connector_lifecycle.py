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


@pytest.mark.asyncio
async def test_execute_order_timeout_cancel_does_not_crash_on_none_fill_fields() -> None:
    connector = AlpacaConnector(api_key="k", secret_key="s")
    connector._connected = True
    connector._client = _FakeClient()

    async def fake_request(method: str, path: str, **kwargs):
        if method == "GET" and path == "/v2/account":
            return {
                "cash": "1000",
                "non_marginable_buying_power": "1000",
                "buying_power": "1000",
            }
        if method == "POST" and path == "/v2/orders":
            return {
                "id": "ord-1",
                "status": "accepted",
                "filled_qty": None,
                "filled_avg_price": None,
            }
        if method == "DELETE" and path == "/v2/orders/ord-1":
            return {}
        raise AssertionError(f"Unexpected request: {method} {path}")

    connector._request = fake_request  # type: ignore[method-assign]
    connector.get_market_price = lambda symbol: asyncio.sleep(0, result=100.0)  # type: ignore[method-assign]
    connector.check_spread_gate = lambda symbol: asyncio.sleep(0, result=(True, 0.0, ""))  # type: ignore[method-assign]
    connector._wait_for_fill = lambda order_id, timeout=10: asyncio.sleep(0, result=None)  # type: ignore[method-assign]

    result = await connector.execute_order("CRYPTO:ETH/USD", "BUY", 1.0)

    assert result is None
    assert connector._pending_orders == {}


@pytest.mark.asyncio
async def test_get_portfolio_value_reconnects_when_client_not_ready() -> None:
    connector = AlpacaConnector(api_key="k", secret_key="s")

    async def fake_connect() -> None:
        connector._connected = True
        connector._client = _FakeClient()

    async def fake_request(method: str, path: str, **kwargs):
        assert method == "GET"
        assert path == "/v2/account"
        return {"id": "acct-1", "equity": "321.45"}

    connector.connect = fake_connect  # type: ignore[method-assign]
    connector._request = fake_request  # type: ignore[method-assign]

    value = await connector.get_portfolio_value()

    assert value == 321.45
