"""Tests for IBKR equity pooling parity and stale fallback behavior."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.exceptions import ApexBrokerError
from models.broker import BrokerConnection, BrokerType
from services.broker.service import BrokerService


def _ibkr_connection() -> BrokerConnection:
    return BrokerConnection(
        id="ibkr-1",
        user_id="tenant-1",
        broker_type=BrokerType.IBKR,
        name="IBKR Primary",
        environment="paper",
        client_id=12,
        credentials={"data": "encrypted"},
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_ibkr_equity_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    service = BrokerService()
    connection = _ibkr_connection()

    monkeypatch.setattr(service, "_decrypt_credentials", lambda _: {"host": "127.0.0.1", "port": 7497, "client_id": 12})
    monkeypatch.setattr(service, "_fetch_ibkr_equity_blocking", lambda creds, client_id: 250000.5)

    snapshot = await service._fetch_equity_with_fallback(connection)

    assert snapshot.broker == "ibkr"
    assert snapshot.value == 250000.5
    assert snapshot.stale is False


@pytest.mark.asyncio
async def test_ibkr_equity_failure_uses_stale_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    service = BrokerService()
    connection = _ibkr_connection()

    monkeypatch.setattr(service, "_decrypt_credentials", lambda _: {"host": "127.0.0.1", "port": 7497, "client_id": 12})
    monkeypatch.setattr(service, "_fetch_ibkr_equity_blocking", lambda creds, client_id: 111.25)
    first = await service._fetch_equity_with_fallback(connection)
    assert first.stale is False

    def raise_fetch(creds, client_id):
        raise RuntimeError("ibkr unavailable")

    monkeypatch.setattr(service, "_fetch_ibkr_equity_blocking", raise_fetch)

    stale = await service._fetch_equity_with_fallback(connection)
    assert stale.value == 111.25
    assert stale.stale is True
    assert stale.broker == "ibkr"


@pytest.mark.asyncio
async def test_ibkr_equity_failure_without_cache_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    service = BrokerService()
    connection = _ibkr_connection()

    monkeypatch.setattr(service, "_decrypt_credentials", lambda _: {"host": "127.0.0.1", "port": 7497, "client_id": 12})

    def raise_fetch(creds, client_id):
        raise RuntimeError("ibkr unavailable")

    monkeypatch.setattr(service, "_fetch_ibkr_equity_blocking", raise_fetch)

    with pytest.raises(ApexBrokerError, match="Unable to fetch equity"):
        await service._fetch_equity_with_fallback(connection)


def test_fetch_ibkr_equity_blocking_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class SummaryRow:
        def __init__(self, tag: str, value: str) -> None:
            self.tag = tag
            self.value = value

    class FakeIB:
        def __init__(self) -> None:
            self._connected = False

        def connect(self, host: str, port: int, clientId: int, timeout: int, readonly: bool) -> None:  # noqa: N803
            self._connected = True

        def accountSummary(self):  # noqa: N802
            return [SummaryRow("NetLiquidation", "12345.67")]

        def isConnected(self) -> bool:  # noqa: N802
            return self._connected

        def disconnect(self) -> None:
            self._connected = False

    monkeypatch.setattr("services.broker.service.IB", FakeIB)
    value = BrokerService._fetch_ibkr_equity_blocking({"host": "127.0.0.1", "port": 7497, "client_id": 1}, client_id=1)
    assert value == 12345.67


def test_fetch_ibkr_equity_blocking_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeIB:
        def connect(self, host: str, port: int, clientId: int, timeout: int, readonly: bool) -> None:  # noqa: N803
            raise RuntimeError("connect failed")

        def isConnected(self) -> bool:  # noqa: N802
            return False

    monkeypatch.setattr("services.broker.service.IB", FakeIB)

    with pytest.raises(RuntimeError, match="connect failed"):
        BrokerService._fetch_ibkr_equity_blocking({"host": "127.0.0.1", "port": 7497, "client_id": 1}, client_id=1)
