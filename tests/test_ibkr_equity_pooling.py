"""Tests for IBKR equity pooling parity and stale fallback behavior."""

from __future__ import annotations

import threading
from datetime import datetime

import pytest

from core.exceptions import ApexBrokerError
from models.broker import BrokerConnection, BrokerType
from services.broker.service import BrokerService


async def _async_val(v):
    """Return v as a coroutine (used to stub async methods in monkeypatch)."""
    return v


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
    # Bypass lease_manager to avoid stale Redis connections from previous test event loops
    monkeypatch.setattr("services.broker.service.lease_manager.allocate", lambda preferred_id=None, ttl=None: _async_val(12))
    monkeypatch.setattr("services.broker.service.lease_manager.release", lambda client_id: _async_val(None))

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
    monkeypatch.setattr("services.broker.service.lease_manager.allocate", lambda preferred_id=None, ttl=None: _async_val(12))
    monkeypatch.setattr("services.broker.service.lease_manager.release", lambda client_id: _async_val(None))
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
    monkeypatch.setattr("services.broker.service.lease_manager.allocate", lambda preferred_id=None, ttl=None: _async_val(12))
    monkeypatch.setattr("services.broker.service.lease_manager.release", lambda client_id: _async_val(None))

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

        @property
        def errorEvent(self):
            class _Event:
                def __iadd__(self, callback):
                    return self
            return _Event()

        def isConnected(self) -> bool:  # noqa: N802
            return False

    monkeypatch.setattr("services.broker.service.IB", FakeIB)

    with pytest.raises(ApexBrokerError) as exc_info:
        BrokerService._fetch_ibkr_equity_blocking({"host": "127.0.0.1", "port": 7497, "client_id": 1}, client_id=1)

    assert exc_info.value.code == "IBKR_CONNECT_FAILED"


def test_fetch_ibkr_equity_blocking_creates_event_loop_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            return [SummaryRow("NetLiquidation", "888.0")]

        def isConnected(self) -> bool:  # noqa: N802
            return self._connected

        def disconnect(self) -> None:
            self._connected = False

    monkeypatch.setattr("services.broker.service.IB", FakeIB)
    result: dict[str, float] = {}
    failure: dict[str, BaseException] = {}

    def run_fetch() -> None:
        try:
            result["value"] = BrokerService._fetch_ibkr_equity_blocking(
                {"host": "127.0.0.1", "port": 7497, "client_id": 1},
                client_id=1,
            )
        except BaseException as exc:  # pragma: no cover - assertion surface
            failure["error"] = exc

    thread = threading.Thread(target=run_fetch)
    thread.start()
    thread.join(timeout=3)

    assert thread.is_alive() is False
    assert "error" not in failure
    assert result["value"] == 888.0


@pytest.mark.asyncio
async def test_fetch_connection_equity_retries_client_id_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = BrokerService()
    connection = _ibkr_connection()
    monkeypatch.setattr(
        service,
        "_decrypt_credentials",
        lambda _: {"host": "127.0.0.1", "port": 7497, "client_id": 12},
    )

    lease_ids = iter([211, 212])

    async def fake_allocate(preferred_id=None, ttl=None):
        return next(lease_ids)

    async def fake_release(client_id):
        return None

    monkeypatch.setattr("services.broker.service.lease_manager.allocate", fake_allocate)
    monkeypatch.setattr("services.broker.service.lease_manager.release", fake_release)

    attempts = {"count": 0}

    def fake_fetch(creds, client_id):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ApexBrokerError(
                code="IBKR_CLIENT_ID_IN_USE",
                message="IBKR rejected the API client ID because it is already in use",
                context={"client_id": client_id},
            )
        return 987.65

    monkeypatch.setattr(service, "_fetch_ibkr_equity_blocking", fake_fetch)

    snapshot = await service._fetch_equity_with_fallback(connection)

    assert attempts["count"] == 2
    assert snapshot.value == 987.65
    assert snapshot.stale is False


def test_fetch_ibkr_equity_blocking_surfaces_paper_disclaimer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self._callbacks = []

        def __iadd__(self, callback):
            self._callbacks.append(callback)
            return self

        def fire(self, req_id: int, error_code: int, error_string: str, contract=None) -> None:
            for callback in self._callbacks:
                callback(req_id, error_code, error_string, contract)

    class FakeIB:
        def __init__(self) -> None:
            self._connected = False
            self.errorEvent = FakeEvent()

        def connect(self, host: str, port: int, clientId: int, timeout: int, readonly: bool) -> None:  # noqa: N803
            self.errorEvent.fire(-1, 10141, "Paper trading disclaimer must first be accepted for API connection.")
            raise RuntimeError("connection reset by peer")

        def isConnected(self) -> bool:  # noqa: N802
            return self._connected

    monkeypatch.setattr("services.broker.service.IB", FakeIB)

    with pytest.raises(ApexBrokerError) as exc_info:
        BrokerService._fetch_ibkr_equity_blocking(
            {"host": "127.0.0.1", "port": 7497, "client_id": 1},
            client_id=1,
        )

    assert exc_info.value.code == "IBKR_PAPER_DISCLAIMER_REQUIRED"
