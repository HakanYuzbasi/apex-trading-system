"""Functional tenant isolation tests for websocket streaming manager."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from api.ws_manager import ConnectionManager


class DummyWebSocket:
    """Minimal websocket stub for manager tests."""

    def __init__(self) -> None:
        self.state = SimpleNamespace()
        self.client = ("127.0.0.1", 0)
        self.messages: list[str] = []

    async def accept(self) -> None:
        return None

    async def send_text(self, payload: str) -> None:
        self.messages.append(payload)


@pytest.mark.asyncio
async def test_tenant_broadcast_isolation() -> None:
    manager = ConnectionManager()
    tenant_a_ws = DummyWebSocket()
    tenant_b_ws = DummyWebSocket()

    await manager.connect(tenant_a_ws, tenant_id="tenant-a")
    await manager.connect(tenant_b_ws, tenant_id="tenant-b")

    await manager.broadcast_to_tenant("tenant-a", {"type": "state_update", "tenant": "tenant-a", "aggregated_equity": 101.0})

    assert len(tenant_a_ws.messages) == 1
    assert len(tenant_b_ws.messages) == 0

    payload = json.loads(tenant_a_ws.messages[0])
    assert payload["tenant"] == "tenant-a"
    assert payload["aggregated_equity"] == 101.0
