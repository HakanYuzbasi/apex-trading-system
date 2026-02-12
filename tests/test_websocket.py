"""WebSocket integration tests for the APEX Trading API."""
import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.server import app, manager, ConnectionManager
from api.auth import USER_STORE, AUTH_CONFIG


@pytest.fixture(autouse=True)
def _disable_auth():
    """Disable auth for WebSocket tests."""
    original = AUTH_CONFIG.enabled
    AUTH_CONFIG.enabled = False
    yield
    AUTH_CONFIG.enabled = original


class TestConnectionManager:
    """Test the ConnectionManager class."""

    @pytest.mark.asyncio
    async def test_manager_has_lock(self):
        """Manager should have an asyncio lock for thread safety."""
        mgr = ConnectionManager()
        assert hasattr(mgr, "_lock")
        assert mgr.active_connections == []

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self):
        """Broadcast should clean up dead connections."""
        mgr = ConnectionManager()
        dead_ws = MagicMock()
        dead_ws.send_json = MagicMock(side_effect=Exception("closed"))
        async with mgr._lock:
            mgr.active_connections.append(dead_ws)

        await mgr.broadcast({"type": "test"})
        assert dead_ws not in mgr.active_connections


class TestWebSocketEndpoint:
    """Test the /ws WebSocket endpoint."""

    def test_ws_connect_and_receive_state(self):
        """WebSocket should accept connection and send initial state."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "state_update"
            assert "capital" in data
            assert "positions" in data
            assert "initial_capital" in data
            assert "starting_capital" in data
            assert "max_positions" in data

    def test_ws_sends_valid_json(self):
        """WebSocket initial state should be valid JSON with expected keys."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            expected_keys = {
                "type", "timestamp", "capital", "initial_capital",
                "starting_capital", "positions", "daily_pnl", "total_pnl",
                "max_drawdown", "sharpe_ratio", "win_rate",
                "sector_exposure", "open_positions", "max_positions",
                "total_trades",
            }
            assert expected_keys.issubset(set(data.keys()))


class TestRESTEndpoints:
    """Smoke tests for REST endpoints (complementary to WebSocket tests)."""

    def test_status_endpoint(self):
        """GET /status should return valid status."""
        client = TestClient(app)
        admin = USER_STORE.get_user("admin")
        resp = client.get("/status", headers={"X-API-Key": admin.api_key})
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "capital" in body

    def test_positions_endpoint(self):
        """GET /positions should return a list."""
        client = TestClient(app)
        admin = USER_STORE.get_user("admin")
        resp = client.get("/positions", headers={"X-API-Key": admin.api_key})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_sectors_endpoint(self):
        """GET /sectors should return sector data."""
        client = TestClient(app)
        admin = USER_STORE.get_user("admin")
        resp = client.get("/sectors", headers={"X-API-Key": admin.api_key})
        assert resp.status_code == 200
