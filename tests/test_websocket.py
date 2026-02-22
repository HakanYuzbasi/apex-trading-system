"""WebSocket integration tests for the APEX Trading API."""
import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.server import app
from api.ws_manager import ConnectionManager
from api.auth import AUTH_CONFIG


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
    async def test_manager_starts_empty(self):
        """Manager should start with empty connections list."""
        mgr = ConnectionManager()
        assert mgr.active_connections == []

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(self):
        """Broadcast should clean up dead connections."""
        mgr = ConnectionManager()
        dead_ws = MagicMock()
        dead_ws.send_text = MagicMock(side_effect=RuntimeError("closed"))
        dead_ws.client = "test-client"
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
        """GET /status should return valid status (auth disabled by fixture)."""
        client = TestClient(app)
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "capital" in body

    def test_positions_endpoint(self):
        """GET /positions should return a list (auth disabled by fixture)."""
        client = TestClient(app)
        resp = client.get("/positions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_sectors_endpoint(self):
        """GET /sectors should return sector data (auth disabled by fixture)."""
        client = TestClient(app)
        resp = client.get("/sectors")
        assert resp.status_code == 200

    def test_status_endpoint_sanitizes_outlier_metrics(self):
        """GET /status should clamp malformed KPI outliers for all consumers."""
        client = TestClient(app)
        with patch("api.server.read_trading_state", return_value={
            "timestamp": "2026-02-22T12:00:00",
            "capital": 1_000_000,
            "starting_capital": 900_000,
            "daily_pnl": "bad",
            "total_pnl": 100_000,
            "max_drawdown": -9999,
            "sharpe_ratio": "-92962852034076208.00",
            "win_rate": 58,
            "open_positions": -3,
            "option_positions": "2",
            "open_positions_total": 1,
            "total_trades": 9_000_000,
        }):
            resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sharpe_ratio"] == 0
        assert body["max_drawdown"] == 0
        assert body["win_rate"] == 0.58
        assert body["open_positions"] == 0
        assert body["option_positions"] == 2
        assert body["open_positions_total"] == 2
        assert body["total_trades"] == 0
        assert body["daily_pnl"] == 0

    def test_public_metrics_endpoint_sanitizes_outliers(self):
        """GET /public/metrics should emit bounded KPIs even with malformed state."""
        client = TestClient(app)
        with patch("api.routers.public.read_trading_state", return_value={
            "timestamp": "not-a-timestamp",
            "capital": "1e20",
            "starting_capital": 100000,
            "daily_pnl": 1200,
            "total_pnl": "broken",
            "max_drawdown": -500,
            "sharpe_ratio": 88,
            "win_rate": 102,
            "open_positions": "7",
            "option_positions": -1,
            "open_positions_total": "NaN",
            "total_trades": 42,
        }):
            resp = client.get("/public/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "offline"
        assert body["capital"] == 0
        assert body["starting_capital"] == 100000
        assert body["total_pnl"] == 0
        assert body["max_drawdown"] == 0
        assert body["sharpe_ratio"] == 0
        assert body["win_rate"] == 0
        assert body["open_positions"] == 7
        assert body["option_positions"] == 0
        assert body["open_positions_total"] == 7
        assert body["total_trades"] == 42
