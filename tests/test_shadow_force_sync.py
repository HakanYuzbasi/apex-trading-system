import os
import json
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
from quant_system.portfolio.shadow_accounting import ShadowAccounting
from quant_system.risk.manager import RiskManager

@pytest.fixture
def shadow_accounting():
    bus = MagicMock()
    ledger = MagicMock()
    ledger.positions = {}
    reconciler = MagicMock()
    sa = ShadowAccounting(bus, ledger, reconciler)
    return sa

@pytest.fixture
def risk_manager(shadow_accounting):
    bus = MagicMock()
    ledger = MagicMock()
    rm = RiskManager(ledger, bus, shadow_accounting=shadow_accounting)
    return rm

@pytest.mark.asyncio
async def test_force_sync_replaces_ledger_with_broker_truth(shadow_accounting):
    """
    Mock Alpaca REST /v2/positions response with known positions.
    Call force_sync(). Assert self.shadow_positions matches broker response.
    """
    mock_positions = [
        {"symbol": "AAPL", "qty": "10"},
        {"symbol": "TSLA", "qty": "-5"}
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = mock_positions
    mock_response.raise_for_status.return_value = None

    shadow_accounting._http.get = AsyncMock(return_value=mock_response)

    await shadow_accounting.force_sync("test")

    assert shadow_accounting.shadow_positions["AAPL"] == 10.0
    assert shadow_accounting.shadow_positions["TSLA"] == -5.0

@pytest.mark.asyncio
async def test_force_sync_resets_pnl_accumulators(shadow_accounting):
    """
    Set intraday PnL to non-zero value. Call force_sync().
    Assert all PnL accumulators reset to 0.0.
    """
    shadow_accounting._intraday_pnl = 123.45

    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_response.raise_for_status.return_value = None
    shadow_accounting._http.get = AsyncMock(return_value=mock_response)

    await shadow_accounting.force_sync("test")

    assert shadow_accounting._intraday_pnl == 0.0

@pytest.mark.asyncio
async def test_force_sync_logs_warn_with_before_after_diff(shadow_accounting):
    """
    Call force_sync() with known before state. Assert a WARN log
    was emitted containing both the before and after snapshots.
    """
    shadow_accounting.shadow_positions = {"MSFT": 100.0}

    mock_response = MagicMock()
    mock_response.json.return_value = [{"symbol": "MSFT", "qty": "50"}]
    mock_response.raise_for_status.return_value = None
    shadow_accounting._http.get = AsyncMock(return_value=mock_response)

    with patch("quant_system.portfolio.shadow_accounting.logger.warning") as mock_log:
        await shadow_accounting.force_sync("test_reason")

        log_call = mock_log.call_args[0][0]
        assert "test_reason" in log_call
        assert "MSFT" in log_call

def test_force_sync_called_after_circuit_breaker_trigger(risk_manager, shadow_accounting):
    """
    Mock circuit breaker trigger. Assert force_sync() is called
    exactly once with reason="circuit_breaker".
    """
    risk_manager.portfolio_ledger.positions = {}

    with patch.object(shadow_accounting, 'force_sync') as mock_sync:
        risk_manager._emit_flatten_all()
        mock_sync.assert_called_once_with("circuit_breaker")

def test_force_sync_called_even_if_flatten_partial_failure(risk_manager, shadow_accounting):
    """
    Mock flatten-all to raise an exception.
    Assert force_sync() is still called (try/finally behavior).
    """
    risk_manager.portfolio_ledger.positions = {"FAKE": MagicMock(quantity=10)}

    with patch.object(risk_manager, '_dispatch_order', side_effect=Exception("Order Failure")):
        with patch.object(shadow_accounting, 'force_sync') as mock_sync:
            try:
                risk_manager._emit_flatten_all()
            except Exception:
                pass
            mock_sync.assert_called_once_with("circuit_breaker")
