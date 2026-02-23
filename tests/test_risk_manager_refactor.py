import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from risk.risk_manager import RiskManager

@pytest.fixture
def risk_manager():
    # Patch load_state to avoid reading from disk during test init
    with patch('risk.risk_session.RiskSession.load_state'), \
         patch('risk.risk_session.RiskSession.save_state'):
        rm = RiskManager(max_daily_loss=0.05, max_drawdown=0.10)
        # Reset default session to known clean state
        rm.sessions["default"].starting_capital = 10000.0
        rm.sessions["default"].peak_capital = 10000.0
        rm.sessions["default"].day_start_capital = 10000.0
        rm.sessions["default"].circuit_breaker.is_tripped = False
        rm.sessions["default"].circuit_breaker.failures = 0
        return rm

def test_default_session_backward_compatibility(risk_manager):
    """Ensure RiskManager API still works for default session."""
    assert risk_manager.default_max_daily_loss == 0.05
    
    # Test property proxies
    # Verify initial state from fixture
    assert risk_manager.starting_capital == 10000.0
    
    # Set new value (now allowed by RiskSession modification)
    risk_manager.starting_capital = 12000
    assert risk_manager.sessions["default"].starting_capital == 12000
    assert risk_manager.starting_capital == 12000
    
    risk_manager.day_start_capital = 10500
    assert risk_manager.day_start_capital == 10500

def test_multi_user_sessions(risk_manager):
    """Ensure multiple users have isolated state."""
    # User 1
    session1 = risk_manager._get_or_create_session("user1")
    session1.set_starting_capital(10000)
    
    # User 2
    session2 = risk_manager._get_or_create_session("user2")
    session2.set_starting_capital(50000)
    
    assert risk_manager.sessions["user1"].starting_capital == 10000
    assert risk_manager.sessions["user2"].starting_capital == 50000

@pytest.mark.asyncio
async def test_check_aggregate_risk(risk_manager):
    """Verify aggregate risk check logic."""
    
    # Mock broker service
    with patch('risk.risk_manager.broker_service') as mock_service:
        # Scenario 1: Good to trade
        mock_service.get_total_equity = AsyncMock(return_value=100000.0)
        
        # Set session state for user_test
        session = risk_manager._get_or_create_session("user_test")
        session.starting_capital = 100000.0
        session.day_start_capital = 100000.0
        session.peak_capital = 100000.0
        session.circuit_breaker.is_tripped = False
        
        result = await risk_manager.check_aggregate_risk("user_test")
        
        # Debug output if assertion fails
        if not result["allowed"]:
            print(f"DEBUG FAIL REASON: {result['reason']}")
            print(f"DEBUG METRICS: {result['metrics']}")
            
        assert result["allowed"] is True
        assert result["metrics"]["total_equity"] == 100000.0
        
        # Scenario 2: Daily Loss Breached
        mock_service.get_total_equity = AsyncMock(return_value=90000.0) # 10% loss > 5% limit
        
        # check_aggregate_risk calls check_daily_loss which updates state
        result = await risk_manager.check_aggregate_risk("user_test")
        
        assert result["allowed"] is False
        assert "Daily loss" in result["reason"]
        assert result["metrics"]["daily_loss"]["breached"] is True

if __name__ == "__main__":
    asyncio.run(test_check_aggregate_risk(RiskManager()))
