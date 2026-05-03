import pytest
import logging
from unittest.mock import MagicMock
from core.alert_aggregator import AlertAggregator

def test_get_instance_returns_registered_instance():
    # Setup
    mock_agg = MagicMock(spec=AlertAggregator)
    AlertAggregator.set_instance(mock_agg)
    
    # Verify
    assert AlertAggregator.get_instance() is mock_agg
    
    # Cleanup
    AlertAggregator._instance = None

def test_get_instance_before_set_raises_runtime_error():
    # Setup
    AlertAggregator._instance = None
    
    # Verify
    with pytest.raises(RuntimeError, match="AlertAggregator not initialized"):
        AlertAggregator.get_instance()

def test_set_instance_replaces_previous_instance():
    # Setup
    mock_agg1 = MagicMock(spec=AlertAggregator)
    mock_agg2 = MagicMock(spec=AlertAggregator)
    
    # Action
    AlertAggregator.set_instance(mock_agg1)
    assert AlertAggregator.get_instance() is mock_agg1
    
    AlertAggregator.set_instance(mock_agg2)
    assert AlertAggregator.get_instance() is mock_agg2
    
    # Cleanup
    AlertAggregator._instance = None

def test_no_bare_global_in_module():
    import core.alert_aggregator as module
    
    # Check module __dict__ for any instance of AlertAggregator
    # (except for the class itself)
    for name, value in module.__dict__.items():
        if isinstance(value, AlertAggregator):
            pytest.fail(f"Found bare global singleton '{name}' of type AlertAggregator in module")
    
    # Ensure legacy names are gone
    assert "_global_aggregator" not in module.__dict__
    assert "get_alert_aggregator" not in module.__dict__
    assert "alert_agg" not in module.__dict__
