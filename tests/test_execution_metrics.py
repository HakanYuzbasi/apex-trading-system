import pytest
import time
from pathlib import Path
from execution.metrics_store import ExecutionMetricsStore
from execution.ibkr_connector import IBKRConnector
from unittest.mock import MagicMock, patch

def test_metrics_store(tmp_path):
    """Test ExecutionMetricsStore functionality."""
    p = tmp_path / "metrics.json"
    store = ExecutionMetricsStore(p)
    
    # Init
    metrics = store.get_metrics()
    assert metrics['total_trades'] == 0
    assert metrics['total_slippage'] == 0.0
    
    # Record
    store.record_metrics(
        slippage_bps=5.0,
        commission=1.0,
        trade_details={'symbol': 'AAPL', 'qty': 10}
    )
    
    metrics = store.get_metrics()
    assert metrics['total_trades'] == 1
    assert metrics['total_slippage'] == 5.0
    assert metrics['total_commission'] == 1.0
    assert len(metrics['slippage_history']) == 1
    assert metrics['slippage_history'][0]['symbol'] == 'AAPL'
    
    # Persistence
    store2 = ExecutionMetricsStore(p)
    metrics2 = store2.get_metrics()
    assert metrics2['total_trades'] == 1
    assert metrics2['total_slippage'] == 5.0

def test_ibkr_connector_integration(tmp_path):
    """Test IBKRConnector integrates with store."""
    # Mock config to use tmp_path
    with patch('execution.ibkr_connector.ApexConfig') as MockConfig:
        MockConfig.DATA_DIR = tmp_path
        MockConfig.IBKR_PORT = 7497
        
        # Instantiate connector (mocks IB connection)
        with patch('execution.ibkr_connector.IB'):
            connector = IBKRConnector()
            
            assert isinstance(connector.metrics_store, ExecutionMetricsStore)
            
            # Record metrics via connector
            connector.record_execution_metrics('AAPL', 150.00, 150.15, 1.0)
            
            # Check store
            metrics = connector.metrics_store.get_metrics()
            assert metrics['total_trades'] == 1
            assert metrics['total_commission'] == 1.0
            # Slippage: (150.15 - 150.00)/150.00 * 10000 = 10 bps
            assert abs(metrics['total_slippage'] - 10.0) < 0.01

