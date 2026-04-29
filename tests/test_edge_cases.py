import sys, os, json, time, gzip
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import scripts.live_trader as lt_mod
import scripts.data_provider as dp_mod
import scripts.execution_log as elog_mod
from scripts.execution_log import ExecutionLog

@pytest.fixture(autouse=True)
def setup_edge(tmp_path, monkeypatch):
    for mod in (lt_mod, dp_mod, elog_mod):
        monkeypatch.setattr(mod, 'DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr(elog_mod, 'LOG_FILE', tmp_path / 'execution_log.jsonl')
    monkeypatch.setenv('APEX_ALPACA_API_KEY', 'MOCK')
    monkeypatch.setenv('APEX_ALPACA_SECRET_KEY', 'MOCK')
    monkeypatch.setattr(lt_mod, '_halt_orders', False)
    yield tmp_path

@patch('scripts.live_trader._is_trading_day', return_value=True)
@patch('scripts.data_provider.DataProvider.fetch')
@patch('scripts.live_trader._get_alpaca_api')
def test_ec5_buying_power_skipped(mock_get_api, mock_fetch, mock_day, tmp_path, monkeypatch):
    api = MagicMock()
    api.get_account.return_value = MagicMock(portfolio_value='100000', last_equity='100000', trading_blocked=False, buying_power='500')
    api.list_orders.return_value = []
    mock_get_api.return_value = api
    
    monkeypatch.setattr(lt_mod, 'MAX_ORDER_USD', 2000.0)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=5, freq='D')
    df = pd.DataFrame({'Close': [11.0]*5, 'Volume': [100]*5}, index=dates)
    mock_fetch.return_value = {'AAPL': df, 'SPY': df}
    
    # Mock model
    mock_model = {'gbm': MagicMock(), 'scaler': MagicMock()}
    mock_model['gbm'].predict_proba.return_value = __import__("numpy").array([[0.0, 0.9]])
    mock_model['scaler'].transform.return_value = [[1.0]*8]
    monkeypatch.setattr(lt_mod, 'load_model', lambda: mock_model)
    monkeypatch.setattr(lt_mod, 'build_features', lambda *a, **k: pd.DataFrame([[1]*8]*5))
    monkeypatch.setattr(lt_mod, '_get_regime_scalar', lambda: 1.0)
    
    # Mock health_check.run_health_check
    import scripts.health_check as hc_mod
    monkeypatch.setattr(hc_mod, 'run_health_check', lambda: {'overall': 'PASS'})

    lt_mod.UNIVERSE = ['AAPL']
    lt_mod.run_once()
    
    api.submit_order.assert_not_called()
    records = ExecutionLog.read_all()
    assert len(records) == 1
    assert records[0]['status'] == 'FAILED'
