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
import scripts.health_check as hc_mod
import scripts.slippage_monitor as sm_mod
import scripts.execution_log as elog_mod
from scripts.execution_log import ExecutionLog
from scripts.data_provider import DataProvider

@pytest.fixture(autouse=True)
def setup_integration(tmp_path, monkeypatch):
    for mod in (lt_mod, dp_mod, hc_mod, sm_mod, elog_mod):
        monkeypatch.setattr(mod, 'DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr(elog_mod, 'LOG_FILE', tmp_path / 'execution_log.jsonl')
    monkeypatch.setenv('APEX_ALPACA_API_KEY', 'MOCK')
    monkeypatch.setenv('APEX_ALPACA_SECRET_KEY', 'MOCK')
    monkeypatch.setenv('POLYGON_API_KEY', 'MOCK')
    monkeypatch.setattr(lt_mod, '_halt_orders', False)
    
    mock_model = {'gbm': MagicMock(), 'scaler': MagicMock()}
    mock_model['gbm'].predict_proba.return_value = __import__("numpy").array([[0.0, 0.9]])
    mock_model['scaler'].transform.return_value = [[1.0]*8]
    monkeypatch.setattr(lt_mod, 'load_model', lambda: mock_model)
    monkeypatch.setattr(lt_mod, 'build_features', lambda *a, **k: pd.DataFrame([[1]*8]*5))
    yield tmp_path

def get_fake_bars():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=5, freq='D')
    return pd.DataFrame({'Open': [10]*5, 'High': [12]*5, 'Low': [9]*5, 'Close': [11.0]*5, 'Volume': [100]*5}, index=dates)

@patch('scripts.live_trader._is_trading_day', return_value=True)
@patch('scripts.data_provider._fetch_alpaca')
@patch('scripts.live_trader._get_alpaca_api')
def test_it1_full_day_happy_path(mock_get_api, mock_alpaca_data, mock_day, tmp_path, monkeypatch):
    api = MagicMock()
    api.get_account.return_value = MagicMock(portfolio_value='100000', last_equity='100000', trading_blocked=False, buying_power='100000')
    api.list_orders.return_value = []
    order_mock = MagicMock()
    order_mock.id = 'mock-id'
    order_mock.status = 'filled'
    order_mock.filled_avg_price = '11.0'
    order_mock.updated_at = datetime.now(timezone.utc)
    api.submit_order.return_value = order_mock
    api.get_order.return_value = order_mock
    mock_get_api.return_value = api
    mock_alpaca_data.return_value = get_fake_bars()
    monkeypatch.setattr(hc_mod, 'run_health_check', lambda: {'overall': 'PASS'})
    monkeypatch.setattr(lt_mod, '_get_regime_scalar', lambda: 1.0)
    
    lt_mod.UNIVERSE = [f'SYM{i}' for i in range(10)]
    lt_mod.run_once()
    
    records = ExecutionLog.read_all()
    assert len(records) == 10
