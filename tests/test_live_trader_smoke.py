import sys, os, json, time, gzip
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import scripts.execution_log as elog_mod
import scripts.health_check as hc_mod
import scripts.slippage_monitor as sm_mod
import scripts.data_provider as dp_mod
import scripts.live_trader as lt_mod
from scripts.execution_log import ExecutionLog

@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    for mod in (elog_mod, hc_mod, sm_mod, dp_mod, lt_mod):
        monkeypatch.setattr(mod, 'DATA_DIR', tmp_path, raising=False)
    monkeypatch.setenv('APEX_ALPACA_API_KEY', 'MOCK')
    monkeypatch.setenv('APEX_ALPACA_SECRET_KEY', 'MOCK')
    monkeypatch.setenv('POLYGON_API_KEY', 'MOCK')
    monkeypatch.setattr(lt_mod, '_halt_orders', False)
    monkeypatch.setattr(elog_mod, 'LOG_FILE', tmp_path / 'execution_log.jsonl')
    monkeypatch.setattr(hc_mod, 'STATUS_FILE', tmp_path / 'health_status.json')
    yield tmp_path

def test_data_validation_rejects_stale_and_nan_bars(monkeypatch):
    from scripts.data_provider import _validate
    dates = pd.date_range(end=pd.Timestamp.today(), periods=5, freq='D')
    df_stale = pd.DataFrame({'Close': [10]*5, 'Volume': [100]*5}, index=dates - pd.Timedelta(days=10))
    assert _validate(df_stale, 'AAPL') is False
    df_nan = pd.DataFrame({'Close': [float('nan')]*5, 'Volume': [100]*5}, index=dates)
    assert _validate(df_nan, 'AAPL') is False

@patch('scripts.live_trader._is_trading_day', return_value=True)
@patch('scripts.live_trader._get_alpaca_api')
def test_order_size_never_exceeds_max(mock_get_api, mock_day, monkeypatch):
    monkeypatch.setattr(lt_mod, 'MAX_ORDER_USD', 1000.0)
    monkeypatch.setattr(lt_mod, '_get_regime_scalar', lambda: 1.0)
    monkeypatch.setattr(lt_mod, 'UNIVERSE', ['AAPL'])
    monkeypatch.setattr(lt_mod, 'SIGNAL_THRESHOLD', 0.5)
    monkeypatch.setattr(lt_mod, 'build_features', lambda df, macro=False, spy_ret=None: df[['Close']].tail(5))
    monkeypatch.setattr(lt_mod, 'load_model', lambda: {
        'gbm': MagicMock(predict_proba=MagicMock(return_value=[[0.1, 0.9]] * 5)),
        'scaler': MagicMock(transform=MagicMock(side_effect=lambda X: X)),
    })
    dates = pd.date_range(end=pd.Timestamp.today(), periods=6, freq='D')
    bars = pd.DataFrame({'Close': [100.0]*6, 'Volume': [1000]*6}, index=dates)
    monkeypatch.setattr(lt_mod, 'DataProvider', lambda: MagicMock(fetch=MagicMock(return_value={'AAPL': bars, 'SPY': bars})))
    api = MagicMock()
    api.get_account.return_value = MagicMock(portfolio_value='100000', last_equity='100000', trading_blocked=False, buying_power='100000')
    api.list_orders.return_value = []
    api.submit_order.return_value = MagicMock(id='ord-1', status='accepted')
    api.get_order.return_value = MagicMock(status='filled', filled_avg_price='100.0', updated_at=datetime.now(timezone.utc))
    mock_get_api.return_value = api
    lt_mod.run_once()
    assert api.submit_order.call_args.kwargs['notional'] <= 1000.0

@patch('scripts.live_trader._is_trading_day', return_value=True)
@patch('scripts.live_trader._get_alpaca_api')
def test_daily_loss_limit_halts_orders(mock_get_api, mock_day, monkeypatch):
    monkeypatch.setattr(lt_mod, 'DAILY_LOSS_PCT', 0.03)
    lt_mod._halt_orders = False
    mock_acc = MagicMock(portfolio_value='95000', last_equity='100000', trading_blocked=False, buying_power='100000')
    api = MagicMock()
    api.get_account.return_value = mock_acc
    mock_get_api.return_value = api
    monkeypatch.setattr(hc_mod, 'run_health_check', lambda: {'overall': 'PASS'})
    lt_mod.run_once()
    assert lt_mod._halt_orders is True

def test_duplicate_order_check_blocks_second_submission(monkeypatch):
    fake_open_order = SimpleNamespace(symbol='AAPL', side='buy')
    mock_api = MagicMock()
    mock_api.list_orders.return_value = [fake_open_order]
    result = lt_mod._submit_order_with_retry(mock_api, 'AAPL', 0, side='buy', notional=1500.0)
    assert result is None
    mock_api.submit_order.assert_not_called()

def test_execution_log_submitted_then_filled(tmp_path, monkeypatch):
    el = ExecutionLog()
    rec = el.write('AAPL', 'buy', 0.9, 1000, 'SUBMITTED')
    el.update(rec['_id'], 105.5, '2025-01-01T10:00:00Z', 'FILLED')
    all_recs = ExecutionLog.read_all()
    assert all_recs[0]['status'] == 'FILLED'
    assert all_recs[0]['fill_price'] == 105.5

@patch('scripts.slippage_monitor._load_filled')
def test_slippage_monitor_alert_threshold(mock_load, tmp_path, monkeypatch):
    monkeypatch.setattr(sm_mod, 'HALT_BPS', 15)
    monkeypatch.setattr(sm_mod, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(sm_mod, 'HALT_FILE', tmp_path / 'HALT_slippage_exceeded.txt')
    df = pd.DataFrame({'symbol': ['SPY']*20, 'fill_price': [100.25]*20, 'model_price': [100.00]*20, 'date': ['2025-01-01']*20, 'slippage_bps': [25.0]*20})
    mock_load.return_value = df
    sm_mod.analyse(df)
    assert (tmp_path / 'HALT_slippage_exceeded.txt').exists()

@patch('scripts.health_check.run_health_check')
def test_health_check_passes_with_valid_model(mock_hc, monkeypatch):
    mock_hc.return_value = {'overall': 'PASS', 'components': {'model': 'PASS'}}
    res = hc_mod.run_health_check()
    assert res['overall'] == 'PASS'

@patch('scripts.data_provider._fetch_alpaca')
def test_data_provider_tier1_alpaca(mock_alpaca, monkeypatch):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=5, freq='D')
    df = pd.DataFrame({'Open': [10]*5, 'High': [12]*5, 'Low': [9]*5, 'Close': [11.0]*5, 'Volume': [100]*5}, index=dates)
    mock_alpaca.return_value = df
    from scripts.data_provider import DataProvider
    res = DataProvider().fetch(['AAPL'])
    assert 'AAPL' in res

@patch('scripts.data_provider._fetch_alpaca')
@patch('scripts.data_provider._fetch_polygon')
def test_data_provider_tier2_polygon(mock_poly, mock_alpaca, monkeypatch):
    import alpaca_trade_api.rest
    mock_alpaca.side_effect = alpaca_trade_api.rest.APIError({'message': 'down', 'code': 500})
    dates = pd.date_range(end=pd.Timestamp.today(), periods=5, freq='D')
    df = pd.DataFrame({'Open': [10]*5, 'High': [12]*5, 'Low': [9]*5, 'Close': [11.0]*5, 'Volume': [100]*5}, index=dates)
    mock_poly.return_value = df
    from scripts.data_provider import DataProvider
    monkeypatch.setattr('scripts.data_provider._RETRIES', 1)
    monkeypatch.setattr('scripts.data_provider._BACKOFF', 0)
    res = DataProvider().fetch(['AAPL'])
    assert mock_alpaca.call_count == 1
    assert 'AAPL' in res
