from unittest.mock import MagicMock

from config import ApexConfig
from core.execution_loop import ApexTradingSystem


def _make_system(monkeypatch, ibkr=None, alpaca=None):
    monkeypatch.setattr(ApexConfig, "BROKER_MODE", "both", raising=False)
    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.ibkr = ibkr
    system.alpaca = alpaca
    system._broker_mode_cache_ts = 0.0
    system._broker_mode_cache_ttl = 5.0
    system._broker_mode_cache = "both"
    system._control_command_file = None
    return system


def test_connector_selection_both_mode_falls_back_to_alpaca_when_ibkr_none(monkeypatch):
    """When ibkr=None, non-crypto falls back to Alpaca (TWS-down fallback path)."""
    alpaca = MagicMock()
    system = _make_system(monkeypatch, ibkr=None, alpaca=alpaca)

    assert system._get_connector_for("AAPL") is alpaca
    assert system._get_connector_for("FX:EUR/USD") is alpaca
    assert system._get_connector_for("CRYPTO:BTC/USDT") is alpaca


def test_connector_selection_both_mode_uses_ibkr_for_non_crypto(monkeypatch):
    """When ibkr is connected, non-crypto is routed to ibkr; crypto always to alpaca."""
    ibkr = MagicMock()
    ibkr.is_connected.return_value = True
    ibkr._persistently_down = False
    alpaca = MagicMock()
    system = _make_system(monkeypatch, ibkr=ibkr, alpaca=alpaca)

    assert system._get_connector_for("AAPL") is ibkr
    assert system._get_connector_for("FX:EUR/USD") is ibkr
    assert system._get_connector_for("CRYPTO:BTC/USDT") is alpaca
