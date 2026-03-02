from config import ApexConfig
from core.execution_loop import ApexTradingSystem


def test_connector_selection_both_mode_requires_ibkr_for_non_crypto(monkeypatch):
    monkeypatch.setattr(ApexConfig, "BROKER_MODE", "both", raising=False)

    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.ibkr = None
    system.alpaca = object()

    assert system._get_connector_for("AAPL") is None
    assert system._get_connector_for("FX:EUR/USD") is None
    assert system._get_connector_for("CRYPTO:BTC/USDT") is system.alpaca


def test_connector_selection_both_mode_uses_ibkr_for_non_crypto(monkeypatch):
    monkeypatch.setattr(ApexConfig, "BROKER_MODE", "both", raising=False)

    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.ibkr = object()
    system.alpaca = object()

    assert system._get_connector_for("AAPL") is system.ibkr
    assert system._get_connector_for("FX:EUR/USD") is system.ibkr
    assert system._get_connector_for("CRYPTO:BTC/USDT") is system.alpaca
