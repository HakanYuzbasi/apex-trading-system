from datetime import datetime

from core.symbols import parse_symbol, is_market_open


def test_parse_crypto_slash_pair():
    parsed = parse_symbol("BTC/USDT")
    assert parsed.asset_class.name == "CRYPTO"
    assert parsed.normalized == "CRYPTO:BTC/USDT"


def test_parse_fx_slash_pair():
    parsed = parse_symbol("EUR/USD")
    assert parsed.asset_class.name == "FOREX"
    assert parsed.normalized == "FX:EUR/USD"


def test_market_hours_weekend():
    # Saturday UTC
    ts = datetime(2026, 2, 7, 12, 0, 0)
    assert is_market_open("AAPL", ts) is False
    assert is_market_open("EUR/USD", ts) is False
    assert is_market_open("BTC/USDT", ts) is True
