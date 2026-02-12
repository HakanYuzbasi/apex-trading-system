import pytest
from datetime import datetime
import pytz
from core.market_hours import is_market_open
from core.symbols import is_market_open as legacy_is_market_open

def test_market_hours_equity():
    """Test equity market hours (NYSE/NASDAQ)."""
    # Monday 10:00 AM EST - OPEN
    dt_open = datetime(2023, 10, 23, 10, 0, 0, tzinfo=pytz.timezone("America/New_York"))
    # 2023-10-23 is a Monday
    
    assert is_market_open("AAPL", dt_open) == True
    assert legacy_is_market_open("AAPL", dt_open) == True

    # Monday 9:00 AM EST - CLOSED
    dt_closed = datetime(2023, 10, 23, 9, 0, 0, tzinfo=pytz.timezone("America/New_York"))
    assert is_market_open("AAPL", dt_closed) == False
    assert legacy_is_market_open("AAPL", dt_closed) == False

    # Saturday - CLOSED
    dt_sat = datetime(2023, 10, 21, 12, 0, 0, tzinfo=pytz.timezone("America/New_York"))
    assert is_market_open("AAPL", dt_sat) == False

def test_market_hours_crypto():
    """Test crypto market hours (24/7)."""
    # Sunday midnight
    dt = datetime(2023, 10, 22, 0, 0, 0, tzinfo=pytz.timezone("America/New_York"))
    
    assert is_market_open("BTC/USD", dt) == True
    assert is_market_open("CRYPTO:BTC/USDT", dt) == True

def test_market_hours_forex():
    """Test forex market hours."""
    # Tuesday 10 AM - OPEN
    dt_open = datetime(2023, 10, 24, 10, 0, 0, tzinfo=pytz.timezone("America/New_York"))
    assert is_market_open("FX:EUR/USD", dt_open) == True
    
    # Saturday noon - CLOSED
    dt_closed = datetime(2023, 10, 21, 12, 0, 0, tzinfo=pytz.timezone("America/New_York"))
    assert is_market_open("FX:EUR/USD", dt_closed) == False
    
    # Sunday 4 PM EST - CLOSED
    dt_sun_closed = datetime(2023, 10, 22, 16, 0, 0, tzinfo=pytz.timezone("America/New_York"))
    assert is_market_open("FX:EUR/USD", dt_sun_closed) == False
    
    # Sunday 6 PM EST - OPEN
    dt_sun_open = datetime(2023, 10, 22, 18, 0, 0, tzinfo=pytz.timezone("America/New_York"))
    assert is_market_open("FX:EUR/USD", dt_sun_open) == True
