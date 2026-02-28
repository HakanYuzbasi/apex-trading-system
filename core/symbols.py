"""
core/symbols.py

Symbol parsing and normalization utilities for multi-asset trading.
Supports equities, forex, and crypto symbols.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple
import re


class AssetClass(Enum):
    """Supported asset classes."""
    EQUITY = "EQUITY"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"


# Common fiat currencies for FX/crypto quotes
KNOWN_FIAT_CURRENCIES = {
    "USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD",
    "SEK", "NOK", "MXN", "ZAR", "HKD", "SGD", "CNH",
}

# Common crypto quote currencies
KNOWN_CRYPTO_QUOTES = {"USD", "USDT", "USDC"}

# Common crypto base assets (used for auto-detection without prefix)
KNOWN_CRYPTO_ASSETS = {
    "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LTC", "BCH",
    "DOGE", "AVAX", "LINK", "MATIC", "XLM", "XMR", "ETC",
    "AAVE", "UNI", "BAT", "RENDER",
}

_PAIR_SEPARATORS = ("/", ".")  # "." supports IBKR format: EUR.USD, USD.CHF
_ALNUM_RE = re.compile(r"^[A-Z0-9]+$")


@dataclass(frozen=True)
class ParsedSymbol:
    raw: str
    asset_class: AssetClass
    base: str
    quote: str
    normalized: str


def _split_pair(pair: str, *, allow_variable_base: bool, require_separator: bool) -> Tuple[str, str]:
    for sep in _PAIR_SEPARATORS:
        if sep in pair:
            parts = [p for p in pair.split(sep) if p]
            if len(parts) != 2:
                raise ValueError(f"Invalid pair format: {pair}")
            return parts[0], parts[1]

    if require_separator:
        raise ValueError(f"Invalid pair format (expected BASE/QUOTE): {pair}")

    # No separator (legacy fallback)
    if allow_variable_base:
        if len(pair) < 6:
            raise ValueError(f"Invalid pair format: {pair}")
        return pair[:-3], pair[-3:]

    if len(pair) != 6:
        raise ValueError(f"Invalid pair format: {pair}")
    return pair[:3], pair[3:]


def _is_forex_pair(base: str, quote: str) -> bool:
    return base in KNOWN_FIAT_CURRENCIES and quote in KNOWN_FIAT_CURRENCIES


def _is_crypto_pair(base: str, quote: str) -> bool:
    return quote in KNOWN_CRYPTO_QUOTES and base in KNOWN_CRYPTO_ASSETS


def parse_symbol(raw: str) -> ParsedSymbol:
    """
    Parse a raw symbol into a normalized, typed representation.

    Supported forms:
    - Equities: "AAPL"
    - Forex: "EUR/USD" or "FX:EUR/USD"
    - Crypto: "BTC/USDT" or "CRYPTO:BTC/USDT"

    Returns:
        ParsedSymbol with asset class and normalized form.
    """
    if not isinstance(raw, str):
        raise ValueError("Symbol must be a string")

    s = raw.strip().upper()
    if not s:
        raise ValueError("Symbol cannot be empty")

    # Explicit prefixes
    if s.startswith("FX:") or s.startswith("FOREX:"):
        pair = s.split(":", 1)[1]
        base, quote = _split_pair(pair, allow_variable_base=False, require_separator=True)
        if not _is_forex_pair(base, quote):
            raise ValueError(f"Invalid forex pair: {pair}")
        normalized = f"FX:{base}/{quote}"
        return ParsedSymbol(raw=raw, asset_class=AssetClass.FOREX, base=base, quote=quote, normalized=normalized)

    if s.startswith("CRYPTO:"):
        pair = s.split(":", 1)[1]
        base, quote = _split_pair(pair, allow_variable_base=True, require_separator=True)
        if quote not in KNOWN_CRYPTO_QUOTES:
            raise ValueError(f"Invalid crypto quote currency: {quote}")
        if not _ALNUM_RE.match(base):
            raise ValueError(f"Invalid crypto base asset: {base}")
        normalized = f"CRYPTO:{base}/{quote}"
        return ParsedSymbol(raw=raw, asset_class=AssetClass.CRYPTO, base=base, quote=quote, normalized=normalized)

    # Auto-detect pairs with separators (e.g., EUR/USD, BTC/USDT)
    if any(sep in s for sep in _PAIR_SEPARATORS):
        try:
            base, quote = _split_pair(s, allow_variable_base=True, require_separator=True)
            if _is_forex_pair(base, quote):
                normalized = f"FX:{base}/{quote}"
                return ParsedSymbol(raw=raw, asset_class=AssetClass.FOREX, base=base, quote=quote, normalized=normalized)
            if _is_crypto_pair(base, quote):
                normalized = f"CRYPTO:{base}/{quote}"
                return ParsedSymbol(raw=raw, asset_class=AssetClass.CRYPTO, base=base, quote=quote, normalized=normalized)
        except ValueError:
            raise

        raise ValueError(f"Invalid pair format: {s}")

    # Default to equity
    return ParsedSymbol(raw=raw, asset_class=AssetClass.EQUITY, base=s, quote="USD", normalized=s)


def normalize_symbol(raw: str) -> str:
    """Return the canonical symbol representation."""
    return parse_symbol(raw).normalized


def to_yfinance_ticker(raw_or_parsed) -> str:
    """Convert a symbol to its yfinance ticker format."""
    parsed = raw_or_parsed if isinstance(raw_or_parsed, ParsedSymbol) else parse_symbol(raw_or_parsed)

    if parsed.asset_class == AssetClass.FOREX:
        return f"{parsed.base}{parsed.quote}=X"
    if parsed.asset_class == AssetClass.CRYPTO:
        return f"{parsed.base}-{parsed.quote}"
    return parsed.base


def is_supported_symbol(raw: str) -> bool:
    try:
        parse_symbol(raw)
        return True
    except ValueError:
        return False


def normalize_symbols(symbols: Iterable[str]) -> Tuple[str, ...]:
    return tuple(normalize_symbol(s) for s in symbols)


def is_market_open(raw_or_parsed, timestamp, assume_daily: bool = False) -> bool:
    """
    Market hours helper by asset class.
    DEPRECATED: Use core.market_hours.is_market_open instead.
    
    For daily bars (midnight timestamps), pass assume_daily=True to avoid
    rejecting valid trading days due to time-of-day checks.
    """
    # Lazy import to avoid circular dependency (market_hours imports AssetClass from symbols)
    from core.market_hours import is_market_open as _is_market_open
    return _is_market_open(raw_or_parsed, timestamp, assume_daily)


# PSEUDO-TESTS
# parse_symbol("FX:EUR/USD") -> FX:EUR/USD
# parse_symbol("CRYPTO:BTC/USDT") -> CRYPTO:BTC/USDT
# parse_symbol("BTC/USDT") -> CRYPTO:BTC/USDT


# --- FIX: MARKET HOURS OVERRIDE ---
import pytz
from datetime import datetime

def custom_is_market_open(symbol: str, timestamp=None, **kwargs) -> bool:
    # Handle assume_daily from kwargs
    assume_daily = kwargs.get('assume_daily', False)
    
    if timestamp is None:
        timestamp = datetime.utcnow()
    try:
        parsed = parse_symbol(symbol)
        
        # Crypto is 24/7
        if parsed.asset_class.name == 'CRYPTO' or parsed.asset_class.value == 'CRYPTO': 
            return True
            
        # Forex is 24/5 (Mon-Fri)
        if parsed.asset_class.name == 'FOREX' or parsed.asset_class.value == 'FOREX': 
            return timestamp.weekday() < 5
            
        # Equities / Options (Mon-Fri)
        if timestamp.weekday() >= 5: 
            return False
            
        # If assume_daily is True, we only care about the day (Mon-Fri)
        if assume_daily:
            return True

        # Calculate precise New York time
        eastern = pytz.timezone('America/New_York')
        now_est = timestamp.astimezone(eastern) if timestamp.tzinfo else pytz.utc.localize(timestamp).astimezone(eastern)
        est_hour = now_est.hour + now_est.minute / 60.0
        
        # 9.5 = 9:30 AM EST | 16.0 = 4:00 PM EST
        return 9.5 <= est_hour <= 16.0
    except Exception:
        # If it fails to parse, assume it's open to be safe
        return True

# Override the buggy library globally
is_market_open = custom_is_market_open
# ----------------------------------

