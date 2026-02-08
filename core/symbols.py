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
    "AAVE", "UNI",
}

_PAIR_SEPARATORS = ("/",)
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


def _to_eastern(timestamp):
    try:
        import pytz
        eastern = pytz.timezone("America/New_York")
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=pytz.UTC).astimezone(eastern)
        return timestamp.astimezone(eastern)
    except Exception:
        return timestamp


def _parse_time(value: str):
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {value}")
    hour = int(parts[0])
    minute = int(parts[1])
    return hour, minute


def _in_custom_session(dt, session: dict) -> bool:
    timezone = session.get("timezone", "America/New_York")
    try:
        import pytz
        tz = pytz.timezone(timezone)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC).astimezone(tz)
        else:
            dt = dt.astimezone(tz)
    except Exception:
        pass

    weekdays = session.get("weekdays")
    if weekdays is not None and dt.weekday() not in weekdays:
        return False

    open_time = session.get("open")
    close_time = session.get("close")
    if not open_time or not close_time:
        return True

    open_h, open_m = _parse_time(open_time)
    close_h, close_m = _parse_time(close_time)

    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m
    now_minutes = dt.hour * 60 + dt.minute

    if open_minutes == close_minutes:
        return True
    if open_minutes < close_minutes:
        return open_minutes <= now_minutes < close_minutes
    return now_minutes >= open_minutes or now_minutes < close_minutes


def is_market_open(raw_or_parsed, timestamp, assume_daily: bool = False) -> bool:
    """
    Market hours helper by asset class.

    For daily bars (midnight timestamps), pass assume_daily=True to avoid
    rejecting valid trading days due to time-of-day checks.
    """
    try:
        from config import ApexConfig
    except Exception:
        ApexConfig = None

    parsed = raw_or_parsed if isinstance(raw_or_parsed, ParsedSymbol) else parse_symbol(raw_or_parsed)

    if ApexConfig is not None and getattr(ApexConfig, "MARKET_ALWAYS_OPEN", False):
        return True

    if parsed.asset_class == AssetClass.CRYPTO:
        if ApexConfig is not None and getattr(ApexConfig, "CRYPTO_ALWAYS_OPEN", False):
            return True
        if ApexConfig is not None:
            sessions = getattr(ApexConfig, "CUSTOM_MARKET_SESSIONS", {}).get("CRYPTO")
            if sessions:
                return _in_custom_session(timestamp, sessions)
        return True

    dt = _to_eastern(timestamp)
    weekday = dt.weekday()
    is_midnight = dt.hour == 0 and dt.minute == 0 and dt.second == 0

    if parsed.asset_class == AssetClass.FOREX:
        if ApexConfig is not None and getattr(ApexConfig, "FX_ALWAYS_OPEN", False):
            return True
        if ApexConfig is not None:
            sessions = getattr(ApexConfig, "CUSTOM_MARKET_SESSIONS", {}).get("FOREX")
            if sessions:
                return _in_custom_session(timestamp, sessions)
        if assume_daily or is_midnight:
            return weekday < 5
        if weekday == 5:  # Saturday
            return False
        if weekday == 6:  # Sunday
            return dt.hour >= 17
        if weekday == 4:  # Friday
            return dt.hour < 17
        return True

    # Equities
    if ApexConfig is not None:
        sessions = getattr(ApexConfig, "CUSTOM_MARKET_SESSIONS", {}).get("EQUITY")
        if sessions:
            return _in_custom_session(timestamp, sessions)
    if assume_daily or is_midnight:
        return weekday < 5
    if weekday >= 5:
        return False
    market_open = (dt.hour > 9) or (dt.hour == 9 and dt.minute >= 30)
    market_close = dt.hour < 16
    return market_open and market_close


# PSEUDO-TESTS
# parse_symbol("FX:EUR/USD") -> FX:EUR/USD
# parse_symbol("CRYPTO:BTC/USDT") -> CRYPTO:BTC/USDT
# parse_symbol("BTC/USDT") -> CRYPTO:BTC/USDT
