"""
utils/formatters.py - Formatting Utilities

Functions for formatting prices, quantities, percentages, durations,
and other values for display.
"""

from datetime import datetime


def format_price(price: float, currency: str = "USD", decimals: int = 2) -> str:
    """
    Format a price value.

    Args:
        price: Price value
        currency: Currency code (USD, EUR, etc.)
        decimals: Decimal places

    Returns:
        Formatted price string

    Example:
        format_price(1234.567) -> "$1,234.57"
        format_price(1234.567, "EUR") -> "€1,234.57"
    """
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$",
    }
    symbol = symbols.get(currency, currency + " ")

    return f"{symbol}{price:,.{decimals}f}"


def format_quantity(quantity: int, include_sign: bool = False) -> str:
    """
    Format a quantity value.

    Args:
        quantity: Quantity value
        include_sign: Whether to include + sign for positive

    Returns:
        Formatted quantity string

    Example:
        format_quantity(1234567) -> "1,234,567"
        format_quantity(100, include_sign=True) -> "+100"
    """
    if include_sign and quantity > 0:
        return f"+{quantity:,}"
    return f"{quantity:,}"


def format_percentage(
    value: float,
    decimals: int = 2,
    include_sign: bool = True,
    multiply: bool = True
) -> str:
    """
    Format a percentage value.

    Args:
        value: Percentage value (0.05 = 5% if multiply=True)
        decimals: Decimal places
        include_sign: Whether to include + sign for positive
        multiply: Whether to multiply by 100

    Returns:
        Formatted percentage string

    Example:
        format_percentage(0.0523) -> "+5.23%"
        format_percentage(-0.02) -> "-2.00%"
    """
    if multiply:
        value = value * 100

    if include_sign and value > 0:
        return f"+{value:.{decimals}f}%"
    return f"{value:.{decimals}f}%"


def format_pnl(
    value: float,
    currency: str = "USD",
    include_sign: bool = True,
    color_codes: bool = False
) -> str:
    """
    Format a P&L value.

    Args:
        value: P&L value
        currency: Currency code
        include_sign: Whether to include + sign
        color_codes: Whether to include ANSI color codes

    Returns:
        Formatted P&L string

    Example:
        format_pnl(1234.56) -> "+$1,234.56"
        format_pnl(-567.89) -> "-$567.89"
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency + " ")

    if value >= 0:
        formatted = f"+{symbol}{value:,.2f}" if include_sign else f"{symbol}{value:,.2f}"
        if color_codes:
            formatted = f"\033[32m{formatted}\033[0m"  # Green
    else:
        formatted = f"-{symbol}{abs(value):,.2f}"
        if color_codes:
            formatted = f"\033[31m{formatted}\033[0m"  # Red

    return formatted


def format_duration(seconds: float, short: bool = False) -> str:
    """
    Format a duration in seconds to human readable format.

    Args:
        seconds: Duration in seconds
        short: Use short format (1h 30m vs 1 hour 30 minutes)

    Returns:
        Formatted duration string

    Example:
        format_duration(3661) -> "1 hour 1 minute 1 second"
        format_duration(3661, short=True) -> "1h 1m 1s"
    """
    if seconds < 0:
        return "N/A"

    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)

    parts = []

    if short:
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or (not parts and ms == 0):
            parts.append(f"{secs}s")
        if ms > 0 and not parts:
            parts.append(f"{ms}ms")
    else:
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 or not parts:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")

    return " ".join(parts) if parts else "0s" if short else "0 seconds"


def format_timestamp(
    dt: datetime,
    format_str: str = None,
    relative: bool = False
) -> str:
    """
    Format a datetime timestamp.

    Args:
        dt: Datetime object
        format_str: Custom format string
        relative: Show relative time (e.g., "5 minutes ago")

    Returns:
        Formatted timestamp string

    Example:
        format_timestamp(datetime.now()) -> "2024-01-15 10:30:45"
        format_timestamp(datetime.now(), relative=True) -> "just now"
    """
    if relative:
        now = datetime.now()
        diff = now - dt

        if diff.total_seconds() < 60:
            return "just now"
        elif diff.total_seconds() < 3600:
            mins = int(diff.total_seconds() / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.days < 7:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        else:
            return dt.strftime("%Y-%m-%d")

    if format_str:
        return dt.strftime(format_str)

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_currency(value: float, currency: str = "USD", compact: bool = False) -> str:
    """
    Format a currency value with optional compact notation.

    Args:
        value: Currency value
        currency: Currency code
        compact: Use compact notation for large numbers

    Returns:
        Formatted currency string

    Example:
        format_currency(1234567.89) -> "$1,234,567.89"
        format_currency(1234567.89, compact=True) -> "$1.23M"
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency + " ")

    if compact:
        abs_value = abs(value)
        sign = "-" if value < 0 else ""

        if abs_value >= 1e12:
            return f"{sign}{symbol}{abs_value / 1e12:.2f}T"
        elif abs_value >= 1e9:
            return f"{sign}{symbol}{abs_value / 1e9:.2f}B"
        elif abs_value >= 1e6:
            return f"{sign}{symbol}{abs_value / 1e6:.2f}M"
        elif abs_value >= 1e3:
            return f"{sign}{symbol}{abs_value / 1e3:.2f}K"

    return f"{symbol}{value:,.2f}"


def format_number(
    value: float,
    decimals: int = 2,
    compact: bool = False,
    include_sign: bool = False
) -> str:
    """
    Format a number.

    Args:
        value: Number value
        decimals: Decimal places
        compact: Use compact notation
        include_sign: Include + sign for positive

    Returns:
        Formatted number string

    Example:
        format_number(1234567.89) -> "1,234,567.89"
        format_number(1234567.89, compact=True) -> "1.23M"
    """
    if compact:
        abs_value = abs(value)
        sign = "-" if value < 0 else ("+" if include_sign else "")

        if abs_value >= 1e12:
            return f"{sign}{abs_value / 1e12:.{decimals}f}T"
        elif abs_value >= 1e9:
            return f"{sign}{abs_value / 1e9:.{decimals}f}B"
        elif abs_value >= 1e6:
            return f"{sign}{abs_value / 1e6:.{decimals}f}M"
        elif abs_value >= 1e3:
            return f"{sign}{abs_value / 1e3:.{decimals}f}K"

    if include_sign and value > 0:
        return f"+{value:,.{decimals}f}"
    return f"{value:,.{decimals}f}"


def format_ratio(numerator: float, denominator: float, decimals: int = 2) -> str:
    """
    Format a ratio.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        decimals: Decimal places

    Returns:
        Formatted ratio string

    Example:
        format_ratio(3, 2) -> "1.50"
        format_ratio(0, 1) -> "0.00"
    """
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator:.{decimals}f}"


def format_order_side(side: str, with_symbol: bool = False) -> str:
    """
    Format an order side.

    Args:
        side: Order side (BUY/SELL)
        with_symbol: Include arrow symbol

    Returns:
        Formatted side string

    Example:
        format_order_side("BUY") -> "BUY"
        format_order_side("BUY", with_symbol=True) -> "↑ BUY"
    """
    side = side.upper()
    if with_symbol:
        symbol = "↑" if side == "BUY" else "↓"
        return f"{symbol} {side}"
    return side


def format_market_status(is_open: bool, market: str = "US") -> str:
    """
    Format market status.

    Args:
        is_open: Whether market is open
        market: Market identifier

    Returns:
        Formatted status string

    Example:
        format_market_status(True) -> "OPEN"
        format_market_status(False) -> "CLOSED"
    """
    return "OPEN" if is_open else "CLOSED"
