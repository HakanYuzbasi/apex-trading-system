"""
core/request_context.py - Request/Trade Context Tracking

Provides context variables for tracking requests and trades across
async boundaries. Enables correlation of logs and metrics.

Features:
- Context variables for async-safe state
- Trade ID tracking across operations
- Correlation ID for request tracing
- Symbol context for logging
- Automatic context propagation
"""

import asyncio
import contextvars
import functools
import logging
import uuid
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Context Variables
# ============================================================================

# Unique identifiers
trade_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'trade_id', default=None
)

correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)

# Trading context
symbol_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'symbol', default=None
)

trading_cycle_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    'trading_cycle', default=None
)

order_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'order_id', default=None
)

# Session context
session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'session_id', default=None
)

user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'user_id', default=None
)


# ============================================================================
# Context Data Class
# ============================================================================

@dataclass
class TradingContext:
    """Complete trading context snapshot."""
    trade_id: Optional[str] = None
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    symbol: Optional[str] = None
    trading_cycle: Optional[int] = None
    order_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'trade_id': self.trade_id,
            'correlation_id': self.correlation_id,
            'request_id': self.request_id,
            'symbol': self.symbol,
            'trading_cycle': self.trading_cycle,
            'order_id': self.order_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() + "Z",
            **self.extra
        }

    def to_log_prefix(self) -> str:
        """Format as log prefix string."""
        parts = []
        if self.trade_id:
            parts.append(f"trade={self.trade_id[:8]}")
        if self.symbol:
            parts.append(f"symbol={self.symbol}")
        if self.trading_cycle is not None:
            parts.append(f"cycle={self.trading_cycle}")
        if self.correlation_id:
            parts.append(f"corr={self.correlation_id[:8]}")

        return f"[{' '.join(parts)}]" if parts else ""


# ============================================================================
# Context Getters
# ============================================================================

def get_trade_id() -> Optional[str]:
    """Get current trade ID."""
    return trade_id_var.get()


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id_var.get()


def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return request_id_var.get()


def get_symbol() -> Optional[str]:
    """Get current symbol."""
    return symbol_var.get()


def get_trading_cycle() -> Optional[int]:
    """Get current trading cycle."""
    return trading_cycle_var.get()


def get_order_id() -> Optional[str]:
    """Get current order ID."""
    return order_id_var.get()


def get_context() -> TradingContext:
    """Get complete context snapshot."""
    return TradingContext(
        trade_id=trade_id_var.get(),
        correlation_id=correlation_id_var.get(),
        request_id=request_id_var.get(),
        symbol=symbol_var.get(),
        trading_cycle=trading_cycle_var.get(),
        order_id=order_id_var.get(),
        session_id=session_id_var.get(),
        user_id=user_id_var.get()
    )


# ============================================================================
# Context Setters
# ============================================================================

def set_trade_id(trade_id: str) -> contextvars.Token:
    """Set trade ID and return token for reset."""
    return trade_id_var.set(trade_id)


def set_correlation_id(correlation_id: str) -> contextvars.Token:
    """Set correlation ID and return token for reset."""
    return correlation_id_var.set(correlation_id)


def set_request_id(request_id: str) -> contextvars.Token:
    """Set request ID and return token for reset."""
    return request_id_var.set(request_id)


def set_symbol(symbol: str) -> contextvars.Token:
    """Set symbol and return token for reset."""
    return symbol_var.set(symbol)


def set_trading_cycle(cycle: int) -> contextvars.Token:
    """Set trading cycle and return token for reset."""
    return trading_cycle_var.set(cycle)


def set_order_id(order_id: str) -> contextvars.Token:
    """Set order ID and return token for reset."""
    return order_id_var.set(order_id)


def set_user_id(user_id: str) -> contextvars.Token:
    """Set user ID and return token for reset."""
    return user_id_var.set(user_id)


def generate_trade_id() -> str:
    """Generate a new trade ID."""
    return f"TRD-{uuid.uuid4().hex[:12].upper()}"


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"COR-{uuid.uuid4().hex[:12].upper()}"


def generate_request_id() -> str:
    """Generate a new request ID."""
    return f"REQ-{uuid.uuid4().hex[:12].upper()}"


# ============================================================================
# Context Managers
# ============================================================================

@contextmanager
def trade_context(
    trade_id: str = None,
    symbol: str = None,
    auto_generate: bool = True
):
    """
    Context manager for trade operations.

    Args:
        trade_id: Trade ID (generated if not provided and auto_generate=True)
        symbol: Trading symbol
        auto_generate: Auto-generate trade ID if not provided

    Example:
        with trade_context(symbol="AAPL"):
            await place_order(...)
            await update_position(...)
    """
    if trade_id is None and auto_generate:
        trade_id = generate_trade_id()

    tokens = []
    if trade_id:
        tokens.append(('trade_id', trade_id_var.set(trade_id)))
    if symbol:
        tokens.append(('symbol', symbol_var.set(symbol)))

    try:
        yield get_context()
    finally:
        for var_name, token in tokens:
            if var_name == 'trade_id':
                trade_id_var.reset(token)
            elif var_name == 'symbol':
                symbol_var.reset(token)


@asynccontextmanager
async def async_trade_context(
    trade_id: str = None,
    symbol: str = None,
    auto_generate: bool = True
):
    """
    Async context manager for trade operations.

    Example:
        async with async_trade_context(symbol="AAPL"):
            await place_order(...)
    """
    with trade_context(trade_id, symbol, auto_generate) as ctx:
        yield ctx


@contextmanager
def request_context(
    correlation_id: str = None,
    request_id: str = None,
    auto_generate: bool = True
):
    """
    Context manager for request tracking.

    Args:
        correlation_id: Correlation ID for tracing
        request_id: Request ID
        auto_generate: Auto-generate IDs if not provided

    Example:
        with request_context():
            await handle_request(...)
    """
    if correlation_id is None and auto_generate:
        correlation_id = generate_correlation_id()
    if request_id is None and auto_generate:
        request_id = generate_request_id()

    tokens = []
    if correlation_id:
        tokens.append(('correlation_id', correlation_id_var.set(correlation_id)))
    if request_id:
        tokens.append(('request_id', request_id_var.set(request_id)))

    try:
        yield get_context()
    finally:
        for var_name, token in tokens:
            if var_name == 'correlation_id':
                correlation_id_var.reset(token)
            elif var_name == 'request_id':
                request_id_var.reset(token)


@contextmanager
def trading_cycle_context(cycle: int):
    """
    Context manager for trading cycle.

    Example:
        with trading_cycle_context(cycle=42):
            await process_signals()
    """
    token = trading_cycle_var.set(cycle)
    try:
        yield get_context()
    finally:
        trading_cycle_var.reset(token)


@contextmanager
def order_context(order_id: str, symbol: str = None):
    """
    Context manager for order operations.

    Example:
        with order_context(order_id="ORD-123", symbol="AAPL"):
            await execute_order(...)
    """
    tokens = [order_id_var.set(order_id)]
    if symbol:
        tokens.append(symbol_var.set(symbol))

    try:
        yield get_context()
    finally:
        order_id_var.reset(tokens[0])
        if symbol:
            symbol_var.reset(tokens[1])


# ============================================================================
# Decorators
# ============================================================================

def with_trade_context(symbol_arg: str = 'symbol'):
    """
    Decorator to automatically set trade context.

    Args:
        symbol_arg: Name of the symbol argument in the function

    Example:
        @with_trade_context()
        async def process_symbol(symbol: str):
            # trade_id and symbol automatically set
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Try to get symbol from kwargs or args
            symbol = kwargs.get(symbol_arg)
            if symbol is None and args:
                # Assume first arg is symbol if not in kwargs
                symbol = args[0] if isinstance(args[0], str) else None

            with trade_context(symbol=symbol):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            symbol = kwargs.get(symbol_arg)
            if symbol is None and args:
                symbol = args[0] if isinstance(args[0], str) else None

            with trade_context(symbol=symbol):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def with_correlation_id(func: Callable):
    """
    Decorator to automatically set correlation ID.

    Example:
        @with_correlation_id
        async def handle_request():
            # correlation_id automatically set
            pass
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        with request_context():
            return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        with request_context():
            return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# ============================================================================
# Logging Integration
# ============================================================================

class ContextFilter(logging.Filter):
    """
    Logging filter that adds context variables to log records.

    Example:
        handler.addFilter(ContextFilter())
        # Then in format: "%(trade_id)s %(symbol)s %(message)s"
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Add context variables to record
        record.trade_id = trade_id_var.get() or '-'
        record.correlation_id = correlation_id_var.get() or '-'
        record.request_id = request_id_var.get() or '-'
        record.symbol = symbol_var.get() or '-'
        record.trading_cycle = trading_cycle_var.get() or '-'
        record.order_id = order_id_var.get() or '-'

        # Add formatted context
        ctx = get_context()
        record.context = ctx.to_log_prefix()

        return True


def configure_context_logging(handler: logging.Handler):
    """
    Configure a logging handler to include context.

    Example:
        handler = logging.StreamHandler()
        configure_context_logging(handler)
    """
    handler.addFilter(ContextFilter())


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **extra
):
    """
    Log message with current context.

    Example:
        log_with_context(logger, logging.INFO, "Order placed", price=185.50)
    """
    ctx = get_context()
    prefix = ctx.to_log_prefix()

    # Merge context and extra data
    log_extra = {**ctx.to_dict(), **extra}

    logger.log(level, f"{prefix} {message}", extra={'context_data': log_extra})
