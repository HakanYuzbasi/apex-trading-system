"""
core/timeout.py - Timeout Utilities

Provides timeout protection for async operations to prevent hanging.
Includes custom exceptions and decorator patterns for easy integration.
"""

import asyncio
import functools
import logging
from typing import TypeVar, Callable, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TimeoutError(Exception):
    """Custom timeout exception with context."""

    def __init__(self, operation: str, timeout: float, message: str = ""):
        self.operation = operation
        self.timeout = timeout
        self.message = message or f"Operation '{operation}' timed out after {timeout}s"
        super().__init__(self.message)


class OrderTimeoutError(TimeoutError):
    """Timeout during order operations."""

    def __init__(self, order_id: str, timeout: float):
        super().__init__(
            operation=f"order_{order_id}",
            timeout=timeout,
            message=f"Order {order_id} timed out after {timeout}s"
        )
        self.order_id = order_id


class MarketDataTimeoutError(TimeoutError):
    """Timeout during market data fetch."""

    def __init__(self, symbol: str, timeout: float):
        super().__init__(
            operation=f"market_data_{symbol}",
            timeout=timeout,
            message=f"Market data fetch for {symbol} timed out after {timeout}s"
        )
        self.symbol = symbol


class ConnectionTimeoutError(TimeoutError):
    """Timeout during connection attempts."""

    def __init__(self, host: str, port: int, timeout: float):
        super().__init__(
            operation=f"connect_{host}:{port}",
            timeout=timeout,
            message=f"Connection to {host}:{port} timed out after {timeout}s"
        )
        self.host = host
        self.port = port


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""
    default_timeout: float = 30.0
    order_timeout: float = 30.0
    market_data_timeout: float = 10.0
    connection_timeout: float = 30.0
    historical_data_timeout: float = 60.0


# Global timeout configuration
TIMEOUT_CONFIG = TimeoutConfig()


def with_timeout(
    timeout: Optional[float] = None,
    operation_name: Optional[str] = None,
    on_timeout: Optional[Callable[[], T]] = None
):
    """
    Decorator to add timeout protection to async functions.

    Args:
        timeout: Timeout in seconds (uses default if not specified)
        operation_name: Name for error messages (uses function name if not specified)
        on_timeout: Optional fallback function to call on timeout

    Example:
        @with_timeout(timeout=10.0, operation_name="fetch_price")
        async def get_price(symbol: str) -> float:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            op_name = operation_name or func.__name__
            effective_timeout = timeout or TIMEOUT_CONFIG.default_timeout

            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout in {op_name} after {effective_timeout}s"
                )
                if on_timeout:
                    return on_timeout()
                raise TimeoutError(op_name, effective_timeout)

        return wrapper
    return decorator


async def run_with_timeout(
    coro,
    timeout: float,
    operation_name: str = "operation",
    fallback: Optional[Any] = None
) -> Any:
    """
    Run a coroutine with timeout protection.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        operation_name: Name for error messages
        fallback: Value to return on timeout (if None, raises exception)

    Returns:
        Result of coroutine or fallback value

    Example:
        result = await run_with_timeout(
            fetch_data(symbol),
            timeout=10.0,
            operation_name="fetch_data",
            fallback={}
        )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Timeout in {operation_name} after {timeout}s")
        if fallback is not None:
            return fallback
        raise TimeoutError(operation_name, timeout)


async def run_with_timeout_and_retry(
    coro_factory: Callable[[], Any],
    timeout: float,
    max_retries: int = 3,
    operation_name: str = "operation",
    fallback: Optional[Any] = None
) -> Any:
    """
    Run a coroutine with timeout and retry logic.

    Args:
        coro_factory: Factory function that creates a new coroutine each attempt
        timeout: Timeout per attempt in seconds
        max_retries: Maximum number of retry attempts
        operation_name: Name for error messages
        fallback: Value to return after all retries fail

    Returns:
        Result of coroutine or fallback value
    """

    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(
                coro_factory(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            if attempt < max_retries:
                logger.warning(
                    f"Timeout in {operation_name} (attempt {attempt + 1}/{max_retries + 1}), retrying..."
                )
                await asyncio.sleep(0.5 * (attempt + 1))  # Brief backoff
            else:
                logger.error(
                    f"Timeout in {operation_name} after {max_retries + 1} attempts"
                )

    if fallback is not None:
        return fallback
    raise TimeoutError(operation_name, timeout)


class TimeoutContext:
    """
    Context manager for timeout protection.

    Example:
        async with TimeoutContext(10.0, "fetch_data") as ctx:
            result = await slow_operation()
            if ctx.remaining_time < 2.0:
                # Running low on time, skip optional work
                pass
    """

    def __init__(self, timeout: float, operation_name: str = "operation"):
        self.timeout = timeout
        self.operation_name = operation_name
        self._task: Optional[asyncio.Task] = None
        self._start_time: float = 0

    @property
    def remaining_time(self) -> float:
        """Get remaining time in the timeout window."""
        if self._start_time == 0:
            return self.timeout
        import time
        elapsed = time.monotonic() - self._start_time
        return max(0, self.timeout - elapsed)

    async def __aenter__(self):
        import time
        self._start_time = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed
        return False


def configure_timeouts(
    default: float = 30.0,
    order: float = 30.0,
    market_data: float = 10.0,
    connection: float = 30.0,
    historical: float = 60.0
):
    """
    Configure global timeout values.

    Args:
        default: Default timeout for unspecified operations
        order: Timeout for order operations
        market_data: Timeout for market data fetches
        connection: Timeout for connection attempts
        historical: Timeout for historical data fetches
    """
    global TIMEOUT_CONFIG
    TIMEOUT_CONFIG = TimeoutConfig(
        default_timeout=default,
        order_timeout=order,
        market_data_timeout=market_data,
        connection_timeout=connection,
        historical_data_timeout=historical
    )
    logger.info(f"Timeout configuration updated: {TIMEOUT_CONFIG}")
