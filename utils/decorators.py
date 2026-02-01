"""
utils/decorators.py - Function Decorators

Reusable decorators for common patterns:
- Execution timing
- Result caching with TTL
- Rate limiting
- Retry on exception
- Execution logging
- Deprecation warnings
"""

import asyncio
import functools
import logging
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Callable, Dict, Optional, Type, Union

logger = logging.getLogger(__name__)


def timeit(func: Callable = None, *, log_level: int = logging.DEBUG, threshold_ms: float = None):
    """
    Decorator to measure and log function execution time.

    Args:
        log_level: Logging level for timing messages
        threshold_ms: Only log if execution exceeds this threshold

    Example:
        @timeit
        def slow_function():
            time.sleep(1)

        @timeit(threshold_ms=100)
        def maybe_slow():
            pass
    """
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if threshold_ms is None or elapsed_ms >= threshold_ms:
                    logger.log(log_level, f"{fn.__name__} took {elapsed_ms:.2f}ms")

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if threshold_ms is None or elapsed_ms >= threshold_ms:
                    logger.log(log_level, f"{fn.__name__} took {elapsed_ms:.2f}ms")

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


class CacheEntry:
    """Cache entry with expiration."""
    def __init__(self, value: Any, expires_at: datetime):
        self.value = value
        self.expires_at = expires_at

    @property
    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at


def cache_result(ttl_seconds: float = 60.0, maxsize: int = 128):
    """
    Decorator to cache function results with TTL.

    Args:
        ttl_seconds: Time to live in seconds
        maxsize: Maximum cache size

    Example:
        @cache_result(ttl_seconds=300)
        def fetch_price(symbol: str) -> float:
            return api.get_price(symbol)
    """
    def decorator(func: Callable):
        cache: Dict[str, CacheEntry] = {}
        lock = Lock()

        def make_key(*args, **kwargs) -> str:
            return f"{args}-{sorted(kwargs.items())}"

        def cleanup():
            """Remove expired entries."""
            now = datetime.now()
            expired = [k for k, v in cache.items() if v.is_expired]
            for k in expired:
                del cache[k]

            # Enforce maxsize
            if len(cache) > maxsize:
                sorted_entries = sorted(cache.items(), key=lambda x: x[1].expires_at)
                for k, _ in sorted_entries[:len(cache) - maxsize]:
                    del cache[k]

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = make_key(*args, **kwargs)

            with lock:
                if key in cache and not cache[key].is_expired:
                    return cache[key].value

            result = await func(*args, **kwargs)

            with lock:
                cache[key] = CacheEntry(result, datetime.now() + timedelta(seconds=ttl_seconds))
                cleanup()

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = make_key(*args, **kwargs)

            with lock:
                if key in cache and not cache[key].is_expired:
                    return cache[key].value

            result = func(*args, **kwargs)

            with lock:
                cache[key] = CacheEntry(result, datetime.now() + timedelta(seconds=ttl_seconds))
                cleanup()

            return result

        # Add cache management methods
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'size': len(cache), 'maxsize': maxsize}

        return wrapper

    return decorator


def rate_limit(calls_per_second: float = 1.0, burst: int = 1):
    """
    Decorator to rate limit function calls.

    Args:
        calls_per_second: Maximum calls per second
        burst: Maximum burst size

    Example:
        @rate_limit(calls_per_second=5)
        async def call_api():
            pass
    """
    min_interval = 1.0 / calls_per_second
    tokens = burst
    last_call = time.monotonic()
    lock = Lock()

    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal tokens, last_call

            async def wait_for_token():
                nonlocal tokens, last_call
                while True:
                    with lock:
                        now = time.monotonic()
                        # Refill tokens
                        elapsed = now - last_call
                        tokens = min(burst, tokens + elapsed * calls_per_second)
                        last_call = now

                        if tokens >= 1:
                            tokens -= 1
                            return

                    await asyncio.sleep(min_interval)

            await wait_for_token()
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal tokens, last_call

            with lock:
                now = time.monotonic()
                elapsed = now - last_call
                tokens_available = min(burst, tokens + elapsed * calls_per_second)

                if tokens_available < 1:
                    wait_time = (1 - tokens_available) / calls_per_second
                    time.sleep(wait_time)
                    tokens_available = 1

                tokens = tokens_available - 1
                last_call = time.monotonic()

            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def retry_on_exception(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    on_retry: Callable = None
):
    """
    Decorator to retry function on exception.

    Args:
        max_retries: Maximum retry attempts
        exceptions: Tuple of exceptions to catch
        delay: Initial delay between retries
        backoff: Backoff multiplier
        max_delay: Maximum delay
        on_retry: Callback on retry (receives exception, attempt number)

    Example:
        @retry_on_exception(max_retries=3, exceptions=(ConnectionError,))
        async def connect():
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)

                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )

                        await asyncio.sleep(current_delay)
                        current_delay = min(current_delay * backoff, max_delay)
                    else:
                        raise

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        if on_retry:
                            on_retry(e, attempt + 1)

                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )

                        time.sleep(current_delay)
                        current_delay = min(current_delay * backoff, max_delay)
                    else:
                        raise

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def log_execution(
    log_level: int = logging.INFO,
    log_args: bool = True,
    log_result: bool = False,
    max_arg_length: int = 100
):
    """
    Decorator to log function execution.

    Args:
        log_level: Logging level
        log_args: Whether to log arguments
        log_result: Whether to log return value
        max_arg_length: Max length for argument string

    Example:
        @log_execution(log_args=True)
        def process_order(order):
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = func.__name__

            if log_args:
                args_str = str(args)[:max_arg_length]
                kwargs_str = str(kwargs)[:max_arg_length]
                logger.log(log_level, f"Calling {func_name}({args_str}, {kwargs_str})")
            else:
                logger.log(log_level, f"Calling {func_name}")

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)

                if log_result:
                    result_str = str(result)[:max_arg_length]
                    logger.log(log_level, f"{func_name} returned: {result_str}")

                return result
            except Exception as e:
                logger.error(f"{func_name} raised {type(e).__name__}: {e}")
                raise
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                logger.log(log_level, f"{func_name} completed in {elapsed:.2f}ms")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = func.__name__

            if log_args:
                args_str = str(args)[:max_arg_length]
                kwargs_str = str(kwargs)[:max_arg_length]
                logger.log(log_level, f"Calling {func_name}({args_str}, {kwargs_str})")
            else:
                logger.log(log_level, f"Calling {func_name}")

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)

                if log_result:
                    result_str = str(result)[:max_arg_length]
                    logger.log(log_level, f"{func_name} returned: {result_str}")

                return result
            except Exception as e:
                logger.error(f"{func_name} raised {type(e).__name__}: {e}")
                raise
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                logger.log(log_level, f"{func_name} completed in {elapsed:.2f}ms")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def deprecated(reason: str = "", version: str = None, replacement: str = None):
    """
    Decorator to mark functions as deprecated.

    Args:
        reason: Reason for deprecation
        version: Version when deprecated
        replacement: Suggested replacement

    Example:
        @deprecated(reason="Use new_function instead", version="2.0")
        def old_function():
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__} is deprecated"

            if version:
                msg += f" as of version {version}"

            if reason:
                msg += f": {reason}"

            if replacement:
                msg += f". Use {replacement} instead."

            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def singleton(cls):
    """
    Decorator to make a class a singleton.

    Example:
        @singleton
        class Config:
            pass
    """
    instances = {}

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper
