"""
core/bulkhead.py - Bulkhead Pattern for Failure Isolation

Implements the bulkhead pattern to isolate failures between different
trading operations, preventing cascading failures.

Features:
- Concurrent execution limits per bulkhead
- Timeout protection
- Failure tracking and circuit breaking
- Metrics collection
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, TypeVar
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BulkheadState(Enum):
    """Bulkhead circuit state."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if recovered


class BulkheadError(Exception):
    """Base exception for bulkhead errors."""
    pass


class BulkheadOpenError(BulkheadError):
    """Raised when bulkhead is open and rejecting requests."""

    def __init__(self, bulkhead_name: str, reason: str = ""):
        self.bulkhead_name = bulkhead_name
        self.reason = reason
        message = f"Bulkhead '{bulkhead_name}' is open"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class BulkheadTimeoutError(BulkheadError):
    """Raised when operation times out within bulkhead."""

    def __init__(self, bulkhead_name: str, timeout: float):
        self.bulkhead_name = bulkhead_name
        self.timeout = timeout
        super().__init__(f"Operation in bulkhead '{bulkhead_name}' timed out after {timeout}s")


class BulkheadFullError(BulkheadError):
    """Raised when bulkhead has reached max concurrent requests."""

    def __init__(self, bulkhead_name: str, max_concurrent: int):
        self.bulkhead_name = bulkhead_name
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Bulkhead '{bulkhead_name}' is full (max {max_concurrent} concurrent)"
        )


@dataclass
class BulkheadConfig:
    """Configuration for a bulkhead."""
    max_concurrent: int = 5           # Max concurrent executions
    timeout: float = 30.0             # Timeout per operation (seconds)
    failure_threshold: int = 5        # Failures before opening circuit
    success_threshold: int = 3        # Successes to close circuit after half-open
    open_duration: float = 60.0       # Time to stay open before half-open (seconds)
    failure_window: float = 60.0      # Window for counting failures (seconds)


@dataclass
class BulkheadMetrics:
    """Metrics for a bulkhead."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    timeout_requests: int = 0
    current_concurrent: int = 0
    max_concurrent_reached: int = 0
    state_changes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class Bulkhead:
    """
    Bulkhead for isolating failures between operations.

    Implements:
    - Concurrency limiting with semaphore
    - Timeout protection
    - Circuit breaker pattern
    - Failure tracking

    Example:
        bulkhead = Bulkhead("orders", BulkheadConfig(max_concurrent=3))

        async def place_order(symbol, quantity):
            return await bulkhead.execute(
                _do_place_order(symbol, quantity)
            )
    """

    def __init__(self, name: str, config: Optional[BulkheadConfig] = None):
        self.name = name
        self.config = config or BulkheadConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._state = BulkheadState.CLOSED
        self._state_changed_at = datetime.now()
        self._failures: deque = deque()  # Recent failure timestamps
        self._half_open_successes = 0
        self._lock = asyncio.Lock()
        self.metrics = BulkheadMetrics()

    @property
    def state(self) -> BulkheadState:
        """Get current bulkhead state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if bulkhead is open (rejecting requests)."""
        return self._state == BulkheadState.OPEN

    async def _update_state(self):
        """Update state based on failures and timeouts."""
        async with self._lock:
            now = datetime.now()

            # Clean old failures outside window
            window_start = now - timedelta(seconds=self.config.failure_window)
            while self._failures and self._failures[0] < window_start:
                self._failures.popleft()

            if self._state == BulkheadState.CLOSED:
                # Check if we should open
                if len(self._failures) >= self.config.failure_threshold:
                    self._transition_to(BulkheadState.OPEN)
                    logger.warning(
                        f"Bulkhead '{self.name}' opened after {len(self._failures)} failures"
                    )

            elif self._state == BulkheadState.OPEN:
                # Check if we should try half-open
                time_in_open = (now - self._state_changed_at).total_seconds()
                if time_in_open >= self.config.open_duration:
                    self._transition_to(BulkheadState.HALF_OPEN)
                    self._half_open_successes = 0
                    logger.info(f"Bulkhead '{self.name}' entering half-open state")

            elif self._state == BulkheadState.HALF_OPEN:
                # Check if we should close or re-open
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(BulkheadState.CLOSED)
                    self._failures.clear()
                    logger.info(f"Bulkhead '{self.name}' closed after recovery")

    def _transition_to(self, new_state: BulkheadState):
        """Transition to a new state."""
        self._state = new_state
        self._state_changed_at = datetime.now()
        self.metrics.state_changes += 1

    async def _record_failure(self):
        """Record a failure."""
        async with self._lock:
            self._failures.append(datetime.now())
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()

            if self._state == BulkheadState.HALF_OPEN:
                # Any failure in half-open re-opens the circuit
                self._transition_to(BulkheadState.OPEN)
                logger.warning(f"Bulkhead '{self.name}' re-opened after failure in half-open")

    async def _record_success(self):
        """Record a success."""
        async with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now()

            if self._state == BulkheadState.HALF_OPEN:
                self._half_open_successes += 1

    async def execute(
        self,
        coro,
        timeout: Optional[float] = None,
        fallback: Optional[Callable[[], T]] = None
    ) -> T:
        """
        Execute a coroutine within the bulkhead.

        Args:
            coro: Coroutine to execute
            timeout: Override default timeout (optional)
            fallback: Function to call if execution fails (optional)

        Returns:
            Result of the coroutine or fallback

        Raises:
            BulkheadOpenError: If bulkhead is open
            BulkheadTimeoutError: If operation times out
            BulkheadFullError: If max concurrent reached (non-blocking mode)
        """
        self.metrics.total_requests += 1

        # Check state and possibly transition
        await self._update_state()

        # Reject if open
        if self._state == BulkheadState.OPEN:
            self.metrics.rejected_requests += 1
            if fallback:
                return fallback()
            raise BulkheadOpenError(self.name, "Circuit is open due to failures")

        effective_timeout = timeout or self.config.timeout

        # Try to acquire semaphore
        try:
            acquired = self.semaphore.locked()
            if acquired:
                self.metrics.current_concurrent = self.config.max_concurrent - self.semaphore._value

            async with self.semaphore:
                self.metrics.current_concurrent = self.config.max_concurrent - self.semaphore._value
                if self.metrics.current_concurrent > self.metrics.max_concurrent_reached:
                    self.metrics.max_concurrent_reached = self.metrics.current_concurrent

                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(coro, timeout=effective_timeout)
                    await self._record_success()
                    return result

                except asyncio.TimeoutError:
                    self.metrics.timeout_requests += 1
                    await self._record_failure()
                    await self._update_state()

                    if fallback:
                        return fallback()
                    raise BulkheadTimeoutError(self.name, effective_timeout)

                except Exception as e:
                    await self._record_failure()
                    await self._update_state()

                    if fallback:
                        logger.warning(
                            f"Bulkhead '{self.name}' execution failed, using fallback: {e}"
                        )
                        return fallback()
                    raise

        finally:
            self.metrics.current_concurrent = self.config.max_concurrent - self.semaphore._value

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        return {
            "name": self.name,
            "state": self._state.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "rejected_requests": self.metrics.rejected_requests,
            "timeout_requests": self.metrics.timeout_requests,
            "current_concurrent": self.metrics.current_concurrent,
            "max_concurrent_reached": self.metrics.max_concurrent_reached,
            "failure_rate": (
                self.metrics.failed_requests / self.metrics.total_requests
                if self.metrics.total_requests > 0 else 0
            ),
            "state_changes": self.metrics.state_changes,
        }

    async def reset(self):
        """Reset bulkhead to initial state."""
        async with self._lock:
            self._state = BulkheadState.CLOSED
            self._state_changed_at = datetime.now()
            self._failures.clear()
            self._half_open_successes = 0
            self.metrics = BulkheadMetrics()
            logger.info(f"Bulkhead '{self.name}' reset")


class BulkheadRegistry:
    """
    Registry for managing multiple bulkheads.

    Example:
        registry = BulkheadRegistry()
        registry.create("orders", BulkheadConfig(max_concurrent=3))
        registry.create("market_data", BulkheadConfig(max_concurrent=10))

        async def place_order(...):
            return await registry.execute("orders", _do_place_order(...))
    """

    def __init__(self):
        self._bulkheads: Dict[str, Bulkhead] = {}
        self._lock = asyncio.Lock()

    def create(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None
    ) -> Bulkhead:
        """Create and register a new bulkhead."""
        bulkhead = Bulkhead(name, config)
        self._bulkheads[name] = bulkhead
        return bulkhead

    def get(self, name: str) -> Optional[Bulkhead]:
        """Get a bulkhead by name."""
        return self._bulkheads.get(name)

    def get_or_create(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None
    ) -> Bulkhead:
        """Get existing bulkhead or create new one."""
        if name not in self._bulkheads:
            self.create(name, config)
        return self._bulkheads[name]

    async def execute(
        self,
        bulkhead_name: str,
        coro,
        **kwargs
    ):
        """Execute within a named bulkhead."""
        bulkhead = self._bulkheads.get(bulkhead_name)
        if not bulkhead:
            raise ValueError(f"Bulkhead '{bulkhead_name}' not found")
        return await bulkhead.execute(coro, **kwargs)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all bulkheads."""
        return {
            name: bulkhead.get_metrics()
            for name, bulkhead in self._bulkheads.items()
        }

    async def reset_all(self):
        """Reset all bulkheads."""
        for bulkhead in self._bulkheads.values():
            await bulkhead.reset()


# Global registry instance
_registry: Optional[BulkheadRegistry] = None


def get_bulkhead_registry() -> BulkheadRegistry:
    """Get the global bulkhead registry."""
    global _registry
    if _registry is None:
        _registry = BulkheadRegistry()
    return _registry


def create_trading_bulkheads() -> BulkheadRegistry:
    """
    Create standard bulkheads for trading system.

    Returns configured bulkheads for:
    - orders: Order placement/execution
    - market_data: Market data fetching
    - historical_data: Historical data fetching
    - signals: Signal generation
    """
    registry = get_bulkhead_registry()

    registry.create("orders", BulkheadConfig(
        max_concurrent=3,
        timeout=30.0,
        failure_threshold=3,
        success_threshold=2,
        open_duration=60.0
    ))

    registry.create("market_data", BulkheadConfig(
        max_concurrent=10,
        timeout=10.0,
        failure_threshold=5,
        success_threshold=3,
        open_duration=30.0
    ))

    registry.create("historical_data", BulkheadConfig(
        max_concurrent=5,
        timeout=60.0,
        failure_threshold=5,
        success_threshold=2,
        open_duration=120.0
    ))

    registry.create("signals", BulkheadConfig(
        max_concurrent=20,
        timeout=5.0,
        failure_threshold=10,
        success_threshold=5,
        open_duration=30.0
    ))

    logger.info("Trading bulkheads initialized")
    return registry
