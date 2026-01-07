"""
utils/circuit_breaker.py - Circuit Breaker Pattern for API Resilience
Prevents cascading failures by stopping calls to failing services
"""

import logging
import time
from enum import Enum
from typing import Callable, Any, Optional
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject calls
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.

    Protects against cascading failures by:
    1. CLOSED: Normal operation, monitoring failures
    2. OPEN: After threshold failures, reject calls immediately
    3. HALF_OPEN: After timeout, allow test calls to check recovery

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)

        @breaker
        def api_call():
            # ... risky operation ...
            pass
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exceptions: tuple = (Exception,)
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Successes in half-open before closing
            timeout: Seconds to wait before trying half-open
            expected_exceptions: Exceptions that count as failures
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exceptions = expected_exceptions

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: datetime = datetime.now()

        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_rejections = 0

    def __call__(self, func: Callable) -> Callable:
        """Decorator usage."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If call fails
        """
        self.total_calls += 1

        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.total_rejections += 1
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. "
                    f"Service unavailable. Retry in {self._time_until_retry():.0f}s"
                )

        try:
            # Attempt the call
            result = func(*args, **kwargs)

            # Success
            self._on_success()
            return result

        except self.expected_exceptions as e:
            # Failure
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        self.total_successes += 1
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()

    def _on_failure(self):
        """Handle failed call."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during testing, go back to open
            self._transition_to_open()

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self.last_failure_time is None:
            return True

        return (time.time() - self.last_failure_time) >= self.timeout

    def _time_until_retry(self) -> float:
        """Get seconds until circuit will try half-open."""
        if self.last_failure_time is None:
            return 0.0

        elapsed = time.time() - self.last_failure_time
        return max(0.0, self.timeout - elapsed)

    def _transition_to_open(self):
        """Transition to OPEN state."""
        if self.state != CircuitState.OPEN:
            logger.warning(f"🔴 Circuit breaker OPEN (failures: {self.failure_count})")
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            self.success_count = 0

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        if self.state != CircuitState.HALF_OPEN:
            logger.info(f"🟡 Circuit breaker HALF_OPEN (testing recovery)")
            self.state = CircuitState.HALF_OPEN
            self.last_state_change = datetime.now()
            self.success_count = 0
            self.failure_count = 0

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        if self.state != CircuitState.CLOSED:
            logger.info(f"🟢 Circuit breaker CLOSED (service recovered)")
            self.state = CircuitState.CLOSED
            self.last_state_change = datetime.now()
            self.failure_count = 0
            self.success_count = 0

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        logger.info("🔄 Circuit breaker manually reset")
        self._transition_to_closed()

    def get_status(self) -> dict:
        """
        Get current circuit breaker status.

        Returns:
            Dict with state and statistics
        """
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_calls': self.total_calls,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'total_rejections': self.total_rejections,
            'failure_rate': self.total_failures / self.total_calls if self.total_calls > 0 else 0,
            'last_state_change': self.last_state_change.isoformat(),
            'time_until_retry': self._time_until_retry() if self.state == CircuitState.OPEN else 0
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Global circuit breakers for common services
IBKR_BREAKER = CircuitBreaker(
    failure_threshold=5,
    success_threshold=2,
    timeout=60.0,
    expected_exceptions=(ConnectionError, TimeoutError, Exception)
)

MARKET_DATA_BREAKER = CircuitBreaker(
    failure_threshold=10,
    success_threshold=3,
    timeout=30.0,
    expected_exceptions=(ConnectionError, TimeoutError, Exception)
)
