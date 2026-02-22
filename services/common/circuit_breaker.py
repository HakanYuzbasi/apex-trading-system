"""
services/common/circuit_breaker.py
Production-Grade SLO Circuit Breaker
Monitors API errors and latency. Transitions system into Fail-Safe Mode if breached.
"""
import time
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self, error_threshold: int = 5, recovery_window: int = 60, slo_latency_ms: float = 50.0) -> None:
        self.error_threshold = error_threshold
        self.recovery_window = recovery_window
        self.slo_latency_ms = slo_latency_ms
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED = OK, OPEN = Tripped, HALF_OPEN = Testing

    def record_error(self) -> None:
        """Increments error count and potentially trips the breaker."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.error_threshold and self.state == "CLOSED":
            self._trip("API Error Threshold Exceeded")

    def record_latency(self, latency_ms: float) -> None:
        """Monitors execution speed against defined SLOs."""
        if latency_ms > self.slo_latency_ms:
            logger.warning(f"🐌 SLO Breach: Latency {latency_ms:.2f}ms > {self.slo_latency_ms}ms")
            self.record_error()
        elif self.state == "HALF_OPEN":
            self._reset()

    def _trip(self, reason: str) -> None:
        """Activates Fail-Safe Mode."""
        self.state = "OPEN"
        logger.critical(f"🛑 CIRCUIT BREAKER TRIPPED: {reason}. System transitioned to FAIL-SAFE MODE.")

    def _reset(self) -> None:
        """Restores normal trading operations."""
        self.state = "CLOSED"
        self.failures = 0
        logger.info("✅ Circuit Breaker Reset. Normal operations resumed.")

    def is_allowed(self) -> bool:
        """Gatekeeper check before allowing API calls or Order Routing."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_window:
                self.state = "HALF_OPEN"
                logger.info("⚠️ Circuit Breaker Half-Open: Testing connection stability...")
                return True
            return False
        return True
