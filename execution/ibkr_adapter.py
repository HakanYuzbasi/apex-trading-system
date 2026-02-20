import asyncio
import logging
import time
from typing import Dict, Any, Optional
from execution.ibkr_connector import IBKRConnector
from config import ApexConfig

logger = logging.getLogger(__name__)

class IBKRAdapterException(Exception):
    pass

class CircuitBreakerOpenException(IBKRAdapterException):
    pass

class IBKRAdapter:
    """
    Adapter around IBKRConnector providing institutional resilience:
    - Circuit Breaking (fails fast if error rate is structurally too high)
    - Stale-value fallback (caches last known good values)
    - Strict timeouts for uncooperative API calls (prevents asyncio lockups)
    """
    def __init__(self, connector: IBKRConnector):
        self.connector = connector
        self._circuit_open = False
        self._error_count = 0
        self._last_error_time = 0.0
        self._failure_threshold = getattr(ApexConfig, "IBKR_CIRCUIT_BREAKER_FAILURES", 5)
        self._recovery_timeout_sec = getattr(ApexConfig, "IBKR_CIRCUIT_BREAKER_RECOVERY_SEC", 30)
        
        # Stale value cache: method_name + args_hash -> (value, timestamp)
        self._value_cache: Dict[str, tuple[Any, float]] = {}
        # Max age to trust a stale value
        self._stale_max_age = getattr(ApexConfig, "IBKR_STALE_MAX_AGE_SEC", 300)

    def _check_circuit(self):
        now = time.time()
        if self._circuit_open:
            if now - self._last_error_time > self._recovery_timeout_sec:
                logger.info("IBKR Circuit Breaker entering HALF-OPEN state for recovery check.")
                self._circuit_open = False
                # If the very next call fails, we want it to immediately re-trip the breaker
                self._error_count = max(0, self._failure_threshold - 1)
            else:
                raise CircuitBreakerOpenException("IBKR Circuit Breaker is OPEN. Halting calls.")

    def _record_success(self):
        self._error_count = 0
        if self._circuit_open:
            logger.info("IBKR Circuit Breaker successfully CLOSED.")
            self._circuit_open = False

    def _record_failure(self, error: Exception):
        now = time.time()
        self._error_count += 1
        self._last_error_time = now
        
        if self._error_count >= self._failure_threshold:
            if not self._circuit_open:
                logger.error(f"IBKR Circuit Breaker TRIPPED due to {self._error_count} consecutive failures: {error}")
                self._circuit_open = True

    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        key_parts = [method] + [str(a) for a in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        return "::".join(key_parts)
        
    async def _execute_with_resilience(self, method_name: str, fallback_allowed: bool, timeout: float, *args, **kwargs) -> Any:
        self._check_circuit()
        cache_key = self._get_cache_key(method_name, *args, **kwargs)
        func = getattr(self.connector, method_name)
        
        try:
            # Enforce strict timeout wrapper around the connector call
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            
            # Record success and update cache for future stale fallbacks
            self._record_success()
            self._value_cache[cache_key] = (result, time.time())
            return result
            
        except asyncio.TimeoutError as e:
            self._record_failure(e)
            if fallback_allowed:
                stale_val = self._attempt_stale_fallback(cache_key)
                if stale_val is not None:
                    return stale_val
            raise IBKRAdapterException(f"IBKR API call '{method_name}' timed out after {timeout}s") from e
            
        except Exception as e:
            self._record_failure(e)
            if fallback_allowed:
                stale_val = self._attempt_stale_fallback(cache_key)
                if stale_val is not None:
                    return stale_val
            raise IBKRAdapterException(f"IBKR API call '{method_name}' failed: {str(e)}") from e

    def _attempt_stale_fallback(self, cache_key: str) -> Optional[Any]:
        if cache_key in self._value_cache:
            val, ts = self._value_cache[cache_key]
            age = time.time() - ts
            if age <= self._stale_max_age:
                logger.warning(f"⚠️  Using stale-value fallback for '{cache_key}' (age: {age:.1f}s)")
                return val
        return None

    # ──────────────────────────────────────────────────────────
    # Wrapped Contract Methods
    # ──────────────────────────────────────────────────────────
    
    async def connect(self):
        try:
            self._check_circuit()
            # Connection has a higher baseline timeout inherently
            await asyncio.wait_for(self.connector.connect(), timeout=15.0)
            self._record_success()
        except Exception as e:
            self._record_failure(e)
            raise IBKRAdapterException(f"IBKR connection failed: {str(e)}") from e
            
    def disconnect(self):
        self.connector.disconnect()

    def is_connected(self) -> bool:
        return self.connector.is_connected()

    async def get_market_price(self, symbol: str) -> float:
        return await self._execute_with_resilience(
            "get_market_price", fallback_allowed=True, timeout=5.0, symbol=symbol
        )

    async def get_position(self, symbol: str) -> float:
        return await self._execute_with_resilience(
            "get_position", fallback_allowed=True, timeout=5.0, symbol=symbol
        )

    async def get_all_positions(self) -> Dict[str, float]:
        # We don't allow fallback for the entire global position block to prevent sync drift
        return await self._execute_with_resilience(
            "get_all_positions", fallback_allowed=False, timeout=10.0
        )

    async def get_portfolio_value(self) -> float:
        return await self._execute_with_resilience(
            "get_portfolio_value", fallback_allowed=True, timeout=10.0
        )
