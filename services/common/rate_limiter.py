"""Rate limiting and throttling for API endpoints and service calls.

Provides token bucket, sliding window, and adaptive rate limiting algorithms
with distributed support via Redis.
"""

import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque
import redis.asyncio as redis
from fastapi import HTTPException, Request
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int  # Maximum requests allowed
    time_window: int  # Time window in seconds
    burst_size: int = 0  # Allow burst capacity (0 = no burst)
    strategy: str = "token_bucket"  # token_bucket, sliding_window, adaptive
    

@dataclass
class RateLimitState:
    """Current state of rate limiter for a key."""
    tokens: float
    last_update: float
    request_times: deque = field(default_factory=deque)
    blocked_until: Optional[float] = None
    

class TokenBucketRateLimiter:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.rate = config.max_requests / config.time_window  # tokens per second
        self.capacity = config.max_requests + config.burst_size
        self.states: Dict[str, RateLimitState] = {}
        
    async def is_allowed(self, key: str) -> Tuple[bool, Dict]:
        """Check if request is allowed under rate limit.
        
        Returns:
            Tuple of (allowed, info_dict with remaining, reset_time)
        """
        now = time.time()
        
        if key not in self.states:
            self.states[key] = RateLimitState(
                tokens=self.capacity,
                last_update=now
            )
            
        state = self.states[key]
        
        # Check if currently blocked
        if state.blocked_until and now < state.blocked_until:
            return False, {
                "remaining": 0,
                "reset": int(state.blocked_until),
                "retry_after": int(state.blocked_until - now)
            }
        
        # Add tokens based on time passed
        elapsed = now - state.last_update
        state.tokens = min(self.capacity, state.tokens + elapsed * self.rate)
        state.last_update = now
        
        # Check if we have tokens
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            return True, {
                "remaining": int(state.tokens),
                "reset": int(now + (self.capacity - state.tokens) / self.rate),
                "limit": self.config.max_requests
            }
        else:
            # Calculate when next token will be available
            wait_time = (1.0 - state.tokens) / self.rate
            state.blocked_until = now + wait_time
            return False, {
                "remaining": 0,
                "reset": int(state.blocked_until),
                "retry_after": int(wait_time),
                "limit": self.config.max_requests
            }


class SlidingWindowRateLimiter:
    """Sliding window algorithm for rate limiting."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.states: Dict[str, RateLimitState] = {}
        
    async def is_allowed(self, key: str) -> Tuple[bool, Dict]:
        """Check if request is allowed under rate limit."""
        now = time.time()
        cutoff = now - self.config.time_window
        
        if key not in self.states:
            self.states[key] = RateLimitState(
                tokens=0,
                last_update=now
            )
            
        state = self.states[key]
        
        # Remove old requests outside the window
        while state.request_times and state.request_times[0] < cutoff:
            state.request_times.popleft()
        
        # Check if we're under the limit
        if len(state.request_times) < self.config.max_requests:
            state.request_times.append(now)
            return True, {
                "remaining": self.config.max_requests - len(state.request_times),
                "reset": int(now + self.config.time_window),
                "limit": self.config.max_requests
            }
        else:
            # Calculate when oldest request will expire
            oldest = state.request_times[0]
            retry_after = int(oldest + self.config.time_window - now)
            return False, {
                "remaining": 0,
                "reset": int(oldest + self.config.time_window),
                "retry_after": retry_after,
                "limit": self.config.max_requests
            }


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.base_limiter = TokenBucketRateLimiter(config)
        self.error_counts: Dict[str, deque] = {}
        self.success_counts: Dict[str, deque] = {}
        self.adjustment_factor = 1.0
        
    async def record_outcome(self, key: str, success: bool):
        """Record request outcome for adaptive adjustment."""
        now = time.time()
        cutoff = now - 60  # Look at last minute
        
        if key not in self.error_counts:
            self.error_counts[key] = deque()
            self.success_counts[key] = deque()
        
        # Clean old entries
        errors = self.error_counts[key]
        successes = self.success_counts[key]
        
        while errors and errors[0] < cutoff:
            errors.popleft()
        while successes and successes[0] < cutoff:
            successes.popleft()
        
        # Record outcome
        if success:
            successes.append(now)
        else:
            errors.append(now)
        
        # Adjust rate based on error rate
        total = len(errors) + len(successes)
        if total > 10:  # Need minimum samples
            error_rate = len(errors) / total
            if error_rate > 0.1:  # More than 10% errors
                self.adjustment_factor = max(0.5, self.adjustment_factor * 0.9)
                logger.warning(
                    f"High error rate {error_rate:.1%} for {key}, "
                    f"reducing rate to {self.adjustment_factor:.1%}"
                )
            elif error_rate < 0.01:  # Less than 1% errors
                self.adjustment_factor = min(1.0, self.adjustment_factor * 1.1)
                
    async def is_allowed(self, key: str) -> Tuple[bool, Dict]:
        """Check if request is allowed with adaptive limits."""
        # Temporarily adjust capacity
        original_capacity = self.base_limiter.capacity
        self.base_limiter.capacity = int(original_capacity * self.adjustment_factor)
        
        result = await self.base_limiter.is_allowed(key)
        
        # Restore capacity
        self.base_limiter.capacity = original_capacity
        
        return result


class DistributedRateLimiter:
    """Distributed rate limiter using Redis."""
    
    def __init__(self, redis_client: redis.Redis, config: RateLimitConfig, prefix: str = "ratelimit"):
        self.redis = redis_client
        self.config = config
        self.prefix = prefix
        
    def _get_key(self, identifier: str) -> str:
        """Generate Redis key for rate limit tracking."""
        return f"{self.prefix}:{identifier}"
    
    async def is_allowed(self, identifier: str) -> Tuple[bool, Dict]:
        """Check if request is allowed using Redis."""
        key = self._get_key(identifier)
        now = int(time.time())
        window_start = now - self.config.time_window
        
        # Use Redis sorted set for sliding window
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, self.config.time_window + 10)
        
        results = await pipe.execute()
        count = results[1]  # Get count before adding new request
        
        if count < self.config.max_requests:
            return True, {
                "remaining": self.config.max_requests - count - 1,
                "reset": now + self.config.time_window,
                "limit": self.config.max_requests
            }
        else:
            # Get oldest request to calculate retry time
            oldest_items = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest_items:
                oldest_time = int(oldest_items[0][1])
                retry_after = oldest_time + self.config.time_window - now
            else:
                retry_after = self.config.time_window
            
            # Remove the request we just added since it's not allowed
            await self.redis.zrem(key, str(now))
            
            return False, {
                "remaining": 0,
                "reset": now + self.config.time_window,
                "retry_after": max(1, retry_after),
                "limit": self.config.max_requests
            }


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, limiter, get_key_func=None):
        self.limiter = limiter
        self.get_key_func = get_key_func or self._default_get_key
        
    def _default_get_key(self, request: Request) -> str:
        """Extract rate limit key from request (default: client IP)."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        key = self.get_key_func(request)
        
        allowed, info = await self.limiter.is_allowed(key)
        
        if not allowed:
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(info.get("limit", 0)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(info.get("reset", 0)),
                    "Retry-After": str(info.get("retry_after", 60))
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info.get("limit", 0))
        response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
        
        response.headers["X-RateLimit-Reset"] = str(info.get("reset", 0))
        
        return response


def create_rate_limiter(config: RateLimitConfig, redis_client: Optional[redis.Redis] = None):
    """Factory function to create appropriate rate limiter."""
    if redis_client:
        return DistributedRateLimiter(redis_client, config)
    elif config.strategy == "sliding_window":
        return SlidingWindowRateLimiter(config)
    elif config.strategy == "adaptive":
        return AdaptiveRateLimiter(config)
    else:
        return TokenBucketRateLimiter(config)
