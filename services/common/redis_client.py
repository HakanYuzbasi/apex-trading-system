"""
services/common/redis_client.py - Redis client singleton with rate-limiting helpers.

Used for session caching, feature-flag caching, and per-user rate limiting.
"""

import os
import json
import logging
import asyncio
from datetime import date
from typing import Any, Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis = None
_redis_lock = asyncio.Lock()
_REDIS_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    logger.warning("redis package not installed - caching/rate-limiting disabled")


async def get_redis():
    """Return the global async Redis client (creates on first call)."""
    global _redis
    if not _REDIS_AVAILABLE:
        return None
    
    if _redis is not None:
        return _redis

    async with _redis_lock:
        # Double-check after acquiring lock
        if _redis is not None:
            return _redis
            
        try:
            client = aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await client.ping()
            _redis = client
            logger.info("Redis connected: %s", REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL)
        except Exception as e:
            logger.warning("Redis connection failed (%s) - caching disabled", e)
            _redis = None
            
    return _redis


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

async def cache_get(key: str) -> Optional[Any]:
    """Get a JSON value from Redis cache."""
    r = await get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    """Set a JSON value in Redis cache with TTL."""
    r = await get_redis()
    if r is None:
        return
    try:
        await r.setex(key, ttl_seconds, json.dumps(value, default=str))
    except Exception as e:
        logger.debug("Redis cache_set error: %s", e)


async def cache_delete(key: str) -> None:
    """Delete a key from Redis."""
    r = await get_redis()
    if r is None:
        return
    try:
        await r.delete(key)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

async def rate_check(user_id: str, feature_key: str, daily_limit: int) -> bool:
    """
    Check if a user is within their daily rate limit for a feature.
    Returns True if allowed, False if rate-limited.
    A daily_limit of -1 means unlimited.
    """
    if daily_limit == -1:
        return True

    r = await get_redis()
    if r is None:
        # No Redis = no rate limiting (fail open)
        return True

    today = date.today().isoformat()
    key = f"rate:{user_id}:{feature_key}:{today}"

    try:
        current = await r.incr(key)
        if current == 1:
            # First request today - set 24h expiry
            await r.expire(key, 86400)
        return current <= daily_limit
    except Exception:
        return True  # Fail open


async def rate_get_remaining(user_id: str, feature_key: str, daily_limit: int) -> int:
    """Get remaining API calls for today."""
    if daily_limit == -1:
        return -1  # unlimited

    r = await get_redis()
    if r is None:
        return daily_limit

    today = date.today().isoformat()
    key = f"rate:{user_id}:{feature_key}:{today}"

    try:
        current = await r.get(key)
        used = int(current) if current else 0
        return max(0, daily_limit - used)
    except Exception:
        return daily_limit
