"""
core/cache.py - Time-To-Live caching utilities

Provides lightweight in-memory caching with automatic expiration to reduce
disk I/O and improve API response times.
"""

import functools
import time
import threading
from typing import Any, Callable, Dict, Tuple


class TTLCache:
    """
    Time-To-Live cache with automatic expiration.
    
    Thread-safe cache that automatically evicts stale entries based on TTL
    and implements LRU eviction when max size is exceeded.
    
    Args:
        ttl_seconds: Time-to-live in seconds for cached items
        maxsize: Maximum number of items to cache (LRU eviction)
    
    Example:
        @TTLCache(ttl_seconds=5.0)
        def expensive_operation(x):
            return x ** 2
    """
    
    def __init__(self, ttl_seconds: float = 1.0, maxsize: int = 128):
        self.ttl = ttl_seconds
        self.maxsize = maxsize
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            key = self._make_key(args, kwargs)
            now = time.time()
            
            with self.lock:
                # Check if cached and fresh
                if key in self.cache:
                    age = now - self.timestamps[key]
                    if age < self.ttl:
                        self.hits += 1
                        return self.cache[key]
                    else:
                        # Stale entry, remove it
                        del self.cache[key]
                        del self.timestamps[key]
                
                self.misses += 1
            
            # Cache miss or stale - compute result
            result = func(*args, **kwargs)
            
            with self.lock:
                # Store in cache
                self.cache[key] = result
                self.timestamps[key] = now
                
                # Evict oldest if max size exceeded
                if len(self.cache) > self.maxsize:
                    oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: self._clear()
        wrapper.cache_info = lambda: self._info()
        wrapper._cache_instance = self
        
        return wrapper
    
    def _make_key(self, args: Tuple, kwargs: Dict) -> str:
        """Create a cache key from function arguments."""
        try:
            # Simple key generation - works for most cases
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            return ":".join(key_parts)
        except Exception:
            # Fallback to hash if str() fails
            return str(hash((args, tuple(sorted(kwargs.items())))))
    
    def _clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def _info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0.0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "ttl": self.ttl,
            }


# Pre-configured cache decorators for common use cases
cache_1s = TTLCache(ttl_seconds=1.0, maxsize=128)
cache_5s = TTLCache(ttl_seconds=5.0, maxsize=64)
cache_10s = TTLCache(ttl_seconds=10.0, maxsize=32)
cache_30s = TTLCache(ttl_seconds=30.0, maxsize=16)
cache_60s = TTLCache(ttl_seconds=60.0, maxsize=16)


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all pre-configured caches."""
    return {
        "cache_1s": cache_1s._info(),
        "cache_5s": cache_5s._info(),
        "cache_10s": cache_10s._info(),
        "cache_30s": cache_30s._info(),
        "cache_60s": cache_60s._info(),
    }


def clear_all_caches():
    """Clear all pre-configured caches."""
    cache_1s._clear()
    cache_5s._clear()
    cache_10s._clear()
    cache_30s._clear()
    cache_60s._clear()
