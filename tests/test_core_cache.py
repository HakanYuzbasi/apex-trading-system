# tests/test_core_cache.py - TTLCache unit tests

import time
import threading
import pytest
from core.cache import TTLCache, get_cache_stats, clear_all_caches


class TestTTLCacheBasic:
    """Test basic cache hit/miss behaviour."""

    def test_caches_return_value(self):
        """Cached function returns the correct computed value."""
        call_count = 0

        @TTLCache(ttl_seconds=10.0)
        def square(x):
            nonlocal call_count
            call_count += 1
            return x ** 2

        assert square(4) == 16
        assert call_count == 1

    def test_cache_hit_avoids_recomputation(self):
        """Second call with same args returns cached result without re-calling."""
        call_count = 0

        @TTLCache(ttl_seconds=10.0)
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        assert add(2, 3) == 5
        assert add(2, 3) == 5
        assert call_count == 1

    def test_different_args_produce_separate_entries(self):
        """Different arguments are cached independently."""
        call_count = 0

        @TTLCache(ttl_seconds=10.0)
        def double(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert double(1) == 2
        assert double(2) == 4
        assert call_count == 2

    def test_kwargs_are_part_of_cache_key(self):
        """Keyword arguments differentiate cache entries."""
        call_count = 0

        @TTLCache(ttl_seconds=10.0)
        def greet(name="world"):
            nonlocal call_count
            call_count += 1
            return f"hello {name}"

        assert greet(name="alice") == "hello alice"
        assert greet(name="bob") == "hello bob"
        assert greet(name="alice") == "hello alice"
        assert call_count == 2


class TestTTLExpiration:
    """Test time-to-live expiration logic."""

    def test_stale_entry_is_evicted(self):
        """Entry older than TTL is recomputed on next access."""
        call_count = 0

        @TTLCache(ttl_seconds=0.05)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x

        compute(1)
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(0.08)

        compute(1)
        assert call_count == 2

    def test_fresh_entry_is_not_evicted(self):
        """Entry within TTL is served from cache."""
        call_count = 0

        @TTLCache(ttl_seconds=5.0)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x

        compute(1)
        compute(1)
        assert call_count == 1


class TestMaxSizeEviction:
    """Test LRU eviction when max size is exceeded."""

    def test_evicts_oldest_when_maxsize_exceeded(self):
        """Oldest entry is removed when cache exceeds maxsize."""
        @TTLCache(ttl_seconds=60.0, maxsize=2)
        def identity(x):
            return x

        identity(1)
        identity(2)
        identity(3)  # should evict key for arg 1

        cache_inst = identity._cache_instance
        assert cache_inst._info()["size"] == 2
        # Key "1" (the oldest) should have been evicted
        assert "1" not in cache_inst.cache

    def test_maxsize_one(self):
        """Cache with maxsize=1 only keeps the last entry."""
        call_count = 0

        @TTLCache(ttl_seconds=60.0, maxsize=1)
        def compute(x):
            nonlocal call_count
            call_count += 1
            return x * 10

        compute(1)
        compute(2)  # evicts 1
        assert call_count == 2

        result = compute(1)  # re-computed because evicted
        assert result == 10
        assert call_count == 3


class TestCacheManagementMethods:
    """Test cache_clear, cache_info, and _make_key."""

    def test_cache_clear_resets_everything(self):
        """cache_clear empties all entries and resets stats."""
        @TTLCache(ttl_seconds=60.0)
        def fn(x):
            return x

        fn(1)
        fn(2)
        fn(1)  # hit

        info = fn.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 2
        assert info["size"] == 2

        fn.cache_clear()

        info = fn.cache_info()
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["size"] == 0

    def test_cache_info_hit_rate(self):
        """Hit rate is calculated correctly."""
        @TTLCache(ttl_seconds=60.0)
        def fn(x):
            return x

        fn(1)       # miss
        fn(1)       # hit
        fn(1)       # hit
        fn(2)       # miss

        info = fn.cache_info()
        assert info["hits"] == 2
        assert info["misses"] == 2
        assert info["hit_rate"] == "50.0%"
        assert info["maxsize"] == 128

    def test_cache_info_zero_calls(self):
        """Hit rate is 0.0% when no calls have been made."""
        @TTLCache(ttl_seconds=60.0)
        def fn(x):
            return x

        info = fn.cache_info()
        assert info["hit_rate"] == "0.0%"
        assert info["size"] == 0

    def test_make_key_sorted_kwargs(self):
        """Key generation is stable regardless of kwarg ordering."""
        cache = TTLCache()
        key1 = cache._make_key((), {"a": 1, "b": 2})
        key2 = cache._make_key((), {"b": 2, "a": 1})
        assert key1 == key2

    def test_make_key_mixed_args_kwargs(self):
        """Positional and keyword args both contribute to the key."""
        cache = TTLCache()
        key_a = cache._make_key((1,), {"mode": "fast"})
        key_b = cache._make_key((2,), {"mode": "fast"})
        assert key_a != key_b


class TestFuncWrapsPreserved:
    """Ensure decorated function preserves its identity."""

    def test_preserves_function_name(self):
        """functools.wraps keeps original __name__."""
        @TTLCache(ttl_seconds=1.0)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


class TestThreadSafety:
    """Test concurrent access to the cache."""

    def test_concurrent_writes_do_not_corrupt(self):
        """Multiple threads writing simultaneously keep cache consistent."""
        call_count = 0
        lock = threading.Lock()

        @TTLCache(ttl_seconds=60.0, maxsize=256)
        def compute(x):
            nonlocal call_count
            with lock:
                call_count += 1
            return x * 2

        errors = []

        def worker(start, end):
            try:
                for i in range(start, end):
                    assert compute(i) == i * 2
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 50, (i + 1) * 50)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        info = compute.cache_info()
        assert info["size"] <= 256


class TestModuleLevelHelpers:
    """Test get_cache_stats and clear_all_caches."""

    def test_get_cache_stats_returns_all_caches(self):
        """get_cache_stats includes all five pre-configured caches."""
        stats = get_cache_stats()
        expected_keys = {"cache_1s", "cache_5s", "cache_10s", "cache_30s", "cache_60s"}
        assert set(stats.keys()) == expected_keys
        for key in expected_keys:
            assert "hits" in stats[key]
            assert "misses" in stats[key]

    def test_clear_all_caches(self):
        """clear_all_caches resets all pre-configured cache instances."""
        clear_all_caches()
        stats = get_cache_stats()
        for key, info in stats.items():
            assert info["hits"] == 0
            assert info["misses"] == 0
            assert info["size"] == 0
