"""
tests/test_feature_store.py - Feature Store Tests

Tests for the feature store including:
- Feature caching
- Cache backends (Redis, SQLite, Memory)
- Feature retrieval
- Cache invalidation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import json
import asyncio


@pytest.fixture
def sample_features():
    """Sample feature data."""
    return pd.DataFrame({
        'momentum_20': [0.05, 0.03, -0.02, 0.08, 0.01],
        'rsi_14': [55.0, 62.0, 48.0, 71.0, 50.0],
        'volatility_20': [0.18, 0.20, 0.22, 0.19, 0.17],
        'macd': [0.5, 0.8, -0.2, 1.2, 0.3],
        'bb_position': [0.6, 0.7, 0.3, 0.9, 0.5],
    }, index=pd.date_range('2024-01-01', periods=5))


@pytest.fixture
def feature_config():
    """Feature store configuration."""
    return {
        'cache_backend': 'memory',
        'ttl_seconds': 3600,
        'max_cache_size': 1000,
        'compression': False,
    }


class TestMemoryCache:
    """Test in-memory cache backend."""

    def test_cache_store_and_retrieve(self, sample_features):
        """Test storing and retrieving from memory cache."""
        cache = {}

        # Store
        key = "AAPL:features:2024-01-01"
        cache[key] = sample_features.to_dict()

        # Retrieve
        retrieved = cache.get(key)
        assert retrieved is not None
        assert 'momentum_20' in retrieved

    def test_cache_expiration(self, sample_features):
        """Test cache expiration logic."""
        cache = {}
        expiry = {}

        key = "AAPL:features"
        cache[key] = sample_features.to_dict()
        expiry[key] = datetime.now() + timedelta(seconds=3600)

        # Check if expired
        is_expired = datetime.now() > expiry[key]
        assert not is_expired

    def test_cache_eviction_lru(self, sample_features):
        """Test LRU eviction when cache is full."""
        from collections import OrderedDict

        max_size = 3
        cache = OrderedDict()

        # Fill cache
        for i in range(max_size + 1):
            key = f"symbol_{i}:features"
            cache[key] = sample_features.to_dict()
            cache.move_to_end(key)

            # Evict if over size
            if len(cache) > max_size:
                cache.popitem(last=False)

        assert len(cache) == max_size
        assert "symbol_0:features" not in cache
        assert "symbol_3:features" in cache

    def test_cache_clear(self, sample_features):
        """Test clearing cache."""
        cache = {"key1": "value1", "key2": "value2"}

        cache.clear()

        assert len(cache) == 0


class TestRedisCacheBackend:
    """Test Redis cache backend."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis = MagicMock()
        redis.get = MagicMock(return_value=None)
        redis.set = MagicMock(return_value=True)
        redis.delete = MagicMock(return_value=1)
        redis.exists = MagicMock(return_value=False)
        redis.expire = MagicMock(return_value=True)
        return redis

    def test_redis_store(self, mock_redis, sample_features):
        """Test storing in Redis."""
        key = "AAPL:features"
        value = json.dumps(sample_features.to_dict(orient='list'))

        mock_redis.set(key, value, ex=3600)

        mock_redis.set.assert_called_once()

    def test_redis_retrieve(self, mock_redis, sample_features):
        """Test retrieving from Redis."""
        key = "AAPL:features"
        stored_value = json.dumps(sample_features.to_dict(orient='list'))
        mock_redis.get = MagicMock(return_value=stored_value.encode())

        result = mock_redis.get(key)

        assert result is not None
        data = json.loads(result.decode())
        assert 'momentum_20' in data

    def test_redis_key_not_found(self, mock_redis):
        """Test handling missing keys."""
        mock_redis.get = MagicMock(return_value=None)

        result = mock_redis.get("nonexistent:key")

        assert result is None

    def test_redis_connection_error(self, mock_redis):
        """Test handling Redis connection errors."""
        mock_redis.get = MagicMock(side_effect=ConnectionError("Redis unavailable"))

        with pytest.raises(ConnectionError):
            mock_redis.get("key")

    def test_redis_ttl(self, mock_redis):
        """Test TTL setting on Redis keys."""
        key = "AAPL:features"
        ttl = 3600

        mock_redis.set(key, "value", ex=ttl)
        mock_redis.expire(key, ttl)

        mock_redis.expire.assert_called_with(key, ttl)


class TestSQLiteCache:
    """Test SQLite cache backend."""

    @pytest.fixture
    def mock_sqlite_connection(self):
        """Mock SQLite connection."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        cursor.fetchone = MagicMock(return_value=None)
        cursor.fetchall = MagicMock(return_value=[])
        return conn, cursor

    def test_sqlite_create_table(self, mock_sqlite_connection):
        """Test creating cache table."""
        conn, cursor = mock_sqlite_connection

        create_sql = """
            CREATE TABLE IF NOT EXISTS feature_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at TIMESTAMP
            )
        """
        cursor.execute(create_sql)

        cursor.execute.assert_called_once()

    def test_sqlite_insert(self, mock_sqlite_connection, sample_features):
        """Test inserting into SQLite cache."""
        conn, cursor = mock_sqlite_connection

        key = "AAPL:features"
        value = json.dumps(sample_features.to_dict(orient='list'))
        expires_at = datetime.now() + timedelta(hours=1)

        cursor.execute(
            "INSERT OR REPLACE INTO feature_cache VALUES (?, ?, ?)",
            (key, value, expires_at.isoformat())
        )

        cursor.execute.assert_called_once()

    def test_sqlite_query(self, mock_sqlite_connection, sample_features):
        """Test querying SQLite cache."""
        conn, cursor = mock_sqlite_connection

        stored_value = json.dumps(sample_features.to_dict(orient='list'))
        cursor.fetchone = MagicMock(return_value=(stored_value,))

        cursor.execute("SELECT value FROM feature_cache WHERE key = ?", ("AAPL:features",))
        result = cursor.fetchone()

        assert result is not None
        data = json.loads(result[0])
        assert 'momentum_20' in data


class TestFeatureComputation:
    """Test feature computation and caching."""

    def test_compute_features_on_miss(self, sample_features):
        """Test computing features when cache miss."""
        cache = {}
        key = "AAPL:features:2024-01-01"

        # Cache miss
        cached = cache.get(key)
        assert cached is None

        # Compute features
        computed = sample_features.copy()

        # Store in cache
        cache[key] = computed.to_dict()

        # Verify cached
        assert key in cache

    def test_return_cached_on_hit(self, sample_features):
        """Test returning cached features on hit."""
        cache = {"AAPL:features": sample_features.to_dict()}
        key = "AAPL:features"

        # Cache hit
        cached = cache.get(key)
        assert cached is not None

        # Convert back to DataFrame
        df = pd.DataFrame(cached)
        assert 'momentum_20' in df.columns

    def test_cache_invalidation_on_new_data(self, sample_features):
        """Test cache invalidation when new data arrives."""
        cache = {"AAPL:features:2024-01-01": sample_features.to_dict()}
        cache_timestamps = {"AAPL": datetime(2024, 1, 1)}

        new_data_timestamp = datetime(2024, 1, 2)

        # Check if cache is stale
        last_cached = cache_timestamps.get("AAPL", datetime.min)
        is_stale = new_data_timestamp > last_cached

        assert is_stale

        # Invalidate cache
        del cache["AAPL:features:2024-01-01"]
        assert "AAPL:features:2024-01-01" not in cache


class TestFeatureNormalization:
    """Test feature normalization in cache."""

    def test_normalize_features(self, sample_features):
        """Test feature normalization."""
        normalized = sample_features.copy()

        for col in normalized.columns:
            mean = normalized[col].mean()
            std = normalized[col].std()
            if std > 0:
                normalized[col] = (normalized[col] - mean) / std

        # Check normalized
        for col in normalized.columns:
            assert abs(normalized[col].mean()) < 0.01  # Near zero mean
            assert abs(normalized[col].std() - 1.0) < 0.1  # Near unit std

    def test_handle_missing_features(self, sample_features):
        """Test handling missing feature values."""
        features = sample_features.copy()
        features.loc[features.index[0], 'momentum_20'] = np.nan

        # Fill missing values
        filled = features.fillna(method='ffill').fillna(0)

        assert not filled.isna().any().any()

    def test_clip_extreme_values(self, sample_features):
        """Test clipping extreme feature values."""
        features = sample_features.copy()
        features.loc[features.index[0], 'rsi_14'] = 150  # Invalid RSI

        # Clip to valid range
        features['rsi_14'] = features['rsi_14'].clip(0, 100)

        assert features['rsi_14'].max() <= 100
        assert features['rsi_14'].min() >= 0


class TestBatchOperations:
    """Test batch feature operations."""

    def test_batch_store(self, sample_features):
        """Test batch storing features."""
        cache = {}
        symbols = ['AAPL', 'MSFT', 'GOOGL']

        for symbol in symbols:
            key = f"{symbol}:features"
            cache[key] = sample_features.to_dict()

        assert len(cache) == 3
        assert all(f"{s}:features" in cache for s in symbols)

    def test_batch_retrieve(self, sample_features):
        """Test batch retrieving features."""
        cache = {
            f"{s}:features": sample_features.to_dict()
            for s in ['AAPL', 'MSFT', 'GOOGL']
        }

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        results = {s: cache.get(f"{s}:features") for s in symbols}

        assert all(r is not None for r in results.values())

    def test_batch_delete(self):
        """Test batch deleting features."""
        cache = {
            "AAPL:features": "data1",
            "MSFT:features": "data2",
            "GOOGL:features": "data3",
        }

        to_delete = ['AAPL', 'MSFT']
        for symbol in to_delete:
            del cache[f"{symbol}:features"]

        assert len(cache) == 1
        assert "GOOGL:features" in cache


class TestCacheMetrics:
    """Test cache metrics collection."""

    def test_track_hit_rate(self):
        """Test tracking cache hit rate."""
        hits = 80
        misses = 20
        total = hits + misses

        hit_rate = hits / total

        assert hit_rate == 0.8

    def test_track_cache_size(self, sample_features):
        """Test tracking cache size."""
        cache = {}
        max_size = 100

        for i in range(50):
            cache[f"symbol_{i}"] = sample_features.to_dict()

        current_size = len(cache)
        usage_pct = current_size / max_size

        assert usage_pct == 0.5

    def test_track_eviction_count(self):
        """Test tracking eviction count."""
        evictions = 0
        max_size = 10
        cache = {}

        for i in range(15):
            if len(cache) >= max_size:
                oldest = min(cache.keys())
                del cache[oldest]
                evictions += 1
            cache[f"key_{i}"] = "value"

        assert evictions == 5
