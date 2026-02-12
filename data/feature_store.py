"""
data/feature_store.py - Feature Store with Redis Caching

Centralized feature computation and caching to:
- Avoid redundant calculations across multiple models
- Share features between signal generators
- Persist computed features across restarts
- Enable feature versioning and lineage tracking

Supports:
- Redis for distributed caching
- SQLite for persistent storage
- In-memory fallback

Usage:
    store = FeatureStore(redis_url="redis://localhost:6379")
    features = store.get_features("AAPL", prices, feature_set="momentum")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import hmac
import json
import logging
import os
import pickle
from pathlib import Path
import threading
from abc import ABC, abstractmethod

from config import ApexConfig

logger = logging.getLogger(__name__)

# Check for Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("redis not installed. Using in-memory cache. Install with: pip install redis")


@dataclass
class FeatureMetadata:
    """Metadata for a computed feature set."""
    symbol: str
    feature_set: str
    version: str
    computed_at: datetime
    data_start: datetime
    data_end: datetime
    n_samples: int
    feature_names: List[str]
    computation_time_ms: float

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'feature_set': self.feature_set,
            'version': self.version,
            'computed_at': self.computed_at.isoformat(),
            'data_start': self.data_start.isoformat(),
            'data_end': self.data_end.isoformat(),
            'n_samples': self.n_samples,
            'feature_names': self.feature_names,
            'computation_time_ms': self.computation_time_ms
        }


class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int = None) -> bool:
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear_pattern(self, pattern: str) -> int:
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend."""

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            value, expiry = self._cache[key]
            if expiry and datetime.now() > expiry:
                del self._cache[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: int = None) -> bool:
        with self._lock:
            expiry = datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
            self._cache[key] = (value, expiry)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear_pattern(self, pattern: str) -> int:
        with self._lock:
            # Simple pattern matching (just prefix)
            prefix = pattern.rstrip('*')
            keys_to_delete = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._cache[k]
            return len(keys_to_delete)


class RedisCache(CacheBackend):
    """Redis cache backend."""

    def __init__(self, url: str = "redis://localhost:6379", db: int = 0):
        self.client = redis.from_url(url, db=db, decode_responses=False)
        self._prefix = "apex:features:"
        signing_key = os.getenv("APEX_FEATURE_CACHE_SIGNING_KEY") or os.getenv("APEX_SECRET_KEY")
        if not signing_key:
            raise ValueError(
                "APEX_SECRET_KEY (or APEX_FEATURE_CACHE_SIGNING_KEY) must be set for secure Redis cache signing"
            )
        self._signing_key = signing_key.encode()

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(self._key(key))
            if data:
                parts = data.split(b":", 2)
                if len(parts) != 3 or parts[0] != b"v1":
                    logger.warning("Ignoring unsigned/legacy Redis cache payload for key=%s", key)
                    return None
                sig_hex, payload = parts[1], parts[2]
                expected_sig = hmac.new(self._signing_key, payload, hashlib.sha256).hexdigest().encode()
                if not hmac.compare_digest(sig_hex, expected_sig):
                    logger.warning("Ignoring tampered Redis cache payload for key=%s", key)
                    return None
                return pickle.loads(payload)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = None) -> bool:
        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            sig_hex = hmac.new(self._signing_key, payload, hashlib.sha256).hexdigest().encode()
            data = b":".join((b"v1", sig_hex, payload))
            if ttl_seconds:
                self.client.setex(self._key(key), ttl_seconds, data)
            else:
                self.client.set(self._key(key), data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            return bool(self.client.delete(self._key(key)))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(self._key(key)))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        try:
            keys = self.client.keys(self._key(pattern))
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear_pattern error: {e}")
            return 0


class FeatureComputer:
    """Computes features for a given feature set."""

    # Feature set definitions
    FEATURE_SETS = {
        'momentum': [
            'ret_5', 'ret_10', 'ret_20', 'ret_60',
            'roc_5', 'roc_10', 'roc_20',
            'mom_accel'
        ],
        'mean_reversion': [
            'zscore_20', 'zscore_50', 'zscore_200',
            'bb_position', 'bb_width',
            'rsi_14', 'rsi_7'
        ],
        'trend': [
            'ma_cross_10_20', 'ma_cross_20_50', 'ma_cross_50_200',
            'trend_strength_20', 'trend_strength_50',
            'adx_14', 'price_vs_ma50', 'price_vs_ma200'
        ],
        'volatility': [
            'vol_5', 'vol_20', 'vol_60',
            'vol_ratio_5_20', 'vol_ratio_20_60',
            'atr_14', 'atr_ratio'
        ],
        'all': None  # Computed dynamically
    }

    @staticmethod
    def compute(prices: pd.Series, feature_set: str = 'all') -> Tuple[np.ndarray, List[str]]:
        """
        Compute features for a price series.

        Args:
            prices: Price series
            feature_set: Which feature set to compute

        Returns:
            Tuple of (feature_array, feature_names)
        """
        if len(prices) < 200:
            return np.array([]), []

        features = {}
        returns = prices.pct_change()

        # Momentum features
        if feature_set in ['momentum', 'all']:
            features['ret_5'] = (prices.iloc[-1] / prices.iloc[-5] - 1)
            features['ret_10'] = (prices.iloc[-1] / prices.iloc[-10] - 1)
            features['ret_20'] = (prices.iloc[-1] / prices.iloc[-20] - 1)
            features['ret_60'] = (prices.iloc[-1] / prices.iloc[-60] - 1)
            features['roc_5'] = returns.iloc[-5:].mean()
            features['roc_10'] = returns.iloc[-10:].mean()
            features['roc_20'] = returns.iloc[-20:].mean()
            features['mom_accel'] = features['roc_5'] - features['roc_10'] / 2

        # Mean reversion features
        if feature_set in ['mean_reversion', 'all']:
            for period in [20, 50, 200]:
                mean = prices.iloc[-period:].mean()
                std = prices.iloc[-period:].std()
                features[f'zscore_{period}'] = (prices.iloc[-1] - mean) / std if std > 0 else 0

            # Bollinger Bands
            ma_20 = prices.rolling(20).mean().iloc[-1]
            std_20 = prices.rolling(20).std().iloc[-1]
            bb_upper = ma_20 + 2 * std_20
            bb_lower = ma_20 - 2 * std_20
            features['bb_position'] = (prices.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            features['bb_width'] = (bb_upper - bb_lower) / ma_20 if ma_20 > 0 else 0

            # RSI
            for period in [7, 14]:
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean().iloc[-1]
                rs = gain / loss if loss != 0 else 100
                features[f'rsi_{period}'] = (100 - (100 / (1 + rs)) - 50) / 50  # Normalized

        # Trend features
        if feature_set in ['trend', 'all']:
            ma_10 = prices.rolling(10).mean().iloc[-1]
            ma_20 = prices.rolling(20).mean().iloc[-1]
            ma_50 = prices.rolling(50).mean().iloc[-1]
            ma_200 = prices.rolling(200).mean().iloc[-1]

            features['ma_cross_10_20'] = 1 if ma_10 > ma_20 else -1
            features['ma_cross_20_50'] = 1 if ma_20 > ma_50 else -1
            features['ma_cross_50_200'] = 1 if ma_50 > ma_200 else -1
            features['price_vs_ma50'] = (prices.iloc[-1] - ma_50) / ma_50 if ma_50 > 0 else 0
            features['price_vs_ma200'] = (prices.iloc[-1] - ma_200) / ma_200 if ma_200 > 0 else 0

            # Trend strength (R-squared of linear fit)
            for period in [20, 50]:
                y = prices.iloc[-period:].values
                x = np.arange(period)
                coeffs = np.polyfit(x, y, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                features[f'trend_strength_{period}'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # ADX approximation
            pos_moves = returns.where(returns > 0, 0).rolling(14).mean().iloc[-1]
            neg_moves = (-returns.where(returns < 0, 0)).rolling(14).mean().iloc[-1]
            total = pos_moves + neg_moves
            features['adx_14'] = abs(pos_moves - neg_moves) / total if total > 0 else 0

        # Volatility features
        if feature_set in ['volatility', 'all']:
            for period in [5, 20, 60]:
                features[f'vol_{period}'] = returns.iloc[-period:].std() * np.sqrt(252)

            features['vol_ratio_5_20'] = features.get('vol_5', 0) / features.get('vol_20', 1) if features.get('vol_20', 1) > 0 else 1
            features['vol_ratio_20_60'] = features.get('vol_20', 0) / features.get('vol_60', 1) if features.get('vol_60', 1) > 0 else 1

            # ATR
            high = prices  # Simplified - would use actual high/low if available
            low = prices
            close = prices
            tr = pd.concat([
                high - low,
                abs(high - close.shift()),
                abs(low - close.shift())
            ], axis=1).max(axis=1)
            features['atr_14'] = tr.rolling(14).mean().iloc[-1] / close.iloc[-1]
            features['atr_ratio'] = tr.iloc[-1] / tr.rolling(14).mean().iloc[-1] if tr.rolling(14).mean().iloc[-1] > 0 else 1

        # Convert to array
        feature_names = list(features.keys())
        feature_array = np.array([features[f] for f in feature_names])

        # Handle NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)

        return feature_array, feature_names


class FeatureStore:
    """
    Centralized feature store with caching.

    Features:
    - Multi-backend caching (Redis, Memory, SQLite)
    - Feature versioning
    - Automatic cache invalidation
    - Batch feature computation
    - Feature lineage tracking
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        redis_url: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
        data_dir: Path = None
    ):
        """
        Initialize feature store.

        Args:
            redis_url: Redis URL (uses memory cache if None)
            cache_ttl_seconds: Default TTL for cached features
            data_dir: Directory for persistent storage
        """
        self.cache_ttl = cache_ttl_seconds
        self.data_dir = data_dir or (ApexConfig.DATA_DIR / "features")

        # Initialize cache backend
        if redis_url and REDIS_AVAILABLE:
            try:
                self.cache = RedisCache(redis_url)
                self.cache_type = "redis"
                logger.info(f"📦 Feature Store initialized with Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using memory cache.")
                self.cache = MemoryCache()
                self.cache_type = "memory"
        else:
            self.cache = MemoryCache()
            self.cache_type = "memory"
            logger.info("📦 Feature Store initialized with in-memory cache")

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'computations': 0,
            'computation_time_ms': 0
        }

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, symbol: str, feature_set: str, data_hash: str) -> str:
        """Generate cache key."""
        return f"{symbol}:{feature_set}:{self.VERSION}:{data_hash}"

    def _data_hash(self, prices: pd.Series) -> str:
        """Generate hash of price data for cache key."""
        # Use last price and length as quick hash
        if len(prices) == 0:
            return "empty"
        last_price = prices.iloc[-1]
        first_price = prices.iloc[0]
        data_str = f"{len(prices)}:{first_price:.4f}:{last_price:.4f}"
        return hashlib.md5(data_str.encode()).hexdigest()[:12]

    def get_features(
        self,
        symbol: str,
        prices: pd.Series,
        feature_set: str = 'all',
        force_compute: bool = False
    ) -> Tuple[np.ndarray, List[str], FeatureMetadata]:
        """
        Get features for a symbol, using cache if available.

        Args:
            symbol: Stock ticker
            prices: Price series
            feature_set: Feature set to compute
            force_compute: Bypass cache and recompute

        Returns:
            Tuple of (features, feature_names, metadata)
        """
        data_hash = self._data_hash(prices)
        cache_key = self._cache_key(symbol, feature_set, data_hash)

        # Try cache first
        if not force_compute:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.stats['hits'] += 1
                logger.debug(f"Cache hit for {symbol}:{feature_set}")
                return cached['features'], cached['names'], cached['metadata']

        # Cache miss - compute features
        self.stats['misses'] += 1
        start_time = datetime.now()

        features, names = FeatureComputer.compute(prices, feature_set)

        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        self.stats['computations'] += 1
        self.stats['computation_time_ms'] += computation_time

        # Create metadata
        metadata = FeatureMetadata(
            symbol=symbol,
            feature_set=feature_set,
            version=self.VERSION,
            computed_at=datetime.now(),
            data_start=prices.index[0] if hasattr(prices.index[0], 'isoformat') else datetime.now(),
            data_end=prices.index[-1] if hasattr(prices.index[-1], 'isoformat') else datetime.now(),
            n_samples=len(prices),
            feature_names=names,
            computation_time_ms=computation_time
        )

        # Cache the result
        cache_data = {
            'features': features,
            'names': names,
            'metadata': metadata
        }
        self.cache.set(cache_key, cache_data, self.cache_ttl)

        logger.debug(f"Computed {len(names)} features for {symbol} in {computation_time:.1f}ms")

        return features, names, metadata

    def get_features_batch(
        self,
        symbols: List[str],
        prices_dict: Dict[str, pd.Series],
        feature_set: str = 'all'
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """
        Get features for multiple symbols.

        Args:
            symbols: List of symbols
            prices_dict: Dict of {symbol: prices}
            feature_set: Feature set to compute

        Returns:
            Dict of {symbol: (features, names)}
        """
        results = {}
        for symbol in symbols:
            if symbol in prices_dict and len(prices_dict[symbol]) >= 200:
                features, names, _ = self.get_features(
                    symbol, prices_dict[symbol], feature_set
                )
                results[symbol] = (features, names)
        return results

    def invalidate(self, symbol: str, feature_set: str = None):
        """
        Invalidate cached features for a symbol.

        Args:
            symbol: Symbol to invalidate
            feature_set: Specific feature set (all if None)
        """
        if feature_set:
            pattern = f"{symbol}:{feature_set}:*"
        else:
            pattern = f"{symbol}:*"

        count = self.cache.clear_pattern(pattern)
        logger.info(f"Invalidated {count} cache entries for {symbol}")

    def invalidate_all(self):
        """Invalidate all cached features."""
        count = self.cache.clear_pattern("*")
        logger.info(f"Invalidated {count} total cache entries")
        self.stats = {'hits': 0, 'misses': 0, 'computations': 0, 'computation_time_ms': 0}

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0

        return {
            'cache_type': self.cache_type,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'computations': self.stats['computations'],
            'avg_computation_ms': (
                self.stats['computation_time_ms'] / self.stats['computations']
                if self.stats['computations'] > 0 else 0
            ),
            'version': self.VERSION
        }

    def save_features_to_disk(
        self,
        symbol: str,
        features: np.ndarray,
        names: List[str],
        metadata: FeatureMetadata
    ):
        """Save computed features to disk for persistence."""
        try:
            filepath = self.data_dir / f"{symbol}_{metadata.feature_set}.npz"
            np.savez(
                filepath,
                features=features,
                names=names,
                metadata=json.dumps(metadata.to_dict())
            )
            logger.debug(f"Saved features to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save features: {e}")

    def load_features_from_disk(
        self,
        symbol: str,
        feature_set: str
    ) -> Optional[Tuple[np.ndarray, List[str], FeatureMetadata]]:
        """Load features from disk."""
        try:
            filepath = self.data_dir / f"{symbol}_{feature_set}.npz"
            if not filepath.exists():
                return None

            data = np.load(filepath, allow_pickle=False)
            metadata_dict = json.loads(str(data['metadata']))

            metadata = FeatureMetadata(
                symbol=metadata_dict['symbol'],
                feature_set=metadata_dict['feature_set'],
                version=metadata_dict['version'],
                computed_at=datetime.fromisoformat(metadata_dict['computed_at']),
                data_start=datetime.fromisoformat(metadata_dict['data_start']),
                data_end=datetime.fromisoformat(metadata_dict['data_end']),
                n_samples=metadata_dict['n_samples'],
                feature_names=metadata_dict['feature_names'],
                computation_time_ms=metadata_dict['computation_time_ms']
            )

            return data['features'], list(data['names']), metadata

        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return None
