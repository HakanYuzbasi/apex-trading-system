"""
Tests for SignalOptimizer caching, eviction, and metrics.
"""
import time
import pandas as pd
import numpy as np

from models.signal_optimizer import (
    SignalOptimizer,
    OptimizationConfig,
    SignalMetrics,
    get_global_optimizer,
)


def _make_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Close": 100 + np.cumsum(rng.normal(0, 1, n)),
        "Volume": rng.integers(100_000, 1_000_000, n),
    })


class _DummyGenerator:
    """Minimal generator stub for testing."""
    def generate_signal(self, data: pd.DataFrame, symbol: str):  # noqa: ARG002
        return {"signal": float(data["Close"].pct_change().mean()), "confidence": 0.60}


class TestOptimizationConfig:
    def test_defaults(self):
        cfg = OptimizationConfig()
        assert cfg.enable_caching is True
        assert cfg.cache_ttl_seconds == 300
        assert cfg.max_workers == 4

    def test_custom(self):
        cfg = OptimizationConfig(cache_ttl_seconds=60, top_k_features=20)
        assert cfg.cache_ttl_seconds == 60
        assert cfg.top_k_features == 20


class TestSignalOptimizerCacheKey:
    def setup_method(self):
        self.opt = SignalOptimizer(OptimizationConfig())
        self.df = _make_df()

    def test_same_data_same_key(self):
        k1 = self.opt._get_cache_key(self.df, "AAPL", {})
        k2 = self.opt._get_cache_key(self.df, "AAPL", {})
        assert k1 == k2

    def test_different_symbol_different_key(self):
        k1 = self.opt._get_cache_key(self.df, "AAPL", {})
        k2 = self.opt._get_cache_key(self.df, "MSFT", {})
        assert k1 != k2

    def test_different_kwargs_different_key(self):
        k1 = self.opt._get_cache_key(self.df, "AAPL", {"a": 1})
        k2 = self.opt._get_cache_key(self.df, "AAPL", {"a": 2})
        assert k1 != k2

    def test_empty_df_does_not_crash(self):
        empty = pd.DataFrame()
        k = self.opt._get_cache_key(empty, "AAPL", {})
        assert isinstance(k, str)


class TestSignalOptimizerCacheTTL:
    def test_cache_miss_on_empty(self):
        opt = SignalOptimizer(OptimizationConfig())
        assert opt._get_cached_signal("nonexistent_key") is None

    def test_cache_hit_within_ttl(self):
        opt = SignalOptimizer(OptimizationConfig(cache_ttl_seconds=60))
        obj = {"signal": 0.5}
        opt._cache_signal("k1", obj)
        assert opt._get_cached_signal("k1") is obj

    def test_cache_expired_returns_none(self):
        opt = SignalOptimizer(OptimizationConfig(cache_ttl_seconds=0))
        opt._cache_signal("k2", {"signal": 0.3})
        time.sleep(0.01)
        assert opt._get_cached_signal("k2") is None

    def test_hit_rate_tracks_correctly(self):
        opt = SignalOptimizer(OptimizationConfig(cache_ttl_seconds=60))
        opt._cache_signal("k3", {"signal": 0.1})
        opt._get_cached_signal("k3")   # hit
        opt._get_cached_signal("miss") # miss
        rate = opt._get_cache_hit_rate()
        assert 0.0 <= rate <= 1.0
        assert opt.cache_hits == 1
        assert opt.cache_misses == 1  # "miss" key only

    def test_caching_disabled_always_miss(self):
        opt = SignalOptimizer(OptimizationConfig(enable_caching=False))
        opt._cache_signal("k4", {"x": 1})  # should be no-op
        assert opt._get_cached_signal("k4") is None

    def test_cache_eviction_above_1000_entries(self):
        opt = SignalOptimizer(OptimizationConfig(cache_ttl_seconds=3600))
        # Need >1000 to trigger eviction (condition: len > 1000)
        for i in range(1002):
            opt._cache_signal(f"k{i}", {"i": i})
        assert len(opt.signal_cache) <= 800  # eviction removed 25% of 1001


class TestOptimizeSigGeneration:
    def setup_method(self):
        self.opt = SignalOptimizer(OptimizationConfig(cache_ttl_seconds=60))
        self.gen = _DummyGenerator()
        self.df = _make_df()

    def test_returns_signal_and_metrics(self):
        signal, metrics = self.opt.optimize_signal_generation(self.df, "AAPL", self.gen)
        assert signal is not None
        assert isinstance(metrics, SignalMetrics)

    def test_second_call_hits_cache(self):
        self.opt.optimize_signal_generation(self.df, "AAPL", self.gen)
        _, metrics2 = self.opt.optimize_signal_generation(self.df, "AAPL", self.gen)
        assert self.opt.cache_hits >= 1

    def test_metrics_has_expected_fields(self):
        _, m = self.opt.optimize_signal_generation(self.df, "TSLA", self.gen)
        assert hasattr(m, "generation_time_ms")
        assert hasattr(m, "cache_hit_rate")
        assert m.generation_time_ms >= 0.0


class TestFeatureImportanceUpdate:
    def test_update_and_retrieve(self):
        opt = SignalOptimizer(OptimizationConfig())
        opt.update_feature_importance("AAPL", {"Close": 0.8, "Volume": 0.2})
        assert "AAPL" in opt.feature_importance
        assert "Close" in opt.feature_importance["AAPL"]

    def test_ema_update(self):
        opt = SignalOptimizer(OptimizationConfig())
        opt.update_feature_importance("AAPL", {"x": 1.0})
        opt.update_feature_importance("AAPL", {"x": 0.0})
        val = opt.feature_importance["AAPL"]["x"]
        assert 0.0 < val < 1.0  # blended via EMA


class TestClearCache:
    def test_clear_resets_counters(self):
        opt = SignalOptimizer(OptimizationConfig(cache_ttl_seconds=60))
        opt._cache_signal("k", {"x": 1})
        opt._get_cached_signal("k")
        opt.clear_cache()
        assert opt.cache_hits == 0
        assert opt.cache_misses == 0
        assert len(opt.signal_cache) == 0


class TestGlobalOptimizer:
    def test_singleton(self):
        g1 = get_global_optimizer()
        g2 = get_global_optimizer()
        assert g1 is g2

    def test_is_signal_optimizer_instance(self):
        assert isinstance(get_global_optimizer(), SignalOptimizer)
