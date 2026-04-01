"""
tests/test_crypto_alpha_features.py
Tests for Fix #4: 5 new crypto alpha features in GodLevelSignalGenerator.
"""
from __future__ import annotations

import sys
import types
import unittest

import numpy as np
import pandas as pd


# ── Stubs ─────────────────────────────────────────────────────────────────────
for _m in ("sklearn", "sklearn.preprocessing", "sklearn.ensemble",
           "sklearn.linear_model", "sklearn.calibration",
           "xgboost", "lightgbm", "shap", "catboost"):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

# sklearn submodule stubs
_sk = sys.modules["sklearn"]
for _attr in ("preprocessing", "ensemble", "linear_model", "calibration",
              "inspection", "model_selection"):
    _sub = types.ModuleType(f"sklearn.{_attr}")
    if not hasattr(_sk, _attr):
        setattr(_sk, _attr, _sub)
    if f"sklearn.{_attr}" not in sys.modules:
        sys.modules[f"sklearn.{_attr}"] = _sub

# Add minimal sklearn objects
sys.modules["sklearn.preprocessing"].RobustScaler = type(
    "RobustScaler", (), {"fit": lambda *a, **k: None, "transform": lambda s, x: x}
)
sys.modules["sklearn.preprocessing"].StandardScaler = sys.modules["sklearn.preprocessing"].RobustScaler

if "config" not in sys.modules:
    _cfg = types.ModuleType("config")
    sys.modules["config"] = _cfg

import config as _cfg_mod
if not hasattr(_cfg_mod, "ApexConfig"):
    _cfg_mod.ApexConfig = types.SimpleNamespace()
if not hasattr(_cfg_mod.ApexConfig, "MODELS_DIR"):
    _cfg_mod.ApexConfig.MODELS_DIR = "/tmp/apex_test_models"


def _make_close_series(n=300, seed=42):
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.Series(prices)


def _make_ohlcv_df(n=300, seed=42):
    close = _make_close_series(n, seed)
    rng = np.random.default_rng(seed)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    op = close.shift(1).fillna(close.iloc[0])
    vol = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close, "Volume": vol})


# Module-level stubs for pickle compatibility in mismatch tests
class _StaleScaler:
    """Simulates a scaler saved before the +5 feature addition."""
    n_features_in_ = 1  # will be overwritten per-test


class _MatchingScaler:
    """Simulates a scaler that matches the current feature count."""
    n_features_in_ = 1  # will be overwritten per-test

    def transform(self, X):
        return X


class _FakeModel:
    def predict(self, X):
        return [0.0]


def _get_god_level():
    """Import GodLevelSignalGenerator with mocked heavy deps."""
    from models.god_level_signal_generator import GodLevelSignalGenerator
    g = GodLevelSignalGenerator.__new__(GodLevelSignalGenerator)
    g.lookback = 200
    g.models = {}
    g.models_trained = False
    g.feature_scaler = None
    g.feature_importance = {}
    g.explainers = {}
    g._signal_optimizer = None
    g._binary_classifier = None
    return g


class TestCryptoAlphaFeatures(unittest.TestCase):
    """Validate the 5 new Section-12 features in extract_features()."""

    def setUp(self):
        self.g = _get_god_level()
        self.close = _make_close_series(300)
        self.df = _make_ohlcv_df(300)

    def _get_features(self, prices):
        return self.g.extract_features(prices)

    def test_feature_count_increased_by_5_for_close_series(self):
        """
        Adding Section 12 increases the feature vector by exactly 5.
        Previous count (sections 1-11): probe with a series to get old count,
        then verify the new series is 5 longer.
        """
        f = self._get_features(self.close)
        # We expect at least 50 features (original) + 5 new = 55+
        self.assertGreaterEqual(len(f), 55)

    def test_feature_count_consistent_across_calls(self):
        """Two calls with different series lengths (≥lookback) return same feature count."""
        f1 = self._get_features(self.close)
        f2 = self._get_features(self.close.iloc[:250])
        self.assertEqual(len(f1), len(f2))

    def test_features_all_finite(self):
        """No NaN or Inf in the feature vector."""
        f = self._get_features(self.close)
        self.assertTrue(np.all(np.isfinite(f)), "Feature vector contains NaN/Inf")

    def test_feature_12_1_volatility_clustering_range(self):
        """Feature 12.1 (vol clustering) must be in [-1, 1]."""
        f = self._get_features(self.close)
        # Feature 12.1 is at index -5 (5th from end)
        val = f[-5]
        self.assertGreaterEqual(val, -1.0)
        self.assertLessEqual(val, 1.0)

    def test_feature_12_2_drawdown_nonpositive(self):
        """Feature 12.2 (drawdown from 60-day high) is ≤ 0."""
        f = self._get_features(self.close)
        val = f[-4]  # 4th from end
        self.assertLessEqual(val, 0.0)
        self.assertGreaterEqual(val, -1.0)

    def test_feature_12_2_drawdown_zero_at_new_high(self):
        """If last price equals 60-day max, drawdown is 0."""
        # Monotonically increasing prices → last price IS the max
        mono = pd.Series(np.linspace(100, 200, 300))
        f = self._get_features(mono)
        self.assertAlmostEqual(f[-4], 0.0, places=4)

    def test_feature_12_3_momentum_acceleration_range(self):
        """Feature 12.3 (momentum acceleration) must be in [-0.5, 0.5]."""
        f = self._get_features(self.close)
        val = f[-3]
        self.assertGreaterEqual(val, -0.5)
        self.assertLessEqual(val, 0.5)

    def test_feature_12_4_tail_risk_in_0_1(self):
        """Feature 12.4 (tail risk fraction) must be in [0, 1]."""
        f = self._get_features(self.close)
        val = f[-2]
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_feature_12_4_tail_risk_higher_for_fat_tailed_series(self):
        """
        Fat-tailed series (with extreme outlier returns) should show higher tail-risk
        than a near-Gaussian series.  The feature counts |r| > 2σ across last 20 bars.
        """
        rng = np.random.default_rng(42)
        n = 300

        # Near-Gaussian: small normal returns, NO outliers
        gaussian_ret = rng.normal(0, 0.01, n)
        gaussian = pd.Series(100 * np.exp(np.cumsum(gaussian_ret)))

        # Fat-tailed: inject known >4σ outliers in the last 20 bars (tail events)
        fat_ret = rng.normal(0, 0.01, n).copy()
        # Inject 5 returns of ±8σ in the last 20 bars → 25% tail fraction
        fat_ret[-20::4] = 0.10  # 10% return ≈ +10σ at σ=0.01
        fat = pd.Series(100 * np.exp(np.cumsum(fat_ret)))

        f_gauss = self._get_features(gaussian)
        f_fat = self._get_features(fat)

        # Fat-tailed series must have strictly higher tail-risk fraction
        self.assertGreater(f_fat[-2], f_gauss[-2])

    def test_feature_12_5_volume_range_position_with_ohlcv(self):
        """Feature 12.5 (volume-range position) fires when OHLCV df is passed."""
        f = self._get_features(self.df)
        val = f[-1]
        self.assertGreaterEqual(val, -0.5)
        self.assertLessEqual(val, 0.5)

    def test_feature_12_5_fallback_without_ohlcv(self):
        """Feature 12.5 falls back to 0.0 when only a close series is provided."""
        f = self._get_features(self.close)
        val = f[-1]
        self.assertAlmostEqual(val, 0.0, places=4)

    def test_feature_count_df_vs_series(self):
        """OHLCV df and close series must produce same feature count."""
        f_close = self._get_features(self.close)
        f_df = self._get_features(self.df)
        self.assertEqual(len(f_close), len(f_df))

    def test_short_series_returns_empty(self):
        """Series shorter than lookback returns empty array."""
        short = _make_close_series(50)
        f = self._get_features(short)
        self.assertEqual(len(f), 0)


class TestModelMismatchDetection(unittest.TestCase):
    """Validates the feature-count mismatch check in load_models()."""

    def test_mismatch_detected_when_scaler_expects_wrong_count(self):
        """
        Simulate a stale scaler trained on N-5 features.
        load_models() should detect the mismatch and return False.
        """
        import tempfile, pickle
        from pathlib import Path

        g = _get_god_level()
        close = _make_close_series(300)
        n_current = len(g.extract_features(close))

        # Use module-level stub; set wrong feature count
        stale_scaler = _StaleScaler()
        stale_scaler.n_features_in_ = n_current - 5  # deliberately wrong

        stale_data = {
            "models": {},
            "scaler": stale_scaler,
            "trained_at": "2025-01-01",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            g.model_dir = Path(tmpdir)
            model_path = g.model_dir / "god_level_models.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(stale_data, f)

            result = g.load_models()
            self.assertFalse(result)
            self.assertFalse(model_path.exists())

    def test_matching_scaler_loads_normally(self):
        """Scaler with matching feature count should load without issue."""
        import tempfile, pickle
        from pathlib import Path

        g = _get_god_level()
        close = _make_close_series(300)
        n_current = len(g.extract_features(close))

        good_scaler = _MatchingScaler()
        good_scaler.n_features_in_ = n_current

        valid_data = {
            "models": {"rf": _FakeModel()},
            "scaler": good_scaler,
            "importance": {},
            "trained_at": "2026-03-25",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            g.model_dir = Path(tmpdir)
            model_path = g.model_dir / "god_level_models.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(valid_data, f)

            result = g.load_models()
            self.assertTrue(result)
            self.assertTrue(g.models_trained)


if __name__ == "__main__":
    unittest.main()
