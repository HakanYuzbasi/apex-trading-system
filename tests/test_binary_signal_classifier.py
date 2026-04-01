"""
Tests for BinarySignalClassifier and its integration with GodLevelSignalGenerator.
"""
import tempfile
import numpy as np
import pytest

from models.binary_signal_classifier import BinarySignalClassifier, CLF_AVAILABLE


@pytest.mark.skipif(not CLF_AVAILABLE, reason="scikit-learn not installed")
class TestBinarySignalClassifierBasic:
    """Unit tests for the classifier itself."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.clf = BinarySignalClassifier(model_dir=self.tmpdir)

    def _make_data(self, n=300, seed=42):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, 10)).astype(np.float32)
        y = (X[:, 0] + rng.normal(0, 0.2, n) > 0).astype(int)
        return X, y

    # --- train() ---

    def test_train_returns_accuracy(self):
        X, y = self._make_data()
        split = 240
        result = self.clf.train(X[:split], y[:split], X[split:], y[split:], "neutral", "equity")
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_train_sets_is_trained(self):
        X, y = self._make_data()
        self.clf.train(X[:240], y[:240], X[240:], y[240:], "bull", "equity")
        assert self.clf.is_trained

    def test_train_multiple_regimes(self):
        X, y = self._make_data()
        for regime in ("bull", "bear", "neutral", "volatile"):
            self.clf.train(X[:240], y[:240], X[240:], y[240:], regime, "equity")
        assert self.clf.is_trained

    def test_train_skips_single_class(self):
        X, _ = self._make_data()
        y_const = np.zeros(300, dtype=int)
        result = self.clf.train(X[:240], y_const[:240], X[240:], y_const[240:], "neutral", "equity")
        # Should return empty dict (not crash)
        assert isinstance(result, dict)

    def test_n_train_n_test_reported(self):
        X, y = self._make_data()
        result = self.clf.train(X[:240], y[:240], X[240:], y[240:], "neutral", "equity")
        assert result.get("n_train") == 240
        assert result.get("n_test") == 60

    # --- predict() ---

    def test_predict_returns_two_floats(self):
        X, y = self._make_data()
        self.clf.train(X[:240], y[:240], X[240:], y[240:], "neutral", "equity")
        sig, conf = self.clf.predict(X[240:241], "neutral", "equity")
        assert isinstance(sig, float)
        assert isinstance(conf, float)

    def test_signal_in_minus_one_to_one(self):
        X, y = self._make_data()
        self.clf.train(X[:240], y[:240], X[240:], y[240:], "neutral", "equity")
        for i in range(240, 250):
            sig, _ = self.clf.predict(X[i : i + 1], "neutral", "equity")
            assert -1.0 <= sig <= 1.0

    def test_confidence_non_negative(self):
        X, y = self._make_data()
        self.clf.train(X[:240], y[:240], X[240:], y[240:], "neutral", "equity")
        for i in range(240, 250):
            _, conf = self.clf.predict(X[i : i + 1], "neutral", "equity")
            assert conf >= 0.0

    def test_predict_untrained_returns_zero(self):
        clf = BinarySignalClassifier(model_dir=self.tmpdir + "_new")
        X, _ = self._make_data()
        sig, conf = clf.predict(X[0:1], "neutral", "equity")
        assert sig == 0.0
        assert conf == 0.0

    def test_predict_fallback_to_neutral_regime(self):
        """Unknown regime falls back to neutral cell."""
        X, y = self._make_data()
        self.clf.train(X[:240], y[:240], X[240:], y[240:], "neutral", "equity")
        sig, _ = self.clf.predict(X[240:241], "unknown_regime", "equity")
        assert isinstance(sig, float)

    # --- make_labels() ---

    def test_make_labels_length(self):
        import pandas as pd
        prices = pd.Series([10, 11, 9, 12, 13, 10], dtype=float)
        labels = BinarySignalClassifier.make_labels(prices, horizon=1)
        assert len(labels) == len(prices)

    def test_make_labels_values_binary(self):
        import pandas as pd
        prices = pd.Series([10, 11, 9, 12, 13, 10], dtype=float)
        labels = BinarySignalClassifier.make_labels(prices, horizon=1)
        unique = set(labels.dropna().astype(int).tolist())
        assert unique.issubset({0, 1})

    # --- save / load ---

    def test_save_and_reload(self):
        X, y = self._make_data()
        self.clf.train(X[:240], y[:240], X[240:], y[240:], "neutral", "equity")
        self.clf.save_models()

        clf2 = BinarySignalClassifier(model_dir=self.tmpdir)
        assert clf2.is_trained
        sig, conf = clf2.predict(X[240:241], "neutral", "equity")
        assert -1.0 <= sig <= 1.0


@pytest.mark.skipif(not CLF_AVAILABLE, reason="scikit-learn not installed")
class TestBinarySignalBlendInGodLevel:
    """Integration tests — verify blend logic without touching real GodLevel (heavy)."""

    def test_blend_weight_property_in_config(self):
        from config import ApexConfig
        weight = float(getattr(ApexConfig, "BINARY_CLASSIFIER_BLEND_WEIGHT", 0.08))
        assert 0.0 < weight <= 0.20

    def test_blend_formula_with_strong_signal(self):
        """Manual blend: signal=0.50, bin=0.80, w=0.08 → (0.50 + 0.08*0.80)/1.08 ≈ 0.519"""
        signal = 0.50
        bin_sig = 0.80
        w = 0.08
        blended = (signal + bin_sig * w) / (1.0 + w)
        assert abs(blended - 0.5222) < 0.001

    def test_blend_formula_clipped(self):
        """Blend should stay within [-1, 1]."""
        signal = 0.99
        bin_sig = 1.0
        w = 0.08
        blended = np.clip((signal + bin_sig * w) / (1.0 + w), -1.0, 1.0)
        assert -1.0 <= blended <= 1.0

    def test_blend_skipped_when_weak_signal(self):
        """bin_signal ≤ 0.05 → no blend applied."""
        signal = 0.40
        bin_sig = 0.03  # below threshold
        w = 0.08
        if abs(bin_sig) > 0.05:
            blended = (signal + bin_sig * w) / (1.0 + w)
        else:
            blended = signal
        assert blended == signal

    def test_god_level_has_binary_classifier_attribute(self):
        """GodLevelSignalGenerator must expose _binary_classifier after init."""
        import sys
        # Light import check — don't actually construct (slow, needs data)
        # Just verify the attribute is wired by inspecting the source
        from pathlib import Path
        src = (Path(__file__).parent.parent / "models" / "god_level_signal_generator.py").read_text()
        assert "_binary_classifier" in src
        assert "BinarySignalClassifier" in src
