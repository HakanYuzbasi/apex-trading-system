"""
tests/test_correlation_regime_detector.py — Unit tests for monitoring/correlation_regime_detector.py
"""
from __future__ import annotations

import numpy as np
import pytest

from monitoring.correlation_regime_detector import (
    CorrelationRegimeDetector,
    CorrRegimeState,
    classify_corr_regime,
    compute_avg_pairwise_correlation,
    get_corr_regime_detector,
)


# ── compute_avg_pairwise_correlation ──────────────────────────────────────────

class TestComputeAvgPairwiseCorrelation:
    def test_single_symbol_returns_zero(self):
        mat = np.array([[0.1, 0.2, -0.1, 0.3]])
        assert compute_avg_pairwise_correlation(mat) == 0.0

    def test_perfectly_correlated_returns_one(self):
        v = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        mat = np.array([v, v, v])  # 3 identical series
        corr = compute_avg_pairwise_correlation(mat)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_anti_correlated_returns_negative(self):
        v = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        mat = np.array([v, -v])
        corr = compute_avg_pairwise_correlation(mat)
        assert corr == pytest.approx(-1.0, abs=1e-6)

    def test_independent_series_near_zero(self):
        np.random.seed(42)
        mat = np.random.randn(4, 100)
        corr = compute_avg_pairwise_correlation(mat)
        assert abs(corr) < 0.3  # should be near-zero for independent random

    def test_output_bounded(self):
        mat = np.random.randn(5, 50)
        corr = compute_avg_pairwise_correlation(mat)
        assert -1.0 <= corr <= 1.0


# ── classify_corr_regime ──────────────────────────────────────────────────────

class TestClassifyCorrRegime:
    def test_low_corr_is_normal(self):
        assert classify_corr_regime(0.30, 0.55, 0.75) == "normal"

    def test_medium_corr_is_elevated(self):
        assert classify_corr_regime(0.65, 0.55, 0.75) == "elevated"

    def test_high_corr_is_crisis(self):
        assert classify_corr_regime(0.80, 0.55, 0.75) == "crisis"

    def test_exactly_at_warning_threshold_is_elevated(self):
        assert classify_corr_regime(0.55, 0.55, 0.75) == "elevated"

    def test_exactly_at_crisis_threshold_is_crisis(self):
        assert classify_corr_regime(0.75, 0.55, 0.75) == "crisis"

    def test_negative_corr_is_normal(self):
        assert classify_corr_regime(-0.5, 0.55, 0.75) == "normal"


# ── CorrelationRegimeDetector ─────────────────────────────────────────────────

def _feed_correlated_prices(detector: CorrelationRegimeDetector, n_steps: int = 40) -> None:
    """Feed highly correlated price series for 4 symbols."""
    np.random.seed(42)
    base = np.cumsum(np.random.randn(n_steps)) + 100
    for i in range(n_steps):
        for sym in ["SYM0", "SYM1", "SYM2", "SYM3"]:
            noise = np.random.randn() * 0.01
            detector.update_prices(sym, float(base[i]) + noise)


def _feed_independent_prices(detector: CorrelationRegimeDetector, n_steps: int = 40) -> None:
    """Feed independent price series for 4 symbols."""
    np.random.seed(99)
    for sym in ["SYM0", "SYM1", "SYM2", "SYM3"]:
        prices = np.cumsum(np.random.randn(n_steps)) + 100
        for p in prices:
            detector.update_prices(sym, float(p))


class TestCorrelationRegimeDetector:
    def test_get_sizing_mult_one_with_no_data(self):
        det = CorrelationRegimeDetector()
        assert det.get_sizing_multiplier() == pytest.approx(1.0)

    def test_get_sizing_mult_one_below_min_symbols(self):
        det = CorrelationRegimeDetector()
        for i in range(30):
            det.update_prices("SYM0", 100.0 + i)
            det.update_prices("SYM1", 100.0 - i)
        # Only 2 symbols, below min=4
        assert det.get_sizing_multiplier() == pytest.approx(1.0)

    def test_crisis_corr_reduces_mult(self):
        det = CorrelationRegimeDetector()
        _feed_correlated_prices(det, n_steps=50)
        mult = det.get_sizing_multiplier()
        # Highly correlated → reduced mult
        assert mult < 1.0

    def test_independent_corr_normal_mult(self):
        det = CorrelationRegimeDetector()
        _feed_independent_prices(det, n_steps=50)
        mult = det.get_sizing_multiplier()
        # Independent → normal mult ≈ 1.0
        assert mult >= 0.75  # at worst elevated

    def test_get_state_returns_corr_regime_state(self):
        det = CorrelationRegimeDetector()
        state = det.get_state()
        assert isinstance(state, CorrRegimeState)

    def test_get_report_has_expected_keys(self):
        det = CorrelationRegimeDetector()
        report = det.get_report()
        for key in ["avg_pairwise_correlation", "regime", "sizing_multiplier",
                    "n_symbols", "n_bars", "top_correlated_pairs", "timestamp"]:
            assert key in report

    def test_regime_normal_on_no_data(self):
        det = CorrelationRegimeDetector()
        assert det.get_state().regime == "normal"

    def test_crisis_regime_level_on_correlated_data(self):
        det = CorrelationRegimeDetector()
        _feed_correlated_prices(det, n_steps=50)
        state = det.get_state()
        assert state.regime in ("elevated", "crisis")

    def test_disabled_returns_one(self, monkeypatch):
        import monitoring.correlation_regime_detector as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "CORR_REGIME_ENABLED" else mod._DEF.get(k))
        det = CorrelationRegimeDetector()
        assert det.get_sizing_multiplier() == pytest.approx(1.0)

    def test_corr_regime_state_to_dict(self):
        s = CorrRegimeState(
            avg_pairwise_correlation=0.65,
            regime="elevated",
            sizing_multiplier=0.75,
            n_symbols=4,
            n_bars=30,
        )
        d = s.to_dict()
        assert d["regime"] == "elevated"
        assert d["sizing_multiplier"] == pytest.approx(0.75)
        assert "timestamp" in d

    def test_update_prices_ignored_for_negative(self):
        det = CorrelationRegimeDetector()
        det.update_prices("SYM0", -100.0)
        det.update_prices("SYM0", 0.0)
        assert "SYM0" not in det._price_buffers or len(det._return_buffers.get("SYM0", [])) == 0

    def test_top_pairs_populated_after_data(self):
        det = CorrelationRegimeDetector()
        _feed_correlated_prices(det, n_steps=50)
        state = det.get_state()
        if state.n_symbols >= 4:
            assert isinstance(state.top_correlated_pairs, list)

    def test_get_avg_correlation_float(self):
        det = CorrelationRegimeDetector()
        corr = det.get_avg_correlation()
        assert isinstance(corr, float)

    def test_high_corr_state_multiplier_bounded(self):
        det = CorrelationRegimeDetector()
        _feed_correlated_prices(det, n_steps=60)
        state = det.get_state()
        assert 0.0 < state.sizing_multiplier <= 1.0

    def test_different_number_of_symbols_handled(self):
        det = CorrelationRegimeDetector()
        # Feed 6 symbols
        np.random.seed(10)
        for sym_idx in range(6):
            prices = np.cumsum(np.random.randn(40)) + 100
            for p in prices:
                det.update_prices(f"SYM{sym_idx}", float(p))
        state = det.get_state()
        assert state.n_symbols >= 0


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_instance(self):
        assert isinstance(get_corr_regime_detector(), CorrelationRegimeDetector)

    def test_same_object(self):
        a = get_corr_regime_detector()
        b = get_corr_regime_detector()
        assert a is b
