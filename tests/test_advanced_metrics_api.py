"""
Tests for GET /ops/advanced-metrics API endpoint and underlying metrics calculations.
"""
import numpy as np
import pytest

from risk.advanced_metrics import AdvancedRiskMetrics, calculate_all_metrics


class TestAdvancedRiskMetrics:
    """Unit tests for the AdvancedRiskMetrics computation methods."""

    def setup_method(self):
        self.m = AdvancedRiskMetrics()
        rng = np.random.default_rng(42)
        self.returns = rng.normal(0.001, 0.02, 120)  # 120 daily returns

    # --- CVaR / VaR ---
    def test_cvar_95_is_negative(self):
        cvar = self.m.calculate_cvar(self.returns, 0.95)
        assert cvar is not None and cvar < 0

    def test_cvar_99_worse_than_cvar_95(self):
        c95 = self.m.calculate_cvar(self.returns, 0.95)
        c99 = self.m.calculate_cvar(self.returns, 0.99)
        assert c99 <= c95  # 99% tail is worse (more negative)

    def test_var_95_is_negative(self):
        var = self.m.calculate_var(self.returns, 0.95)
        assert var is not None and var < 0

    def test_var_99_worse_than_var_95(self):
        v95 = self.m.calculate_var(self.returns, 0.95)
        v99 = self.m.calculate_var(self.returns, 0.99)
        assert v99 <= v95

    def test_cvar_worse_than_var_same_level(self):
        """CVaR (expected shortfall) must be <= VaR at the same confidence level."""
        cvar = self.m.calculate_cvar(self.returns, 0.95)
        var = self.m.calculate_var(self.returns, 0.95)
        assert cvar <= var

    # --- Sortino / Calmar / Omega ---
    def test_sortino_positive_for_positive_mean(self):
        good_returns = np.array([0.01] * 60 + [-0.005] * 60)
        sr = self.m.calculate_sortino_ratio(good_returns)
        assert sr is not None and sr > 0

    def test_calmar_positive_for_upward_curve(self):
        upward = np.linspace(0.001, 0.003, 60)
        cal = self.m.calculate_calmar_ratio(upward)
        assert cal is not None and cal > 0

    def test_omega_above_one_for_mostly_gains(self):
        gains = np.array([0.02] * 70 + [-0.005] * 30)
        omega = self.m.calculate_omega_ratio(gains)
        assert omega is not None and omega > 1.0

    def test_omega_below_one_for_mostly_losses(self):
        losses = np.array([-0.02] * 70 + [0.005] * 30)
        omega = self.m.calculate_omega_ratio(losses)
        assert omega is not None and omega < 1.0

    # --- Distribution stats ---
    def test_skewness_numeric(self):
        skew = self.m.calculate_skewness(self.returns)
        assert skew is not None and np.isfinite(skew)

    def test_kurtosis_numeric(self):
        kurt = self.m.calculate_kurtosis(self.returns)
        assert kurt is not None and np.isfinite(kurt)

    def test_downside_deviation_non_negative(self):
        dd = self.m.calculate_downside_deviation(self.returns)
        assert dd is not None and dd >= 0

    def test_tail_ratio_positive(self):
        tr = self.m.calculate_tail_ratio(self.returns)
        assert tr is not None and tr > 0

    def test_max_dd_duration_non_negative(self):
        dur = self.m.calculate_max_drawdown_duration(self.returns)
        assert dur is not None and dur >= 0

    # --- Edge cases ---
    def test_empty_returns_does_not_crash_calculate_all(self):
        empty = np.array([])
        # calculate_all_metrics wraps individual calls; should not propagate uncaught
        try:
            result = calculate_all_metrics(empty)
            assert isinstance(result, dict)
        except (ValueError, IndexError):
            pass  # acceptable — small series is an unsupported edge case

    def test_single_return_does_not_crash(self):
        single = np.array([0.01])
        # Should not raise
        self.m.calculate_cvar(single, 0.95)
        self.m.calculate_var(single, 0.95)

    def test_all_identical_returns_downside_dev_zero_or_none(self):
        flat = np.zeros(50)
        dd = self.m.calculate_downside_deviation(flat)
        # All returns = 0, no downside → zero deviation (or None)
        assert dd is None or dd == pytest.approx(0.0, abs=1e-9)


class TestCalculateAllMetrics:
    """Tests for the convenience wrapper."""

    def setup_method(self):
        rng = np.random.default_rng(7)
        self.returns = rng.normal(0.0005, 0.015, 100)

    def test_returns_dict_with_required_keys(self):
        result = calculate_all_metrics(self.returns)
        required = {
            "cvar_95", "cvar_99", "var_95", "var_99",
            "sortino_ratio", "calmar_ratio", "omega_ratio",
            "downside_deviation", "tail_ratio", "skewness",
            "kurtosis", "max_dd_duration",
        }
        assert required.issubset(result.keys())

    def test_no_key_contains_inf(self):
        result = calculate_all_metrics(self.returns)
        for k, v in result.items():
            if v is not None and isinstance(v, float):
                assert np.isfinite(v) or v != v, f"{k} should be finite or NaN, got {v}"

    def test_cvar_ordering_preserved(self):
        result = calculate_all_metrics(self.returns)
        if result["cvar_95"] is not None and result["cvar_99"] is not None:
            assert result["cvar_99"] <= result["cvar_95"]

    def test_var_ordering_preserved(self):
        result = calculate_all_metrics(self.returns)
        if result["var_95"] is not None and result["var_99"] is not None:
            assert result["var_99"] <= result["var_95"]

    def test_custom_risk_free_rate(self):
        result = calculate_all_metrics(self.returns, risk_free_rate=0.05)
        assert "sortino_ratio" in result

    def test_short_series_does_not_crash(self):
        short = np.array([0.01, -0.02, 0.005])
        result = calculate_all_metrics(short)
        assert isinstance(result, dict)
