"""Tests for Backtest Validator service functions."""
import pytest
import numpy as np

from services.backtest_validator.service import (
    run_monte_carlo_on_returns,
    compute_robustness_score,
)


class TestMonteCarloOnReturns:
    """Test Monte Carlo bootstrap simulation on return series."""

    def test_basic_monte_carlo(self):
        """MC should return metric keys for valid returns."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        result = run_monte_carlo_on_returns(returns, n_sims=200)

        assert "mc_min_equity" in result
        assert "mc_median_equity" in result
        assert "mc_95_pct_equity" in result
        assert "mc_99_pct_equity" in result
        assert "mc_max_equity" in result

    def test_mc_ordering(self):
        """MC percentiles should be in correct order."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        result = run_monte_carlo_on_returns(returns, n_sims=500)

        assert result["mc_min_equity"] <= result["mc_99_pct_equity"]
        assert result["mc_99_pct_equity"] <= result["mc_95_pct_equity"]
        assert result["mc_95_pct_equity"] <= result["mc_median_equity"]
        assert result["mc_median_equity"] <= result["mc_max_equity"]

    def test_mc_empty_returns(self):
        """MC with too few returns should return empty dict."""
        result = run_monte_carlo_on_returns(np.array([0.01, 0.02]), n_sims=100)
        assert result == {}

    def test_mc_none_returns(self):
        """MC with None returns should return empty dict."""
        result = run_monte_carlo_on_returns(None)
        assert result == {}

    def test_mc_start_equity(self):
        """MC should scale with start equity."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 50)
        result_1 = run_monte_carlo_on_returns(returns, n_sims=100, start_equity=1.0)
        result_100 = run_monte_carlo_on_returns(returns, n_sims=100, start_equity=100.0)

        # Median equity at 100x start should be ~100x larger
        assert result_100["mc_median_equity"] > result_1["mc_median_equity"] * 50


class TestRobustnessScore:
    """Test robustness score computation."""

    def test_moderate_score(self):
        """A reasonable MC + moderate DD should give mid-range score."""
        mc = {
            "mc_min_equity": 0.7,
            "mc_median_equity": 1.2,
            "mc_95_pct_equity": 0.95,
            "mc_99_pct_equity": 0.85,
            "mc_max_equity": 1.8,
        }
        score = compute_robustness_score(mc, {}, max_drawdown_pct=15.0)
        assert 0 <= score <= 100

    def test_empty_inputs(self):
        """Empty MC and stress should return base score."""
        score = compute_robustness_score({}, {}, max_drawdown_pct=0.0)
        assert score == 50.0

    def test_high_drawdown_penalty(self):
        """Large drawdown should reduce score."""
        score_low = compute_robustness_score({}, {}, max_drawdown_pct=5.0)
        score_high = compute_robustness_score({}, {}, max_drawdown_pct=50.0)
        assert score_high < score_low

    def test_score_bounds(self):
        """Score should always be in [0, 100]."""
        mc = {"mc_95_pct_equity": 0.01, "mc_median_equity": 2.0}
        score = compute_robustness_score(mc, {}, max_drawdown_pct=200.0)
        assert 0 <= score <= 100

        mc_good = {"mc_95_pct_equity": 2.0, "mc_median_equity": 1.0}
        score = compute_robustness_score(mc_good, {}, max_drawdown_pct=0.0)
        assert 0 <= score <= 100
