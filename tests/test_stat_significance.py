"""Tests for stat_significance module."""
from __future__ import annotations

import pytest

from monitoring.stat_significance import (
    binomial_pvalue,
    is_significant,
    significance_summary,
    wilson_interval,
)


class TestWilsonInterval:

    def test_zero_wins_returns_valid_interval(self):
        lo, hi = wilson_interval(0, 10)
        assert 0.0 <= lo <= hi <= 1.0

    def test_all_wins_returns_valid_interval(self):
        lo, hi = wilson_interval(10, 10)
        assert 0.0 <= lo <= hi <= 1.0

    def test_50pct_win_rate_interval_contains_0_5(self):
        lo, hi = wilson_interval(50, 100)
        assert lo < 0.5 < hi

    def test_interval_narrows_with_more_data(self):
        lo10, hi10 = wilson_interval(5, 10)
        lo100, hi100 = wilson_interval(50, 100)
        assert (hi10 - lo10) > (hi100 - lo100)

    def test_zero_n_returns_full_range(self):
        lo, hi = wilson_interval(0, 0)
        assert lo == 0.0 and hi == 1.0

    def test_high_confidence_wider_than_low_confidence(self):
        lo95, hi95 = wilson_interval(50, 100, confidence=0.95)
        lo90, hi90 = wilson_interval(50, 100, confidence=0.90)
        assert (hi95 - lo95) > (hi90 - lo90)


class TestBinomialPvalue:

    def test_50pct_win_rate_high_pvalue(self):
        """50% WR on 100 trades → fail to reject H0."""
        p = binomial_pvalue(50, 100)
        assert p > 0.05

    def test_extreme_win_rate_low_pvalue(self):
        """100% WR on 30 trades → strongly reject H0."""
        p = binomial_pvalue(30, 30)
        assert p < 0.001

    def test_zero_wins_very_low_pvalue(self):
        p = binomial_pvalue(0, 30)
        assert p < 0.001

    def test_empty_n_returns_one(self):
        p = binomial_pvalue(0, 0)
        assert p == 1.0

    def test_symmetric(self):
        """P(k=8 | n=10) == P(k=2 | n=10) for two-sided test."""
        p1 = binomial_pvalue(8, 10)
        p2 = binomial_pvalue(2, 10)
        assert p1 == pytest.approx(p2, abs=0.001)


class TestIsSignificant:

    def test_50_pct_100_trades_not_significant(self):
        assert not is_significant(50, 100)

    def test_100_pct_30_trades_significant(self):
        assert is_significant(30, 30)

    def test_0_pct_30_trades_significant(self):
        assert is_significant(0, 30)

    def test_zero_n_not_significant(self):
        assert not is_significant(0, 0)

    def test_moderate_deviation_small_n_not_significant(self):
        """30% WR but only 5 trades → not significant."""
        assert not is_significant(wins=1, n=5, alpha=0.05)

    def test_moderate_deviation_large_n_significant(self):
        """30% WR over 100 trades → significant."""
        assert is_significant(wins=30, n=100, alpha=0.05)

    def test_binomial_method(self):
        assert is_significant(30, 30, method="binomial")
        assert not is_significant(50, 100, method="binomial")

    def test_alpha_controls_strictness(self):
        """Same data: significant at alpha=0.10 but possibly not at 0.01."""
        # 70% WR over 30 trades
        loose = is_significant(21, 30, alpha=0.10)
        strict = is_significant(21, 30, alpha=0.01)
        # loose should be at least as permissive as strict
        if strict:
            assert loose

    def test_null_p_custom(self):
        """Test against a non-0.5 null hypothesis."""
        # 60% WR vs null=0.55: should not be significant with small n
        assert not is_significant(6, 10, null_p=0.55)


class TestSignificanceSummary:

    def test_summary_structure(self):
        s = significance_summary(30, 30)
        assert "significant" in s
        assert "win_rate" in s
        assert "pvalue" in s
        assert "ci_lo" in s
        assert "ci_hi" in s
        assert "n" in s

    def test_summary_significant_case(self):
        s = significance_summary(30, 30)
        assert s["significant"] is True
        assert s["win_rate"] == 1.0

    def test_summary_not_significant_case(self):
        s = significance_summary(50, 100)
        assert s["significant"] is False

    def test_summary_zero_n(self):
        s = significance_summary(0, 0)
        assert s["significant"] is False
        assert s["pvalue"] == 1.0


class TestSignificanceGateIntegration:
    """Confirm the gate behaves correctly for the auto-tuner use case."""

    def test_low_win_rate_large_sample_significant(self):
        """AutoTuner scenario: 30% WR over 50 trades → should trigger adjustment."""
        wins = 15
        n = 50
        assert is_significant(wins, n, alpha=0.05)

    def test_low_win_rate_tiny_sample_not_significant(self):
        """AutoTuner scenario: 30% WR over 5 trades → hold off."""
        wins = 1
        n = 5
        assert not is_significant(wins, n, alpha=0.05)

    def test_high_win_rate_large_sample_significant(self):
        """AutoTuner scenario: 70% WR over 50 trades → lower threshold."""
        wins = 35
        n = 50
        assert is_significant(wins, n, alpha=0.05)
