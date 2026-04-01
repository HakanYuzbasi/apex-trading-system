"""
Tests for AdaptivePositionSizer.
"""
import pytest

from risk.adaptive_position_sizer import AdaptivePositionSizer


class TestAdaptivePositionSizerBasic:
    def setup_method(self):
        self.sizer = AdaptivePositionSizer(base_position_size=5000)

    # --- multiplier bounds ---

    def test_multiplier_within_bounds(self):
        result = self.sizer.calculate_position_size(
            signal_confidence=0.7, volatility=0.20
        )
        assert 0.2 <= result["multiplier"] <= 2.5

    def test_returns_required_keys(self):
        result = self.sizer.calculate_position_size(
            signal_confidence=0.7, volatility=0.20
        )
        assert "position_size" in result
        assert "multiplier" in result
        assert "components" in result
        assert "base_size" in result

    def test_base_size_preserved(self):
        result = self.sizer.calculate_position_size(
            signal_confidence=0.7, volatility=0.20
        )
        assert result["base_size"] == 5000

    # --- confidence ---

    def test_high_confidence_larger_than_low(self):
        hi = self.sizer.calculate_position_size(signal_confidence=0.95, volatility=0.20)
        lo = self.sizer.calculate_position_size(signal_confidence=0.10, volatility=0.20)
        assert hi["position_size"] > lo["position_size"]

    def test_confidence_component_range(self):
        result = self.sizer.calculate_position_size(signal_confidence=1.0, volatility=0.20)
        assert result["components"]["confidence"] == pytest.approx(1.0)

        result2 = self.sizer.calculate_position_size(signal_confidence=0.0, volatility=0.20)
        assert result2["components"]["confidence"] == pytest.approx(0.5)

    # --- drawdown protection ---

    def test_deep_drawdown_reduces_size(self):
        normal = self.sizer.calculate_position_size(
            signal_confidence=0.7, volatility=0.20, current_drawdown=0.01
        )
        deep = self.sizer.calculate_position_size(
            signal_confidence=0.7, volatility=0.20, current_drawdown=0.12
        )
        assert deep["position_size"] < normal["position_size"]

    def test_severe_drawdown_multiplier(self):
        result = self.sizer.calculate_position_size(
            signal_confidence=0.7, volatility=0.20, current_drawdown=0.15
        )
        assert result["components"]["drawdown"] == pytest.approx(0.3)

    # --- Sharpe ratio ---

    def test_great_sharpe_increases_size(self):
        bad = self.sizer.calculate_position_size(
            signal_confidence=0.7, volatility=0.20, sharpe_ratio=-1.0
        )
        great = self.sizer.calculate_position_size(
            signal_confidence=0.7, volatility=0.20, sharpe_ratio=2.0
        )
        assert great["position_size"] > bad["position_size"]

    # --- portfolio cap ---

    def test_portfolio_cap_applied(self):
        result = self.sizer.calculate_position_size(
            signal_confidence=0.95, volatility=0.05, sharpe_ratio=2.0,
            win_rate=0.70, portfolio_value=10_000, max_position_pct=0.01
        )
        assert result["position_size"] <= 10_000 * 0.01 + 0.01

    # --- win rate ---

    def test_high_win_rate_larger_than_low(self):
        hi = self.sizer.calculate_position_size(signal_confidence=0.7, volatility=0.20, win_rate=0.8)
        lo = self.sizer.calculate_position_size(signal_confidence=0.7, volatility=0.20, win_rate=0.3)
        assert hi["position_size"] > lo["position_size"]

    # --- regime multiplier ---

    def test_regime_multiplier_applied(self):
        normal = self.sizer.calculate_position_size(signal_confidence=0.7, volatility=0.20, regime_multiplier=1.0)
        crisis = self.sizer.calculate_position_size(signal_confidence=0.7, volatility=0.20, regime_multiplier=0.2)
        assert crisis["position_size"] < normal["position_size"]

    # --- position size is positive ---

    def test_position_size_always_positive(self):
        result = self.sizer.calculate_position_size(
            signal_confidence=0.01, volatility=0.80, sharpe_ratio=-2.0,
            win_rate=0.1, current_drawdown=0.20, regime_multiplier=0.2
        )
        assert result["position_size"] > 0


class TestKellyCriterion:
    def setup_method(self):
        self.sizer = AdaptivePositionSizer()

    def test_positive_expectancy_positive_kelly(self):
        k = self.sizer.kelly_criterion(win_rate=0.60, avg_win=100, avg_loss=50)
        assert k > 0

    def test_negative_expectancy_zero_kelly(self):
        k = self.sizer.kelly_criterion(win_rate=0.30, avg_win=50, avg_loss=100)
        assert k == 0.0

    def test_zero_loss_returns_zero(self):
        k = self.sizer.kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=0)
        assert k == 0.0

    def test_quarter_kelly_smaller_than_full(self):
        """At fraction=1.0 kelly may hit KELLY_MAX cap; use moderate scenario."""
        full = self.sizer.kelly_criterion(
            win_rate=0.52, avg_win=60, avg_loss=55, kelly_fraction=1.0
        )
        quarter = self.sizer.kelly_criterion(
            win_rate=0.52, avg_win=60, avg_loss=55, kelly_fraction=0.25
        )
        assert quarter <= full  # quarter-Kelly can't exceed full-Kelly

    def test_kelly_clipped_to_max(self):
        from config import ApexConfig
        k = self.sizer.kelly_criterion(win_rate=0.99, avg_win=1000, avg_loss=1, kelly_fraction=1.0)
        assert k <= ApexConfig.KELLY_MAX_POSITION_PCT


class TestVolatilityPercentile:
    def test_percentile_0_to_1(self):
        sizer = AdaptivePositionSizer()
        for vol in [0.10, 0.20, 0.30, 0.40, 0.50]:
            p = sizer._get_volatility_percentile(vol)
            assert 0.0 <= p <= 1.0

    def test_warm_up_returns_0_5(self):
        sizer = AdaptivePositionSizer()
        p = sizer._get_volatility_percentile(0.20)
        # First call — fewer than 10 obs, should return 0.5
        assert p == pytest.approx(0.5)

    def test_high_vol_higher_percentile(self):
        sizer = AdaptivePositionSizer()
        # Build history with many low-vol observations
        for _ in range(50):
            sizer._get_volatility_percentile(0.10)
        low_p = sizer._get_volatility_percentile(0.10)
        high_p = sizer._get_volatility_percentile(0.80)
        assert high_p > low_p
