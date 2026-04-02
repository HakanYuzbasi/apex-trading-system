# tests/test_risk_coverage.py - Tests for entry_filters and adaptive_position_sizer

import numpy as np
import pandas as pd
import pytest

from risk.entry_filters import _compute_20d_vwap, vwap_gate_check, rvol_check
from risk.adaptive_position_sizer import AdaptivePositionSizer


# ── _compute_20d_vwap ───────────────────────────────────────────────────────

class TestCompute20dVwap:
    def _make_df(self, n=25, base_price=100.0, volume=1000):
        dates = pd.date_range("2026-01-01", periods=n)
        return pd.DataFrame({
            "Open": base_price,
            "High": base_price + 1,
            "Low": base_price - 1,
            "Close": base_price,
            "Volume": volume,
        }, index=dates)

    def test_basic_vwap(self):
        df = self._make_df()
        vwap = _compute_20d_vwap(df)
        assert vwap is not None
        assert abs(vwap - 100.0) < 0.5  # TP = (101+99+100)/3 = 100

    def test_none_on_empty(self):
        assert _compute_20d_vwap(None) is None
        assert _compute_20d_vwap(pd.DataFrame()) is None

    def test_none_on_too_few_rows(self):
        df = self._make_df(n=3)
        assert _compute_20d_vwap(df) is None

    def test_none_on_missing_volume(self):
        df = self._make_df()
        df = df.drop(columns=["Volume"])
        assert _compute_20d_vwap(df) is None

    def test_none_on_zero_volume(self):
        df = self._make_df(volume=0)
        assert _compute_20d_vwap(df) is None

    def test_uses_last_20_bars(self):
        df = self._make_df(n=40)
        vwap = _compute_20d_vwap(df)
        assert vwap is not None


# ── vwap_gate_check ─────────────────────────────────────────────────────────

class TestVwapGateCheck:
    def _make_df(self, close=100.0, n=25):
        dates = pd.date_range("2026-01-01", periods=n)
        return pd.DataFrame({
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1000,
        }, index=dates)

    def test_not_blocked_near_vwap(self):
        df = self._make_df(close=100.0)
        blocked, dev, reason = vwap_gate_check(100.5, df, signal=1.0)
        assert not blocked

    def test_blocked_long_above_vwap(self):
        df = self._make_df(close=100.0)
        blocked, dev, reason = vwap_gate_check(105.0, df, signal=1.0, max_deviation_pct=2.0)
        assert blocked
        assert "VWAP gate" in reason

    def test_blocked_short_below_vwap(self):
        df = self._make_df(close=100.0)
        blocked, dev, reason = vwap_gate_check(95.0, df, signal=-1.0, max_deviation_pct=2.0)
        assert blocked

    def test_not_blocked_when_vwap_unavailable(self):
        blocked, dev, reason = vwap_gate_check(100.0, pd.DataFrame(), signal=1.0)
        assert not blocked

    def test_crypto_has_wider_threshold(self):
        df = self._make_df(close=100.0)
        # 3% above VWAP: blocked for equity (2% threshold), not for crypto (4%)
        blocked_eq, _, _ = vwap_gate_check(103.5, df, signal=1.0, max_deviation_pct=2.0, is_crypto=False)
        blocked_cr, _, _ = vwap_gate_check(103.5, df, signal=1.0, max_deviation_pct=2.0, is_crypto=True)
        assert blocked_eq
        assert not blocked_cr

    def test_atr_adjust_widens_threshold(self):
        df = self._make_df(close=100.0)
        # With high ATR, threshold widens
        blocked, _, _ = vwap_gate_check(
            103.5, df, signal=1.0, atr_pct=3.0,
            max_deviation_pct=2.0, atr_adjust=True,
        )
        assert not blocked  # 3.0 * 1.5 = 4.5% threshold > 3.5%


# ── rvol_check ──────────────────────────────────────────────────────────────

class TestRvolCheck:
    def _make_df(self, volumes):
        dates = pd.date_range("2026-01-01", periods=len(volumes))
        return pd.DataFrame({
            "Close": 100.0,
            "Volume": volumes,
        }, index=dates)

    def test_normal_volume_passes(self):
        vols = [1000] * 25
        df = self._make_df(vols)
        rvol, blocked, reason = rvol_check(df)
        assert not blocked
        assert abs(rvol - 1.0) < 0.1

    def test_low_volume_blocked(self):
        vols = [1000] * 24 + [100]  # today's volume is 10% of average
        df = self._make_df(vols)
        rvol, blocked, reason = rvol_check(df)
        assert blocked
        assert "RVOL" in reason

    def test_high_volume_passes(self):
        vols = [1000] * 24 + [5000]
        df = self._make_df(vols)
        rvol, blocked, reason = rvol_check(df)
        assert not blocked
        assert rvol > 1.0

    def test_empty_df_not_blocked(self):
        rvol, blocked, _ = rvol_check(pd.DataFrame())
        assert not blocked

    def test_no_volume_column(self):
        df = pd.DataFrame({"Close": [100.0] * 10})
        rvol, blocked, _ = rvol_check(df)
        assert not blocked

    def test_too_few_rows(self):
        df = self._make_df([1000, 1000, 1000])
        rvol, blocked, _ = rvol_check(df)
        assert not blocked


# ── AdaptivePositionSizer ───────────────────────────────────────────────────

class TestAdaptivePositionSizer:
    def test_default_init(self):
        sizer = AdaptivePositionSizer()
        assert sizer.base_position_size == 5000

    def test_custom_base(self):
        sizer = AdaptivePositionSizer(base_position_size=10000)
        assert sizer.base_position_size == 10000

    def test_basic_calculation(self):
        sizer = AdaptivePositionSizer(base_position_size=5000)
        result = sizer.calculate_position_size(
            signal_confidence=0.8,
            volatility=0.15,
        )
        assert "position_size" in result
        assert "multiplier" in result
        assert "components" in result
        assert result["position_size"] > 0

    def test_high_confidence_larger(self):
        sizer = AdaptivePositionSizer(base_position_size=5000)
        low = sizer.calculate_position_size(signal_confidence=0.1, volatility=0.15)
        high = sizer.calculate_position_size(signal_confidence=0.9, volatility=0.15)
        assert high["position_size"] > low["position_size"]

    def test_drawdown_reduces_size(self):
        sizer = AdaptivePositionSizer(base_position_size=5000)
        normal = sizer.calculate_position_size(
            signal_confidence=0.8, volatility=0.15, current_drawdown=0.01,
        )
        deep_dd = sizer.calculate_position_size(
            signal_confidence=0.8, volatility=0.15, current_drawdown=0.15,
        )
        assert deep_dd["position_size"] < normal["position_size"]
        assert deep_dd["components"]["drawdown"] == 0.3

    def test_sharpe_multipliers(self):
        sizer = AdaptivePositionSizer(base_position_size=5000)
        # Great sharpe
        great = sizer.calculate_position_size(
            signal_confidence=0.8, volatility=0.15, sharpe_ratio=2.0,
        )
        # Poor sharpe
        poor = sizer.calculate_position_size(
            signal_confidence=0.8, volatility=0.15, sharpe_ratio=-0.5,
        )
        assert great["components"]["sharpe"] == 1.3
        assert poor["components"]["sharpe"] == 0.6

    def test_portfolio_cap(self):
        sizer = AdaptivePositionSizer(base_position_size=100_000)
        result = sizer.calculate_position_size(
            signal_confidence=1.0,
            volatility=0.10,
            portfolio_value=100_000,
            max_position_pct=0.02,
        )
        assert result["position_size"] <= 100_000 * 0.02

    def test_multiplier_capped(self):
        sizer = AdaptivePositionSizer(base_position_size=5000)
        # Try to get extreme multiplier
        result = sizer.calculate_position_size(
            signal_confidence=1.0,
            volatility=0.01,
            sharpe_ratio=3.0,
            win_rate=1.0,
            regime_multiplier=5.0,
        )
        assert result["multiplier"] <= 2.5

    def test_multiplier_floor(self):
        sizer = AdaptivePositionSizer(base_position_size=5000)
        result = sizer.calculate_position_size(
            signal_confidence=0.0,
            volatility=0.50,
            sharpe_ratio=-1.0,
            win_rate=0.0,
            current_drawdown=0.20,
            regime_multiplier=0.1,
        )
        assert result["multiplier"] >= 0.2


class TestKellyCriterion:
    def test_basic_kelly(self):
        sizer = AdaptivePositionSizer()
        pct = sizer.kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=50)
        assert pct > 0

    def test_zero_loss_returns_zero(self):
        sizer = AdaptivePositionSizer()
        pct = sizer.kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=0)
        assert pct == 0.0

    def test_zero_win_rate_returns_zero(self):
        sizer = AdaptivePositionSizer()
        pct = sizer.kelly_criterion(win_rate=0, avg_win=100, avg_loss=50)
        assert pct == 0.0

    def test_quarter_kelly_fraction(self):
        sizer = AdaptivePositionSizer()
        full = sizer.kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=50, kelly_fraction=1.0)
        quarter = sizer.kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=50, kelly_fraction=0.25)
        assert quarter < full or quarter == full  # quarter Kelly is smaller or equal (capped)

    def test_negative_kelly_clipped_to_zero(self):
        sizer = AdaptivePositionSizer()
        pct = sizer.kelly_criterion(win_rate=0.2, avg_win=10, avg_loss=100)
        assert pct == 0.0


class TestVolatilityPercentile:
    def test_insufficient_data(self):
        sizer = AdaptivePositionSizer()
        # Less than 10 data points → returns 0.5
        pct = sizer._get_volatility_percentile(0.15)
        assert pct == 0.5

    def test_with_history(self):
        sizer = AdaptivePositionSizer()
        # Feed enough data points
        for v in np.linspace(0.10, 0.40, 20):
            sizer._get_volatility_percentile(v)
        pct = sizer._get_volatility_percentile(0.25)
        assert 0.0 <= pct <= 1.0

    def test_history_capped_at_252(self):
        sizer = AdaptivePositionSizer()
        for i in range(300):
            sizer._get_volatility_percentile(float(i) / 1000)
        assert len(sizer.volatility_history) <= 252
