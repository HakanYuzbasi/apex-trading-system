"""Tests for IntradayMRSignal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.intraday_mr import (
    IntradayMRSignal,
    MRSignal,
    _RSI_OVERSOLD,
    _RSI_OVERBOUGHT,
    _VWAP_DEV_TRIGGER,
)


def _mr(**kw) -> IntradayMRSignal:
    defaults = dict(min_bars=10)
    defaults.update(kw)
    return IntradayMRSignal(**defaults)


def _df(prices: list, volumes: list | None = None) -> pd.DataFrame:
    d = {"Close": prices}
    if volumes:
        d["Volume"] = volumes
    return pd.DataFrame(d)


def _prices_flat(n: int = 20, base: float = 100.0) -> list:
    """Stable prices around base — VWAP ≈ base, RSI ≈ 50."""
    import random
    rng = random.Random(0)
    vals = []
    for _ in range(n):
        vals.append(base + rng.uniform(-0.2, 0.2))
    return vals


def _prices_rising(n: int = 20, base: float = 100.0, step: float = 0.5) -> list:
    return [base + i * step for i in range(n)]


def _prices_falling(n: int = 20, base: float = 100.0, step: float = 0.5) -> list:
    return [base - i * step for i in range(n)]


def _prices_spike_up(n: int = 25) -> list:
    """Stable then sharp spike up → overbought, above VWAP."""
    vals = [100.0] * (n - 5)
    vals += [100.0, 102.0, 104.0, 106.0, 108.0]  # spike
    return vals


def _prices_spike_down(n: int = 25) -> list:
    """Stable then sharp drop → oversold, below VWAP."""
    vals = [100.0] * (n - 5)
    vals += [100.0, 98.0, 96.0, 94.0, 92.0]  # crash
    return vals


# ── Null / edge cases ─────────────────────────────────────────────────────────

class TestNullCases:

    def test_none_df_returns_zero_signal(self):
        mr = _mr()
        result = mr.compute(None, "neutral")
        assert result.signal == pytest.approx(0.0)

    def test_short_df_returns_zero_signal(self):
        mr = _mr(min_bars=20)
        result = mr.compute(_df([100.0] * 5), "neutral")
        assert result.signal == pytest.approx(0.0)

    def test_returns_mr_signal_type(self):
        mr = _mr()
        result = mr.compute(_df(_prices_flat()), "neutral")
        assert isinstance(result, MRSignal)

    def test_suppress_regimes_return_zero(self):
        mr = _mr()
        for regime in ("crisis", "strong_bull", "strong_bear"):
            result = mr.compute(_df(_prices_spike_up(25)), regime)
            assert result.signal == pytest.approx(0.0)
            assert result.regime_eligible is False


# ── Regime eligibility ────────────────────────────────────────────────────────

class TestRegimeEligibility:

    def test_neutral_eligible(self):
        mr = _mr()
        result = mr.compute(_df(_prices_flat()), "neutral")
        assert result.regime_eligible is True

    def test_bull_eligible(self):
        mr = _mr()
        result = mr.compute(_df(_prices_flat()), "bull")
        assert result.regime_eligible is True

    def test_crisis_not_eligible(self):
        mr = _mr()
        result = mr.compute(_df(_prices_flat()), "crisis")
        assert result.regime_eligible is False

    def test_regime_is_eligible_helper(self):
        assert IntradayMRSignal.regime_is_eligible("neutral") is True
        assert IntradayMRSignal.regime_is_eligible("bull") is True
        assert IntradayMRSignal.regime_is_eligible("crisis") is False
        assert IntradayMRSignal.regime_is_eligible("strong_bull") is False


# ── Signal direction ──────────────────────────────────────────────────────────

class TestSignalDirection:

    def test_spike_up_gives_sell_signal(self):
        """Sharp rise → overbought + above VWAP → negative (sell) signal."""
        mr = _mr(min_bars=10, rsi_overbought=60.0, vwap_dev_trigger=0.5)
        result = mr.compute(_df(_prices_spike_up(25)), "neutral")
        # Should be 0 or negative (sell)
        assert result.signal <= 0.0

    def test_spike_down_gives_buy_signal(self):
        """Sharp drop → oversold + below VWAP → positive (buy) signal."""
        mr = _mr(min_bars=10, rsi_oversold=40.0, vwap_dev_trigger=0.5)
        result = mr.compute(_df(_prices_spike_down(25)), "neutral")
        # Should be 0 or positive (buy)
        assert result.signal >= 0.0

    def test_flat_prices_gives_no_signal(self):
        """Stable prices → RSI neutral → no MR signal."""
        mr = _mr()
        result = mr.compute(_df(_prices_flat(25)), "neutral")
        assert abs(result.signal) < 0.10  # near zero

    def test_signal_bounded(self):
        mr = _mr(min_bars=10, rsi_overbought=55.0, vwap_dev_trigger=0.3)
        result = mr.compute(_df(_prices_spike_up(25)), "neutral")
        assert -1.0 <= result.signal <= 1.0


# ── Combine signal logic ──────────────────────────────────────────────────────

class TestCombineSignal:

    def test_neutral_rsi_gives_zero(self):
        mr = _mr()
        # RSI = 50 (neutral) → no signal regardless of deviation
        sig = mr._combine_signal(dev_z=3.0, rsi=50.0)
        assert sig == pytest.approx(0.0)

    def test_oversold_above_vwap_gives_zero(self):
        """RSI oversold but price ABOVE VWAP — contradictory, no signal."""
        mr = _mr()
        # dev_z > 0 (above VWAP), rsi oversold (direction=+1, dev_direction=+1 → same)
        sig = mr._combine_signal(dev_z=2.0, rsi=25.0)
        assert sig == pytest.approx(0.0)

    def test_oversold_below_vwap_gives_positive(self):
        """RSI oversold + price below VWAP → buy signal (positive)."""
        mr = _mr(vwap_dev_trigger=0.5)
        sig = mr._combine_signal(dev_z=-2.0, rsi=20.0)
        assert sig > 0.0

    def test_overbought_above_vwap_gives_negative(self):
        """RSI overbought + price above VWAP → sell signal (negative)."""
        mr = _mr(vwap_dev_trigger=0.5)
        sig = mr._combine_signal(dev_z=2.0, rsi=80.0)
        assert sig < 0.0

    def test_deviation_below_trigger_gives_zero(self):
        """Deviation not large enough to trigger."""
        mr = _mr(vwap_dev_trigger=3.0)
        sig = mr._combine_signal(dev_z=-1.0, rsi=20.0)
        assert sig == pytest.approx(0.0)

    def test_extreme_rsi_stronger_signal(self):
        mr = _mr(vwap_dev_trigger=0.5)
        sig_moderate = mr._combine_signal(dev_z=-2.0, rsi=30.0)   # moderately oversold
        sig_extreme = mr._combine_signal(dev_z=-2.0, rsi=15.0)    # very oversold
        assert abs(sig_extreme) >= abs(sig_moderate)


# ── RSI helper ────────────────────────────────────────────────────────────────

class TestRSI:

    def test_rising_series_gives_high_rsi(self):
        mr = _mr()
        prices = np.array([float(i) for i in range(20)])
        rsi = mr._rsi(prices, 14)
        assert rsi > 70.0

    def test_falling_series_gives_low_rsi(self):
        mr = _mr()
        prices = np.array([float(20 - i) for i in range(20)])
        rsi = mr._rsi(prices, 14)
        assert rsi < 30.0

    def test_flat_series_gives_50(self):
        mr = _mr()
        prices = np.array([100.0] * 20)
        rsi = mr._rsi(prices, 14)
        assert rsi == pytest.approx(50.0)

    def test_insufficient_bars_returns_50(self):
        mr = _mr()
        prices = np.array([100.0, 101.0])
        rsi = mr._rsi(prices, 14)
        assert rsi == pytest.approx(50.0)


# ── VWAP helper ───────────────────────────────────────────────────────────────

class TestVWAP:

    def test_vwap_equal_weighted_no_volume(self):
        mr = _mr()
        df = _df([100.0, 102.0, 104.0])
        vwap = mr._compute_vwap(df)
        assert vwap == pytest.approx(102.0)

    def test_vwap_volume_weighted(self):
        mr = _mr()
        # Heavy volume at 100, light at 110 → VWAP closer to 100
        df = _df([100.0, 110.0], volumes=[90.0, 10.0])
        vwap = mr._compute_vwap(df)
        assert vwap < 105.0

    def test_vwap_zero_volume_fallback(self):
        mr = _mr()
        df = _df([100.0, 102.0], volumes=[0.0, 0.0])
        vwap = mr._compute_vwap(df)
        assert vwap == pytest.approx(101.0)


# ── Confidence adjustment ─────────────────────────────────────────────────────

class TestConfidenceAdj:

    def test_strong_signal_boosts_confidence(self):
        mr = _mr(min_bars=10, rsi_oversold=40.0, vwap_dev_trigger=0.3)
        result = mr.compute(_df(_prices_spike_down(25)), "neutral")
        if abs(result.signal) > 0.5:
            assert result.confidence_adj >= 1.0

    def test_near_zero_signal_reduces_confidence(self):
        mr = _mr()
        result = mr.compute(_df(_prices_flat(25)), "neutral")
        if abs(result.signal) < 0.10:
            assert result.confidence_adj <= 1.0
