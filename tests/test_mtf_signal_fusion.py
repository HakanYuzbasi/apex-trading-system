"""Tests for MTFSignalFuser."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.mtf_signal_fusion import MTFSignalFuser, MTFFusedSignal


def _make_df(closes: list[float], add_volume: bool = True) -> pd.DataFrame:
    data = {"Close": closes, "Open": closes, "High": closes, "Low": closes}
    if add_volume:
        data["Volume"] = [1_000_000] * len(closes)
    return pd.DataFrame(data)


def _rising(n: int = 20, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    return _make_df([start + i * step for i in range(n)])


def _falling(n: int = 20, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    return _make_df([start - i * step for i in range(n)])


def _flat(n: int = 20, price: float = 100.0) -> pd.DataFrame:
    return _make_df([price] * n)


class TestFuseNoHigherTF:

    def test_no_data_returns_daily_signal(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(daily_signal=0.30, hourly_df=None, fivemin_df=None)
        assert result.signal == pytest.approx(0.30, abs=0.01)

    def test_no_data_confidence_adj_one(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(daily_signal=0.30, hourly_df=None, fivemin_df=None)
        assert result.confidence_adj == pytest.approx(1.0, abs=0.01)

    def test_too_few_bars_falls_back_to_daily(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(
            daily_signal=0.25,
            hourly_df=_make_df([100.0] * 3),   # < MIN_1H_BARS=5
            fivemin_df=_make_df([100.0] * 2),  # < MIN_5M_BARS=6
        )
        assert result.signal == pytest.approx(0.25, abs=0.01)


class TestFuseReturnType:

    def test_returns_mtf_fused_signal(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.2, _rising(20), _rising(15), regime="bull")
        assert isinstance(result, MTFFusedSignal)

    def test_signal_bounded(self):
        fuser = MTFSignalFuser()
        for _ in range(10):
            result = fuser.fuse(
                daily_signal=np.random.uniform(-1, 1),
                hourly_df=_rising(20),
                fivemin_df=_rising(15),
                regime="neutral",
            )
            assert -1.0 <= result.signal <= 1.0

    def test_confidence_adj_bounded(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.3, _rising(20), _rising(15), regime="bull")
        assert 0.75 <= result.confidence_adj <= 1.15

    def test_weights_sum_to_one(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.3, _rising(20), _rising(15), regime="bull")
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_tf_signals_dict_has_three_keys(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.3, _rising(20), _rising(15))
        assert set(result.tf_signals.keys()) == {"5m", "1h", "daily"}


class TestAlignment:

    def test_all_rising_is_aligned(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(
            daily_signal=0.40,
            hourly_df=_rising(20),
            fivemin_df=_rising(15),
            regime="bull",
        )
        assert result.aligned is True

    def test_opposing_signals_not_aligned(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(
            daily_signal=0.50,    # bullish daily
            hourly_df=_falling(20),  # bearish hourly
            fivemin_df=_falling(15),
            regime="neutral",
        )
        assert result.aligned is False

    def test_aligned_boosts_confidence(self):
        fuser = MTFSignalFuser()
        aligned_result = fuser.fuse(0.30, _rising(20), _rising(15), regime="bull")
        assert aligned_result.confidence_adj > 1.0

    def test_misaligned_reduces_confidence(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.50, _falling(20), _falling(15), regime="bull")
        assert result.confidence_adj < 1.0


class TestRegimeWeights:

    def test_crisis_weights_daily_heavy(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.20, _rising(20), _rising(15), regime="crisis")
        assert result.weights["daily"] >= 0.60

    def test_trending_weights_hourly_heavy(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.20, _rising(20), _rising(15), regime="bull")
        assert result.weights["1h"] >= 0.40

    def test_neutral_balanced_weights(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.20, _rising(20), _rising(15), regime="neutral")
        # 5m and 1h should both be ≥ 0.30
        assert result.weights["5m"] >= 0.30
        assert result.weights["1h"] >= 0.30

    def test_unknown_regime_uses_default(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.20, _rising(20), _rising(15), regime="unknown_xyz")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=0.01)


class TestDominantTF:

    def test_dominant_tf_is_valid_key(self):
        fuser = MTFSignalFuser()
        result = fuser.fuse(0.30, _rising(20), _rising(15))
        assert result.dominant_tf in ("5m", "1h", "daily")


class TestHourlySignal:

    def test_rising_hourly_positive(self):
        fuser = MTFSignalFuser()
        sig = fuser._hourly_signal(_rising(20))
        assert sig is not None
        assert sig > 0

    def test_falling_hourly_negative(self):
        fuser = MTFSignalFuser()
        sig = fuser._hourly_signal(_falling(20))
        assert sig is not None
        assert sig < 0

    def test_flat_hourly_near_zero(self):
        fuser = MTFSignalFuser()
        sig = fuser._hourly_signal(_flat(20))
        assert sig is not None
        assert abs(sig) < 0.3

    def test_too_few_bars_returns_none(self):
        fuser = MTFSignalFuser()
        assert fuser._hourly_signal(_make_df([100.0] * 3)) is None


class TestFiveminSignal:

    def test_rising_5m_positive(self):
        fuser = MTFSignalFuser()
        sig = fuser._fivemin_signal(_rising(20, step=0.1))
        assert sig is not None
        assert sig > 0

    def test_falling_5m_negative(self):
        fuser = MTFSignalFuser()
        sig = fuser._fivemin_signal(_falling(20, step=0.1))
        assert sig is not None
        assert sig < 0

    def test_too_few_bars_returns_none(self):
        fuser = MTFSignalFuser()
        assert fuser._fivemin_signal(_make_df([100.0] * 4)) is None

    def test_signal_bounded(self):
        fuser = MTFSignalFuser()
        sig = fuser._fivemin_signal(_rising(20, step=5.0))  # large moves
        assert sig is not None
        assert -1.0 <= sig <= 1.0


class TestEMA:

    def test_ema_same_length_as_input(self):
        fuser = MTFSignalFuser()
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        ema = fuser._ema(prices, 3)
        assert len(ema) == len(prices)

    def test_ema_rising_above_price_mean_for_flat(self):
        fuser = MTFSignalFuser()
        prices = np.array([100.0] * 10)
        ema = fuser._ema(prices, 3)
        assert ema[-1] == pytest.approx(100.0, abs=0.01)


class TestRSI:

    def test_all_gains_returns_100(self):
        fuser = MTFSignalFuser()
        prices = np.array([100.0 + i for i in range(20)])
        rsi = fuser._rsi(prices, 14)
        assert rsi == 100.0

    def test_all_losses_returns_zero(self):
        fuser = MTFSignalFuser()
        prices = np.array([100.0 - i for i in range(20)])
        rsi = fuser._rsi(prices, 14)
        assert rsi == pytest.approx(0.0, abs=0.01)

    def test_flat_returns_fifty(self):
        fuser = MTFSignalFuser()
        prices = np.array([100.0] * 20)
        rsi = fuser._rsi(prices, 14)
        assert rsi == pytest.approx(50.0, abs=5.0)

    def test_too_few_bars_returns_50(self):
        fuser = MTFSignalFuser()
        prices = np.array([100.0, 101.0])
        rsi = fuser._rsi(prices, 14)
        assert rsi == 50.0
