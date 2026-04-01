"""
tests/test_bear_regime_strategies.py — Unit tests for models/bear_regime_strategies.py
"""
import numpy as np
import pandas as pd
import pytest
from models.bear_regime_strategies import (
    MeanReversionSignal,
    SectorRotationSignal,
    get_bear_regime_blend,
    _rsi,
    _vwap_zscore,
    _momentum,
)


# ── _rsi ─────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_all_gains_returns_100(self):
        closes = np.linspace(100, 120, 20)
        assert _rsi(closes) == pytest.approx(100.0)

    def test_all_losses_returns_near_zero(self):
        closes = np.linspace(120, 100, 20)
        rsi = _rsi(closes)
        assert rsi < 10.0

    def test_insufficient_data_returns_50(self):
        assert _rsi(np.array([100, 101, 102]), period=14) == pytest.approx(50.0)

    def test_balanced_returns_near_50(self):
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.choice([-1, 1], size=50))
        rsi = _rsi(closes)
        assert 30 < rsi < 70

    def test_returns_float(self):
        closes = np.linspace(100, 110, 20)
        assert isinstance(_rsi(closes), float)


# ── _vwap_zscore ─────────────────────────────────────────────────────────────

class TestVwapZscore:
    def _make_df(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        df = pd.DataFrame({
            "Open": closes,
            "High": closes + rng.uniform(0, 1, n),
            "Low":  closes - rng.uniform(0, 1, n),
            "Close": closes,
            "Volume": rng.integers(1000, 10000, n).astype(float),
        })
        return df

    def test_returns_float(self):
        df = self._make_df()
        assert isinstance(_vwap_zscore(df), float)

    def test_insufficient_data_returns_zero(self):
        df = self._make_df(n=5)
        assert _vwap_zscore(df, period=20) == pytest.approx(0.0)

    def test_no_volume_uses_price_fallback(self):
        rng = np.random.default_rng(1)
        closes = 100.0 + np.cumsum(rng.normal(0, 0.5, 30))
        df = pd.DataFrame({"Close": closes, "High": closes, "Low": closes})
        z = _vwap_zscore(df)
        assert isinstance(z, float)

    def test_price_far_above_mean_gives_positive_z(self):
        closes = np.ones(25) * 100.0
        closes[-1] = 200.0  # huge outlier above
        df = pd.DataFrame({"Close": closes, "High": closes, "Low": closes, "Volume": np.ones(25) * 1000})
        assert _vwap_zscore(df) > 1.0


# ── _momentum ────────────────────────────────────────────────────────────────

class TestMomentum:
    def test_positive_trend(self):
        closes = np.linspace(100, 120, 25)
        assert _momentum(closes, 20) > 0

    def test_negative_trend(self):
        closes = np.linspace(120, 100, 25)
        assert _momentum(closes, 20) < 0

    def test_insufficient_data_returns_zero(self):
        assert _momentum(np.array([100, 105, 110]), 20) == pytest.approx(0.0)


# ── MeanReversionSignal ───────────────────────────────────────────────────────

class TestMeanReversionSignal:
    def _make_df(self, n: int = 40, trend: float = 0.0, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        closes = 100.0 + np.cumsum(rng.normal(trend, 0.5, n))
        return pd.DataFrame({"Close": closes, "High": closes + 0.5, "Low": closes - 0.5, "Volume": np.ones(n) * 1000})

    def test_neutral_regime_can_fire(self):
        mr = MeanReversionSignal()
        df = self._make_df()
        sig = mr.get_signal("AAPL", df, regime="neutral")
        assert isinstance(sig, float)
        assert -1.0 <= sig <= 1.0

    def test_bull_regime_returns_zero(self):
        mr = MeanReversionSignal()
        df = self._make_df()
        assert mr.get_signal("AAPL", df, regime="bull") == pytest.approx(0.0)
        assert mr.get_signal("AAPL", df, regime="strong_bull") == pytest.approx(0.0)

    def test_bear_regime_active(self):
        mr = MeanReversionSignal()
        df = self._make_df()
        sig = mr.get_signal("AAPL", df, regime="bear")
        assert isinstance(sig, float)

    def test_none_df_returns_zero(self):
        mr = MeanReversionSignal()
        assert mr.get_signal("AAPL", None, regime="bear") == pytest.approx(0.0)

    def test_insufficient_data_returns_zero(self):
        mr = MeanReversionSignal()
        df = pd.DataFrame({"Close": [100.0, 101.0, 100.5]})
        assert mr.get_signal("AAPL", df, regime="bear") == pytest.approx(0.0)

    def test_signal_clipped_to_unit_range(self):
        mr = MeanReversionSignal()
        df = self._make_df(n=40)
        for regime in ["bear", "strong_bear", "neutral", "volatile"]:
            sig = mr.get_signal("X", df, regime=regime)
            assert -1.0 <= sig <= 1.0

    def test_strong_bear_regime_active(self):
        mr = MeanReversionSignal()
        df = self._make_df()
        sig = mr.get_signal("AAPL", df, regime="strong_bear")
        assert isinstance(sig, float)


# ── SectorRotationSignal ─────────────────────────────────────────────────────

def _sector_data(n: int = 30, seed: int = 0, trend: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(trend, 0.3, n))
    return pd.DataFrame({"Close": closes})


class TestSectorRotationSignal:
    def _make_hist(self, trends: dict) -> dict:
        return {etf: _sector_data(n=30, seed=i, trend=t) for i, (etf, t) in enumerate(trends.items())}

    def test_known_top_sector_positive_signal(self):
        sr = SectorRotationSignal()
        # XLK is the best performer
        hist = self._make_hist({
            "XLK": 0.5, "XLF": 0.1, "XLE": -0.1, "XLV": 0.0, "XLI": 0.0,
            "XLC": 0.0, "XLU": -0.1, "XLB": 0.0, "XLRE": 0.0, "XLP": 0.0, "XLY": 0.0,
        })
        sig = sr.get_signal("NVDA", hist, regime="bear")  # NVDA → XLK
        assert sig > 0.0

    def test_bottom_sector_negative_signal(self):
        sr = SectorRotationSignal()
        # XLE is the worst performer
        hist = self._make_hist({
            "XLK": 0.5, "XLF": 0.1, "XLE": -0.5, "XLV": 0.2, "XLI": 0.1,
            "XLC": 0.2, "XLU": 0.0, "XLB": 0.1, "XLRE": 0.1, "XLP": 0.1, "XLY": 0.2,
        })
        sig = sr.get_signal("XOM", hist, regime="bear")  # XOM → XLE
        assert sig < 0.0

    def test_unknown_symbol_returns_zero(self):
        sr = SectorRotationSignal()
        hist = self._make_hist({"XLK": 0.5, "XLF": 0.1})
        sig = sr.get_signal("BTC-USD", hist, regime="neutral")
        assert sig == pytest.approx(0.0)

    def test_spy_returns_zero(self):
        sr = SectorRotationSignal()
        hist = self._make_hist({"XLK": 0.5, "XLF": 0.1, "XLE": -0.1})
        assert sr.get_signal("SPY", hist, regime="bear") == pytest.approx(0.0)

    def test_too_few_sectors_returns_zero(self):
        sr = SectorRotationSignal()
        hist = {"XLK": _sector_data()}  # only 1 sector
        assert sr.get_signal("NVDA", hist, regime="neutral") == pytest.approx(0.0)

    def test_signal_clipped(self):
        sr = SectorRotationSignal()
        hist = self._make_hist({
            "XLK": 0.5, "XLF": 0.1, "XLE": -0.3, "XLV": 0.0, "XLI": 0.1,
            "XLC": 0.0, "XLU": 0.0, "XLB": 0.0, "XLRE": 0.0, "XLP": 0.0, "XLY": 0.0,
        })
        sig = sr.get_signal("NVDA", hist, regime="bull")
        assert -1.0 <= sig <= 1.0


# ── get_bear_regime_blend ────────────────────────────────────────────────────

class TestGetBearRegimeBlend:
    def _df(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        return pd.DataFrame({"Close": closes, "High": closes + 0.5, "Low": closes - 0.5, "Volume": np.ones(n) * 1000})

    def _hist(self):
        sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLU", "XLB", "XLRE", "XLP", "XLY"]
        return {s: _sector_data(n=30, seed=i) for i, s in enumerate(sectors)}

    def test_returns_tuple_of_float_and_dict(self):
        blended, components = get_bear_regime_blend("AAPL", self._df(), self._hist(), "neutral", 0.20)
        assert isinstance(blended, float)
        assert isinstance(components, dict)

    def test_blended_signal_in_range(self):
        for regime in ["bull", "bear", "neutral", "volatile"]:
            blended, _ = get_bear_regime_blend("AAPL", self._df(), self._hist(), regime, 0.15)
            assert -1.0 <= blended <= 1.0

    def test_bull_regime_signal_close_to_base(self):
        """In bull regime, MR is suppressed → blended close to base signal."""
        blended, components = get_bear_regime_blend("AAPL", self._df(), self._hist(), "bull", 0.20)
        # MR component should be 0 in bull
        assert components["mean_reversion"] == pytest.approx(0.0)

    def test_components_has_expected_keys(self):
        _, components = get_bear_regime_blend("AAPL", self._df(), self._hist(), "bear", 0.15)
        assert "mean_reversion" in components
        assert "sector_rotation" in components
        assert "blended" in components

    def test_empty_hist_does_not_raise(self):
        blended, _ = get_bear_regime_blend("AAPL", self._df(), {}, "bear", 0.20)
        assert isinstance(blended, float)
