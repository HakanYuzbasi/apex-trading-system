"""Tests for AdaptiveATRStops."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from risk.adaptive_atr_stops import AdaptiveATRStops


def _mgr(**kw) -> AdaptiveATRStops:
    defaults = dict(atr_period=5, min_stop_pct=0.004, max_stop_pct=0.12)
    defaults.update(kw)
    return AdaptiveATRStops(**defaults)


def _prices(n: int = 20, base: float = 100.0, drift: float = 0.001, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    vals = [base]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + rng.normal(drift, 0.01)))
    return pd.Series(vals)


def _stops(sl: float = 95.0, tp: float = 110.0, trail: float = 0.03, atr: float = 1.0) -> dict:
    return {"stop_loss": sl, "take_profit": tp, "trailing_stop_pct": trail, "atr": atr}


# ── compute_stop_distance ─────────────────────────────────────────────────────

class TestComputeStopDistance:

    def test_neutral_regime_smaller_than_trending(self):
        mgr = _mgr()
        neutral = mgr.compute_stop_distance(0.01, "neutral", 20.0)
        bull = mgr.compute_stop_distance(0.01, "bull", 20.0)
        assert neutral < bull

    def test_volatile_regime_wider_than_neutral(self):
        mgr = _mgr()
        neutral = mgr.compute_stop_distance(0.01, "neutral", 20.0)
        vol = mgr.compute_stop_distance(0.01, "volatile", 20.0)
        assert vol > neutral

    def test_high_vix_widens_stop(self):
        mgr = _mgr()
        low_vix = mgr.compute_stop_distance(0.01, "neutral", 15.0)
        high_vix = mgr.compute_stop_distance(0.01, "neutral", 50.0)
        assert high_vix > low_vix

    def test_profit_tightens_trailing(self):
        mgr = _mgr()
        flat = mgr.compute_stop_distance(0.01, "bull", 20.0, pnl_pct=0.0)
        profitable = mgr.compute_stop_distance(0.01, "bull", 20.0, pnl_pct=0.05)
        assert profitable < flat

    def test_large_profit_tighter_than_small_profit(self):
        mgr = _mgr()
        small = mgr.compute_stop_distance(0.01, "bull", 20.0, pnl_pct=0.02)
        large = mgr.compute_stop_distance(0.01, "bull", 20.0, pnl_pct=0.08)
        assert large < small

    def test_always_above_min(self):
        mgr = _mgr(min_stop_pct=0.004)
        dist = mgr.compute_stop_distance(0.0001, "neutral", 15.0, pnl_pct=0.10)
        assert dist >= 0.004

    def test_always_below_max(self):
        mgr = _mgr(max_stop_pct=0.12)
        dist = mgr.compute_stop_distance(0.50, "crisis", 80.0, pnl_pct=0.0)
        assert dist <= 0.12

    def test_unknown_regime_uses_default(self):
        mgr = _mgr()
        dist = mgr.compute_stop_distance(0.01, "sideways", 20.0)
        assert mgr.min_stop_pct <= dist <= mgr.max_stop_pct

    def test_zero_atr_pct_gives_min_stop(self):
        mgr = _mgr()
        dist = mgr.compute_stop_distance(0.0, "neutral", 20.0)
        assert dist == pytest.approx(mgr.min_stop_pct, abs=0.0001)

    def test_crisis_wider_than_volatile(self):
        mgr = _mgr()
        vol = mgr.compute_stop_distance(0.01, "volatile", 20.0)
        crisis = mgr.compute_stop_distance(0.01, "crisis", 20.0)
        assert crisis >= vol


class TestUpdateStop:

    def test_returns_dict(self):
        mgr = _mgr()
        result = mgr.update_stop("AAPL", _stops(), 100.0, _prices(), "neutral", 20.0, 0.0)
        assert isinstance(result, dict)

    def test_does_not_mutate_original(self):
        mgr = _mgr()
        orig = _stops(sl=95.0)
        mgr.update_stop("AAPL", orig, 100.0, _prices(), "neutral", 20.0, 0.05, is_long=True)
        assert orig["stop_loss"] == 95.0  # original unchanged

    def test_atr_updated(self):
        mgr = _mgr()
        result = mgr.update_stop("AAPL", _stops(atr=0.0), 100.0, _prices(20), "neutral", 20.0, 0.0)
        assert result["atr"] > 0.0

    def test_trailing_stop_pct_updated(self):
        mgr = _mgr()
        result = mgr.update_stop("AAPL", _stops(trail=0.99), 100.0, _prices(20), "bull", 20.0, 0.0)
        assert result["trailing_stop_pct"] < 0.99

    # ── Hard stop ratchet for LONG positions ──────────────────────────────────

    def test_long_profitable_stop_ratchets_up(self):
        """Profitable LONG: hard stop should move up (not down)."""
        mgr = _mgr()
        # Entry 90, current 100 → pnl=+11%. Existing stop at 88.
        result = mgr.update_stop("A", _stops(sl=88.0), 100.0, _prices(20), "bull", 20.0, 0.11, is_long=True)
        assert result["stop_loss"] > 88.0

    def test_long_losing_stop_not_moved(self):
        """Losing LONG: hard stop stays in place (no loosening)."""
        mgr = _mgr()
        result = mgr.update_stop("A", _stops(sl=95.0), 90.0, _prices(20), "neutral", 20.0, -0.05, is_long=True)
        assert result["stop_loss"] == pytest.approx(95.0)

    def test_long_stop_never_loosened_when_profitable(self):
        """If existing stop is already tight, don't loosen it even when profitable."""
        mgr = _mgr()
        # existing stop at 99 (very tight) — don't let it slip below
        result = mgr.update_stop("A", _stops(sl=99.0), 100.0, _prices(20), "neutral", 20.0, 0.05, is_long=True)
        # Should be >= existing stop (ratchet rule = max)
        assert result["stop_loss"] >= 99.0

    # ── Hard stop ratchet for SHORT positions ─────────────────────────────────

    def test_short_profitable_stop_ratchets_down(self):
        """Profitable SHORT (price fell): stop should move down (not up)."""
        mgr = _mgr()
        # Short entry 100, current 90 → pnl=+10%. Existing stop at 105.
        result = mgr.update_stop("B", _stops(sl=105.0), 90.0, _prices(20), "bull", 20.0, 0.10, is_long=False)
        assert result["stop_loss"] < 105.0

    def test_short_losing_stop_not_moved(self):
        """Losing SHORT: stop stays in place."""
        mgr = _mgr()
        result = mgr.update_stop("B", _stops(sl=105.0), 110.0, _prices(20), "neutral", 20.0, -0.05, is_long=False)
        assert result["stop_loss"] == pytest.approx(105.0)


class TestCalcATR:

    def test_atr_positive(self):
        mgr = _mgr()
        p = _prices(20)
        atr = mgr._calc_atr(p)
        assert atr > 0.0

    def test_atr_scales_with_volatility(self):
        mgr = _mgr()
        low_vol = _prices(20, drift=0.0)
        high_vol = pd.Series(np.linspace(100, 200, 20))   # deterministic large moves
        # high_vol has larger abs diffs → larger ATR
        # Actually let's make it more volatile
        rng = np.random.default_rng(0)
        vals = [100.0]
        for _ in range(19):
            vals.append(vals[-1] * (1 + rng.normal(0, 0.05)))  # 5% daily vol
        high_vol = pd.Series(vals)
        atr_high = mgr._calc_atr(high_vol)
        atr_low = mgr._calc_atr(low_vol)
        assert atr_high > atr_low

    def test_atr_short_series_fallback(self):
        mgr = _mgr(atr_period=14)
        p = pd.Series([100.0])  # only 1 bar
        atr = mgr._calc_atr(p)
        assert atr == pytest.approx(2.0, rel=0.01)  # 2% of 100


class TestRegimeHelpers:

    def test_trending_regimes(self):
        assert AdaptiveATRStops.regime_is_trending("strong_bull")
        assert AdaptiveATRStops.regime_is_trending("bull")
        assert AdaptiveATRStops.regime_is_trending("bear")
        assert AdaptiveATRStops.regime_is_trending("strong_bear")
        assert not AdaptiveATRStops.regime_is_trending("neutral")
        assert not AdaptiveATRStops.regime_is_trending("volatile")

    def test_mean_reverting_regime(self):
        assert AdaptiveATRStops.regime_is_mean_reverting("neutral")
        assert not AdaptiveATRStops.regime_is_mean_reverting("bull")
        assert not AdaptiveATRStops.regime_is_mean_reverting("volatile")


class TestUpdateAll:

    def _make_hdata(self, n: int = 20, seed: int = 0) -> dict:
        return {"Close": _prices(n, seed=seed)}

    def test_returns_dict(self):
        mgr = _mgr()
        positions = {"AAPL": 10, "MSFT": 5}
        stops = {
            "AAPL": _stops(sl=95.0),
            "MSFT": _stops(sl=180.0),
        }
        hdata = {
            "AAPL": self._make_hdata(seed=0),
            "MSFT": self._make_hdata(seed=1),
        }
        entry_prices = {"AAPL": 90.0, "MSFT": 175.0}
        result = mgr.update_all(positions, stops, entry_prices, hdata, "bull", 20.0)
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result

    def test_skips_zero_qty(self):
        mgr = _mgr()
        positions = {"AAPL": 0, "MSFT": 5}
        stops = {"AAPL": _stops(), "MSFT": _stops()}
        hdata = {"AAPL": self._make_hdata(), "MSFT": self._make_hdata(seed=1)}
        entry_prices = {"AAPL": 100.0, "MSFT": 100.0}
        result = mgr.update_all(positions, stops, entry_prices, hdata, "neutral", 20.0)
        assert "AAPL" not in result
        assert "MSFT" in result

    def test_skips_missing_stops(self):
        mgr = _mgr()
        positions = {"AAPL": 10}
        stops = {}  # no stops for AAPL
        hdata = {"AAPL": self._make_hdata()}
        result = mgr.update_all(positions, stops, {"AAPL": 100.0}, hdata, "neutral", 20.0)
        assert "AAPL" not in result

    def test_skips_missing_data(self):
        mgr = _mgr()
        positions = {"AAPL": 10}
        stops = {"AAPL": _stops()}
        hdata = {}  # no historical data
        result = mgr.update_all(positions, stops, {"AAPL": 100.0}, hdata, "neutral", 20.0)
        assert "AAPL" not in result

    def test_profit_ratchet_in_batch(self):
        """Profitable position should have its stop ratcheted up in batch mode."""
        mgr = _mgr()
        # Entry at 90, prices around 100+ → profitable
        prices = _prices(20, base=100.0)
        positions = {"AAPL": 10}
        stops = {"AAPL": _stops(sl=85.0)}
        hdata = {"AAPL": {"Close": prices}}
        entry_prices = {"AAPL": 90.0}
        result = mgr.update_all(positions, stops, entry_prices, hdata, "bull", 20.0)
        assert result["AAPL"]["stop_loss"] > 85.0


# ── min_stop_pct / max_stop_pct property access ───────────────────────────────

def test_min_stop_pct_accessible():
    mgr = AdaptiveATRStops(min_stop_pct=0.005)
    assert mgr.min_stop_pct == 0.005


def test_max_stop_pct_accessible():
    mgr = AdaptiveATRStops(max_stop_pct=0.10)
    assert mgr.max_stop_pct == 0.10
