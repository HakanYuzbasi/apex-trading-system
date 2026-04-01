"""
tests/test_macro_cross_asset_signal.py — Unit tests for models/macro_cross_asset_signal.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.macro_cross_asset_signal import (
    MacroCrossAssetSignal,
    MacroState,
    _dxy_momentum_signal,
    _fetch_series,
    _vix_velocity_signal,
    _yield_curve_signal,
    get_macro_signal,
)


# ── _vix_velocity_signal ──────────────────────────────────────────────────────

class TestVixVelocitySignal:
    def test_rising_vix_returns_negative(self):
        vix = list(range(15, 30))  # rising
        sig = _vix_velocity_signal(vix, lookback=10)
        assert sig < 0.0

    def test_falling_vix_returns_positive(self):
        vix = list(range(30, 15, -1))  # falling
        sig = _vix_velocity_signal(vix, lookback=10)
        assert sig > 0.0

    def test_flat_vix_returns_near_zero(self):
        vix = [20.0] * 20
        sig = _vix_velocity_signal(vix, lookback=10)
        assert abs(sig) < 1e-6

    def test_insufficient_data_returns_zero(self):
        sig = _vix_velocity_signal([20.0, 21.0], lookback=10)
        assert sig == 0.0

    def test_output_bounded(self):
        vix = [10.0] + [100.0] * 10  # extreme spike
        sig = _vix_velocity_signal(vix, lookback=10)
        assert -1.0 <= sig <= 1.0

    def test_zero_start_returns_zero(self):
        vix = [0.0] * 15
        sig = _vix_velocity_signal(vix, lookback=10)
        assert sig == 0.0


# ── _yield_curve_signal ───────────────────────────────────────────────────────

class TestYieldCurveSignal:
    def test_positive_spread_returns_positive(self):
        spread = [1.5, 1.6, 1.7, 1.8, 1.9]  # steep curve
        sig = _yield_curve_signal(spread, lookback=5)
        assert sig > 0.0

    def test_inverted_spread_returns_negative(self):
        spread = [-0.5, -0.4, -0.6, -0.5]  # inverted
        sig = _yield_curve_signal(spread, lookback=4)
        assert sig < 0.0

    def test_zero_spread_returns_near_zero(self):
        spread = [0.0] * 10
        sig = _yield_curve_signal(spread, lookback=5)
        assert abs(sig) < 1e-6

    def test_empty_returns_zero(self):
        sig = _yield_curve_signal([], lookback=5)
        assert sig == 0.0

    def test_output_bounded(self):
        spread = [5.0] * 10  # extreme steep
        sig = _yield_curve_signal(spread, lookback=5)
        assert -1.0 <= sig <= 1.0

    def test_shorter_than_lookback_still_works(self):
        spread = [1.0, 1.2]
        sig = _yield_curve_signal(spread, lookback=20)
        assert isinstance(sig, float)


# ── _dxy_momentum_signal ──────────────────────────────────────────────────────

class TestDxyMomentumSignal:
    def test_rising_dxy_returns_negative(self):
        dxy = list(range(100, 110))  # USD strengthening
        sig = _dxy_momentum_signal(dxy, lookback=8)
        assert sig < 0.0

    def test_falling_dxy_returns_positive(self):
        dxy = list(range(110, 100, -1))  # USD weakening
        sig = _dxy_momentum_signal(dxy, lookback=8)
        assert sig > 0.0

    def test_flat_dxy_returns_near_zero(self):
        dxy = [104.0] * 25
        sig = _dxy_momentum_signal(dxy, lookback=20)
        assert abs(sig) < 1e-6

    def test_insufficient_data_returns_zero(self):
        sig = _dxy_momentum_signal([100.0, 101.0], lookback=20)
        assert sig == 0.0

    def test_output_bounded(self):
        dxy = [90.0] + [200.0] * 20  # extreme move
        sig = _dxy_momentum_signal(dxy, lookback=20)
        assert -1.0 <= sig <= 1.0

    def test_zero_start_returns_zero(self):
        dxy = [0.0] * 25
        sig = _dxy_momentum_signal(dxy, lookback=20)
        assert sig == 0.0


# ── MacroCrossAssetSignal ─────────────────────────────────────────────────────

def _patched_fetch(ticker, period="3mo", field="Close"):
    """Return synthetic data for each ticker."""
    if "VIX" in ticker.upper():
        # Falling VIX → risk-on
        return [25.0, 24.5, 24.0, 23.5, 23.0, 22.5, 22.0, 21.5, 21.0, 20.5, 20.0] * 2
    elif "TNX" in ticker.upper():
        # 10Y
        return [3.5, 3.6, 3.7, 3.8, 3.9] * 5
    elif "IRX" in ticker.upper():
        # 2Y (IRX = 3m, used as proxy)
        return [2.5, 2.5, 2.5, 2.5, 2.5] * 5
    elif "DX" in ticker.upper():
        # Flat DXY
        return [104.0] * 25
    return []


class TestMacroCrossAssetSignal:
    def test_get_signal_returns_float(self):
        gen = MacroCrossAssetSignal()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            sig = gen.get_signal()
        assert isinstance(sig, float)

    def test_get_signal_bounded(self):
        gen = MacroCrossAssetSignal()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            sig = gen.get_signal()
        assert -1.0 <= sig <= 1.0

    def test_get_report_has_expected_keys(self):
        gen = MacroCrossAssetSignal()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            report = gen.get_report()
        for key in ["vix_signal", "yield_signal", "dxy_signal", "composite_signal", "timestamp"]:
            assert key in report

    def test_falling_vix_gives_positive_signal(self):
        gen = MacroCrossAssetSignal()
        gen.invalidate_cache()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            state = gen.get_state()
        # Falling VIX → vix_signal > 0
        assert state.vix_signal > 0.0

    def test_positive_spread_gives_positive_yield_signal(self):
        gen = MacroCrossAssetSignal()
        gen.invalidate_cache()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            state = gen.get_state()
        # 10Y(3.8) > 2Y(2.5) → positive
        assert state.yield_signal > 0.0

    def test_caching_returns_same_object(self):
        gen = MacroCrossAssetSignal()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            s1 = gen.get_state()
            s2 = gen.get_state()
        assert s1 is s2

    def test_invalidate_cache_forces_recompute(self):
        gen = MacroCrossAssetSignal()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            s1 = gen.get_state()
            gen.invalidate_cache()
            s2 = gen.get_state()
        # Different objects after cache clear
        assert s1 is not s2

    def test_disabled_returns_zero(self, monkeypatch):
        import models.macro_cross_asset_signal as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "MACRO_ENABLED" else mod._DEF.get(k))
        gen = MacroCrossAssetSignal()
        assert gen.get_signal() == 0.0

    def test_fetch_failure_returns_neutral_state(self):
        gen = MacroCrossAssetSignal()
        with patch("models.macro_cross_asset_signal._fetch_series", return_value=[]):
            gen.invalidate_cache()
            sig = gen.get_signal()
        assert isinstance(sig, float)

    def test_get_state_returns_macro_state(self):
        gen = MacroCrossAssetSignal()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            state = gen.get_state()
        assert isinstance(state, MacroState)

    def test_composite_combines_sub_signals(self):
        gen = MacroCrossAssetSignal()
        gen.invalidate_cache()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            state = gen.get_state()
        # With falling VIX (+), positive spread (+), flat DXY (0) → composite should be positive
        assert state.composite_signal > 0.0

    def test_error_state_has_error_field(self):
        gen = MacroCrossAssetSignal()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=RuntimeError("boom")):
            gen.invalidate_cache()
            state = gen.get_state()
        assert state.error is not None

    def test_macro_state_to_dict(self):
        s = MacroState(vix_signal=0.5, yield_signal=0.3, dxy_signal=-0.1, composite_signal=0.25)
        d = s.to_dict()
        assert d["composite_signal"] == pytest.approx(0.25)
        assert "timestamp" in d

    def test_vix_level_populated(self):
        gen = MacroCrossAssetSignal()
        gen.invalidate_cache()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            state = gen.get_state()
        assert state.vix_level > 0.0

    def test_yield_spread_populated(self):
        gen = MacroCrossAssetSignal()
        gen.invalidate_cache()
        with patch("models.macro_cross_asset_signal._fetch_series", side_effect=_patched_fetch):
            state = gen.get_state()
        # 10Y ≈ 3.9, 2Y ≈ 2.5 → spread ≈ 1.4
        assert state.yield_spread > 0.0


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_macro_signal_returns_instance(self):
        inst = get_macro_signal()
        assert isinstance(inst, MacroCrossAssetSignal)

    def test_singleton_same_object(self):
        a = get_macro_signal()
        b = get_macro_signal()
        assert a is b
