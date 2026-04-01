"""
tests/test_iv_skew_signal.py — Unit tests for models/iv_skew_signal.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.iv_skew_signal import (
    IVSkewSignal,
    IVSkewState,
    compute_put_call_skew,
    compute_vix_term_signal,
    get_iv_skew_signal,
)


# ── compute_put_call_skew ─────────────────────────────────────────────────────

class TestComputePutCallSkew:
    def test_empty_returns_zero(self):
        assert compute_put_call_skew([], []) == 0.0
        assert compute_put_call_skew([0.2], []) == 0.0
        assert compute_put_call_skew([], [0.2]) == 0.0

    def test_higher_put_iv_returns_negative(self):
        # Heavy put demand (bearish skew)
        sig = compute_put_call_skew([0.40, 0.45], [0.20, 0.22])
        assert sig < 0.0

    def test_higher_call_iv_returns_positive(self):
        # Heavy call demand (bullish skew)
        sig = compute_put_call_skew([0.20, 0.22], [0.40, 0.45])
        assert sig > 0.0

    def test_equal_ivs_returns_near_zero(self):
        sig = compute_put_call_skew([0.25, 0.25], [0.25, 0.25])
        assert abs(sig) < 1e-6

    def test_output_bounded(self):
        sig = compute_put_call_skew([0.0001], [100.0])
        assert -1.0 <= sig <= 1.0
        sig = compute_put_call_skew([100.0], [0.0001])
        assert -1.0 <= sig <= 1.0

    def test_zero_ivs_filtered_out(self):
        # Zero IVs should be ignored
        sig = compute_put_call_skew([0.0, 0.3], [0.0, 0.4])
        assert isinstance(sig, float)

    def test_all_zero_ivs_returns_zero(self):
        sig = compute_put_call_skew([0.0, 0.0], [0.0, 0.0])
        assert sig == 0.0


# ── compute_vix_term_signal ───────────────────────────────────────────────────

class TestComputeVixTermSignal:
    def test_contango_positive(self):
        # VIX3M > VIX → normal term structure → positive
        sig = compute_vix_term_signal(vix_spot=18.0, vix_3m=22.0)
        assert sig > 0.0

    def test_backwardation_negative(self):
        # VIX3M < VIX → inverted → negative
        sig = compute_vix_term_signal(vix_spot=35.0, vix_3m=28.0)
        assert sig < 0.0

    def test_flat_term_near_zero(self):
        # Ratio ≈ 1.0 → ~zero
        sig = compute_vix_term_signal(vix_spot=20.0, vix_3m=20.0)
        assert abs(sig) < 1e-6

    def test_zero_vix_returns_zero(self):
        assert compute_vix_term_signal(0.0, 20.0) == 0.0
        assert compute_vix_term_signal(20.0, 0.0) == 0.0

    def test_output_bounded(self):
        # Extreme contango
        sig = compute_vix_term_signal(1.0, 1000.0)
        assert -1.0 <= sig <= 1.0
        # Extreme backwardation
        sig = compute_vix_term_signal(1000.0, 1.0)
        assert -1.0 <= sig <= 1.0

    def test_mild_contango_partial_positive(self):
        # VIX3M/VIX = 1.075 → midway to 1.15
        sig = compute_vix_term_signal(20.0, 21.5)
        assert 0.0 < sig < 1.0


# ── IVSkewSignal ──────────────────────────────────────────────────────────────

def _mock_no_options(symbol, n_strikes):
    return [], [], 0.0


def _mock_with_options(symbol, n_strikes):
    return [0.30, 0.32], [0.22, 0.24], 150.0


def _mock_vix_levels_contango():
    return 18.0, 22.0  # (vix_spot, vix_3m) → contango


def _mock_vix_levels_zero():
    return 0.0, 0.0


class TestIVSkewSignal:
    def test_get_signal_returns_float(self):
        gen = IVSkewSignal()
        with patch.object(gen, "_fetch_options_ivs", side_effect=_mock_no_options), \
             patch.object(gen, "_get_vix_levels", return_value=(18.0, 22.0)):
            sig = gen.get_signal("AAPL")
        assert isinstance(sig, float)

    def test_get_signal_bounded(self):
        gen = IVSkewSignal()
        with patch.object(gen, "_fetch_options_ivs", side_effect=_mock_with_options), \
             patch.object(gen, "_get_vix_levels", return_value=(18.0, 22.0)):
            sig = gen.get_signal("AAPL")
        assert -1.0 <= sig <= 1.0

    def test_heavy_put_skew_negative_signal(self):
        gen = IVSkewSignal()
        # Heavy put demand + backwardation (both bearish)
        with patch.object(gen, "_fetch_options_ivs", return_value=([0.45, 0.50], [0.20, 0.22], 100.0)), \
             patch.object(gen, "_get_vix_levels", return_value=(35.0, 28.0)):
            gen._symbol_cache.clear()
            sig = gen.get_signal("AAPL")
        assert sig < 0.0

    def test_heavy_call_skew_positive_signal(self):
        gen = IVSkewSignal()
        # Heavy call demand + contango (both bullish)
        with patch.object(gen, "_fetch_options_ivs", return_value=([0.20, 0.22], [0.45, 0.50], 100.0)), \
             patch.object(gen, "_get_vix_levels", return_value=(18.0, 23.0)):
            gen._symbol_cache.clear()
            sig = gen.get_signal("AAPL")
        assert sig > 0.0

    def test_crypto_returns_zero(self):
        gen = IVSkewSignal()
        sig = gen.get_signal("CRYPTO:BTC/USD")
        assert sig == 0.0

    def test_disabled_returns_zero(self, monkeypatch):
        import models.iv_skew_signal as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "IV_SKEW_ENABLED" else mod._DEF.get(k))
        gen = IVSkewSignal()
        assert gen.get_signal("AAPL") == 0.0

    def test_get_state_returns_iv_skew_state(self):
        gen = IVSkewSignal()
        with patch.object(gen, "_fetch_options_ivs", side_effect=_mock_no_options), \
             patch.object(gen, "_get_vix_levels", return_value=(18.0, 22.0)):
            state = gen.get_state("AAPL")
        assert isinstance(state, IVSkewState)

    def test_get_report_has_expected_keys(self):
        gen = IVSkewSignal()
        with patch.object(gen, "_fetch_options_ivs", side_effect=_mock_no_options), \
             patch.object(gen, "_get_vix_levels", return_value=(18.0, 22.0)):
            report = gen.get_report("AAPL")
        for key in ["symbol", "put_call_skew", "vix_term_signal", "combined_signal", "timestamp"]:
            assert key in report

    def test_caching_same_object(self):
        gen = IVSkewSignal()
        with patch.object(gen, "_fetch_options_ivs", side_effect=_mock_with_options), \
             patch.object(gen, "_get_vix_levels", return_value=(18.0, 22.0)):
            s1 = gen.get_state("AAPL")
            s2 = gen.get_state("AAPL")
        assert s1 is s2  # cached

    def test_different_symbols_independent(self):
        gen = IVSkewSignal()
        # AAPL: heavy put skew; MSFT: heavy call skew
        def _fetch(sym, n):
            if sym == "AAPL":
                return [0.45, 0.50], [0.20, 0.22], 100.0
            return [0.20, 0.22], [0.45, 0.50], 300.0

        with patch.object(gen, "_fetch_options_ivs", side_effect=_fetch), \
             patch.object(gen, "_get_vix_levels", return_value=(20.0, 20.0)):
            gen._symbol_cache.clear()
            sig_aapl = gen.get_signal("AAPL")
            gen._symbol_cache.clear()
            sig_msft = gen.get_signal("MSFT")
        assert sig_aapl < sig_msft

    def test_get_vix_term_signal_returns_float(self):
        gen = IVSkewSignal()
        with patch.object(gen, "_get_vix_levels", return_value=(18.0, 22.0)):
            sig = gen.get_vix_term_signal()
        assert isinstance(sig, float)

    def test_vix_fetch_failure_graceful(self):
        gen = IVSkewSignal()
        with patch.object(gen, "_fetch_options_ivs", side_effect=_mock_no_options), \
             patch.object(gen, "_get_vix_levels", return_value=(0.0, 0.0)):
            gen._symbol_cache.clear()
            sig = gen.get_signal("AAPL")
        assert isinstance(sig, float)

    def test_options_fetch_failure_graceful(self):
        gen = IVSkewSignal()
        with patch.object(gen, "_fetch_options_ivs", side_effect=RuntimeError("no options")), \
             patch.object(gen, "_get_vix_levels", return_value=(18.0, 22.0)):
            gen._symbol_cache.clear()
            sig = gen.get_signal("AAPL")
        assert isinstance(sig, float)

    def test_iv_skew_state_to_dict(self):
        s = IVSkewState(symbol="AAPL", put_call_skew=-0.3, vix_term_signal=0.5, combined_signal=0.1)
        d = s.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["combined_signal"] == pytest.approx(0.1)
        assert "timestamp" in d


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_iv_skew_signal_returns_instance(self):
        assert isinstance(get_iv_skew_signal(), IVSkewSignal)

    def test_singleton_same_object(self):
        a = get_iv_skew_signal()
        b = get_iv_skew_signal()
        assert a is b
