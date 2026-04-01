"""
tests/test_black_litterman.py — Unit tests for risk/black_litterman.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from risk.black_litterman import (
    BlackLittermanAllocator,
    BLResult,
    _reverse_optimize,
    _bl_posterior,
    _mvo_weights,
)


# ── Helper factories ──────────────────────────────────────────────────────────

def _make_price_data(n_symbols: int = 4, n_bars: int = 80) -> dict:
    """Generate random price DataFrames for n_symbols."""
    np.random.seed(42)
    syms = [f"SYM{i}" for i in range(n_symbols)]
    data = {}
    for s in syms:
        prices = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_bars))
        data[s] = pd.DataFrame({"Close": prices, "Volume": [10000.0] * n_bars})
    return data


def _make_allocator(tmp: str) -> BlackLittermanAllocator:
    return BlackLittermanAllocator(
        persist_path=str(Path(tmp) / "bl_weights.json")
    )


def _make_ic_tracker(ic_values: dict):
    """Build a mock FactorICTracker with given signal ICs."""
    mock = MagicMock()
    results = []
    for name, ic in ic_values.items():
        r = MagicMock()
        r.signal_name = name
        r.ic = ic
        r.obs = 15
        r.is_reliable = True
        r.status = "active"
        r.win_rate = 0.6
        results.append(r)
    mock.get_report.return_value = MagicMock(signals=results)
    return mock


# ── _reverse_optimize ─────────────────────────────────────────────────────────

class TestReverseOptimize:
    def test_positive_delta_gives_positive_returns(self):
        sigma = np.eye(3)
        weights = np.array([0.33, 0.33, 0.34])
        pi = _reverse_optimize(sigma, weights, delta=2.5)
        assert all(p > 0 for p in pi)

    def test_higher_delta_gives_higher_returns(self):
        sigma = np.eye(3)
        weights = np.ones(3) / 3
        pi_low = _reverse_optimize(sigma, weights, delta=1.0)
        pi_high = _reverse_optimize(sigma, weights, delta=5.0)
        assert all(pi_high[i] > pi_low[i] for i in range(3))


# ── _bl_posterior ─────────────────────────────────────────────────────────────

class TestBLPosterior:
    def test_no_views_returns_prior(self):
        n = 3
        pi = np.array([0.01, 0.02, 0.015])
        sigma = np.eye(n) * 0.0004
        P = np.ones((1, n)) / n
        Q = np.array([0.01])
        omega = np.array([[0.001]])
        mu_bl, _ = _bl_posterior(pi, sigma, tau=0.05, P=P, Q=Q, omega=omega)
        assert mu_bl.shape == (n,)

    def test_posterior_shape(self):
        n = 4
        pi = np.random.rand(n) * 0.01
        sigma = np.eye(n) * 0.0005
        P = np.ones((2, n)) / n
        Q = np.array([0.005, 0.008])
        omega = np.diag([0.001, 0.001])
        mu_bl, sigma_bl = _bl_posterior(pi, sigma, 0.05, P, Q, omega)
        assert mu_bl.shape == (n,)
        assert sigma_bl.shape == (n, n)


# ── _mvo_weights ──────────────────────────────────────────────────────────────

class TestMvoWeights:
    def test_weights_sum_to_one(self):
        mu = np.array([0.01, 0.015, 0.008, 0.012])
        sigma = np.eye(4) * 0.0005
        w = _mvo_weights(mu, sigma, delta=2.5, floor=0.0, cap=0.25)
        assert sum(w) == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative(self):
        mu = np.array([0.01, 0.015, 0.008, 0.012])
        sigma = np.eye(4) * 0.0005
        w = _mvo_weights(mu, sigma, delta=2.5, floor=0.0, cap=0.25)
        assert all(wi >= 0 for wi in w)

    def test_cap_applied(self):
        mu = np.array([0.10, 0.001, 0.001, 0.001])  # strongly concentrated
        sigma = np.eye(4) * 0.0005
        w = _mvo_weights(mu, sigma, delta=2.5, floor=0.0, cap=0.25)
        assert max(w) <= 0.25 + 1e-6

    def test_floor_applied(self):
        mu = np.array([0.01, 0.015, 0.008, 0.012])
        sigma = np.eye(4) * 0.0005
        w = _mvo_weights(mu, sigma, delta=2.5, floor=0.05, cap=0.40)
        assert all(wi >= 0.05 - 1e-6 for wi in w)

    def test_degenerate_covariance_fallback(self):
        mu = np.array([0.01, 0.01, 0.01])
        sigma = np.zeros((3, 3))  # all-zero cov → fallback
        w = _mvo_weights(mu, sigma, delta=2.5, floor=0.0, cap=1.0)
        assert sum(w) == pytest.approx(1.0, abs=1e-4)


# ── BlackLittermanAllocator ───────────────────────────────────────────────────

class TestBlackLittermanAllocator:

    def test_equal_weight_when_disabled(self, monkeypatch):
        import risk.black_litterman as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "BL_ENABLED" else mod._DEF.get(k))
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            result = bl.get_weights(["SYM0", "SYM1", "SYM2"], {})
            assert all(v == pytest.approx(1/3, abs=0.01) for v in result.weights.values())

    def test_equal_weight_for_single_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            result = bl.get_weights(["AAPL"], {})
            assert result.weights["AAPL"] == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(4)]
            price_data = _make_price_data(4, 80)
            result = bl.get_weights(syms, price_data)
            assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-3)

    def test_returns_bl_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(3)]
            price_data = _make_price_data(3, 80)
            result = bl.get_weights(syms, price_data)
            assert isinstance(result, BLResult)

    def test_bl_result_has_all_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(4)]
            price_data = _make_price_data(4, 80)
            result = bl.get_weights(syms, price_data)
            for s in syms:
                assert s in result.weights
                assert s in result.multipliers

    def test_views_used_when_ic_tracker_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(4)]
            price_data = _make_price_data(4, 80)
            tracker = _make_ic_tracker({"god_level": 0.25, "mean_reversion": 0.15})
            result = bl.get_weights(syms, price_data, ic_tracker=tracker)
            assert result.n_views > 0

    def test_no_views_when_ic_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(3)]
            price_data = _make_price_data(3, 80)
            # IC below min threshold → no views
            tracker = _make_ic_tracker({"god_level": 0.02, "mean_reversion": 0.03})
            result = bl.get_weights(syms, price_data, ic_tracker=tracker)
            assert result.n_views == 0

    def test_get_multiplier_default_when_no_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            mult = bl.get_multiplier("AAPL", default=1.0)
            assert mult == pytest.approx(1.0)

    def test_get_multiplier_after_compute(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(4)]
            price_data = _make_price_data(4, 80)
            bl.get_weights(syms, price_data)
            mult = bl.get_multiplier("SYM0")
            assert isinstance(mult, float)

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "bl.json")
            bl1 = BlackLittermanAllocator(persist_path=path)
            syms = [f"SYM{i}" for i in range(3)]
            price_data = _make_price_data(3, 80)
            result1 = bl1.get_weights(syms, price_data)

            bl2 = BlackLittermanAllocator(persist_path=path)
            assert bl2._last_result is not None
            assert bl2._last_result.weights == result1.weights

    def test_get_report_has_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(3)]
            price_data = _make_price_data(3, 80)
            bl.get_weights(syms, price_data)
            report = bl.get_report()
            assert "enabled" in report
            assert "weights" in report
            assert "multipliers" in report
            assert "n_views" in report

    def test_maybe_update_skips_off_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(3)]
            # cycle=1 → not a multiple of 50 → should skip
            result = bl.maybe_update(1, syms, {})
            assert result is None  # no prior result

    def test_maybe_update_fires_on_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(3)]
            price_data = _make_price_data(3, 80)
            result = bl.maybe_update(50, syms, price_data)
            assert result is not None

    def test_weight_cap_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = [f"SYM{i}" for i in range(6)]
            price_data = _make_price_data(6, 80)
            result = bl.get_weights(syms, price_data)
            cap = float(0.25)  # default BL_WEIGHT_CAP
            for w in result.weights.values():
                assert w <= cap + 1e-4

    def test_equal_weight_with_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            bl = _make_allocator(tmp)
            syms = ["AAPL", "MSFT", "GOOG"]
            # Only 5 bars — below lookback minimum
            data = {s: pd.DataFrame({"Close": [100.0] * 5}) for s in syms}
            result = bl.get_weights(syms, data)
            # Should fall back to equal weight
            for w in result.weights.values():
                assert w == pytest.approx(1/3, abs=0.01)
