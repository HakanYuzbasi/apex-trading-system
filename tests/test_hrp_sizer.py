"""Tests for HRPSizer."""
from __future__ import annotations

import math
import numpy as np
import pytest

from risk.hrp_sizer import HRPSizer


def _sizer(**kwargs) -> HRPSizer:
    defaults = dict(min_history=10, dampen_floor=0.50)
    defaults.update(kwargs)
    return HRPSizer(**defaults)


def _returns(n: int = 20, mu: float = 0.001, sigma: float = 0.01, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.normal(mu, sigma, n))


def _correlated_returns(base: list[float], noise: float = 0.002) -> list[float]:
    rng = np.random.default_rng(42)
    return [b + rng.normal(0, noise) for b in base]


class TestNoPositions:

    def test_no_existing_positions_returns_one(self):
        sizer = _sizer()
        mult = sizer.get_size_multiplier("AAPL", {}, {"AAPL": _returns()})
        assert mult == pytest.approx(1.0)

    def test_single_position_returns_one(self):
        sizer = _sizer()
        mult = sizer.get_size_multiplier(
            "AAPL",
            {"MSFT": 10},
            {"AAPL": _returns(20, seed=1), "MSFT": _returns(20, seed=2)},
        )
        # Only 1 existing position → 1 pair total: should compute
        assert isinstance(mult, float)
        assert 0.5 <= mult <= 1.0

    def test_insufficient_history_returns_one(self):
        sizer = _sizer(min_history=20)
        mult = sizer.get_size_multiplier(
            "AAPL",
            {"MSFT": 10},
            {"AAPL": _returns(5), "MSFT": _returns(5)},  # < 20 bars
        )
        assert mult == pytest.approx(1.0)


class TestCorrelation:

    def test_uncorrelated_positions_no_dampening(self):
        sizer = _sizer()
        base = _returns(30, seed=0)
        # Completely independent returns
        rets = {
            "AAPL": _returns(30, seed=1),
            "MSFT": _returns(30, seed=2),
            "GOOGL": _returns(30, seed=3),
        }
        positions = {"MSFT": 10, "GOOGL": 5}
        mult = sizer.get_size_multiplier("AAPL", positions, rets)
        assert mult >= 0.90  # uncorrelated → little to no dampening

    def test_highly_correlated_positions_dampened(self):
        sizer = _sizer()
        base = _returns(30, seed=7)
        rets = {
            "A": base,
            "B": _correlated_returns(base, noise=0.0001),  # nearly identical
            "C": _correlated_returns(base, noise=0.0001),
        }
        positions = {"B": 10, "C": 5}
        mult = sizer.get_size_multiplier("A", positions, rets)
        assert mult < 1.0  # highly correlated → should be dampened

    def test_multiplier_bounded_by_floor(self):
        sizer = _sizer(dampen_floor=0.50)
        base = _returns(30, seed=9)
        rets = {
            "X": base,
            "Y": _correlated_returns(base, noise=1e-6),
            "Z": _correlated_returns(base, noise=1e-6),
        }
        positions = {"Y": 10, "Z": 10}
        mult = sizer.get_size_multiplier("X", positions, rets)
        assert mult >= 0.50

    def test_multiplier_at_most_one(self):
        sizer = _sizer()
        rets = {s: _returns(30, seed=i) for i, s in enumerate(["A", "B", "C", "D"])}
        positions = {"B": 10, "C": 5, "D": 3}
        mult = sizer.get_size_multiplier("A", positions, rets)
        assert mult <= 1.0


class TestPortfolioWeights:

    def test_returns_none_insufficient_data(self):
        sizer = _sizer(min_history=20)
        weights = sizer.get_portfolio_weights(["A", "B"], {"A": _returns(5), "B": _returns(5)})
        assert weights is None

    def test_weights_sum_to_one(self):
        sizer = _sizer()
        syms = ["A", "B", "C"]
        rets = {s: _returns(20, seed=i) for i, s in enumerate(syms)}
        weights = sizer.get_portfolio_weights(syms, rets)
        assert weights is not None
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_weights_all_positive(self):
        sizer = _sizer()
        syms = ["A", "B", "C", "D"]
        rets = {s: _returns(20, seed=i) for i, s in enumerate(syms)}
        weights = sizer.get_portfolio_weights(syms, rets)
        assert weights is not None
        assert all(w >= 0 for w in weights.values())

    def test_correlated_pair_lower_weight(self):
        sizer = _sizer()
        base = _returns(30, seed=5)
        rets = {
            "A": _returns(30, seed=1),       # independent
            "B": base,                         # correlated pair
            "C": _correlated_returns(base),    # correlated pair
        }
        weights = sizer.get_portfolio_weights(["A", "B", "C"], rets)
        assert weights is not None
        # A should get weight comparable to B+C combined (B,C eat into each other)
        assert weights["B"] + weights["C"] <= 0.75  # cluster dampened


class TestHRPAlgorithm:

    def test_two_symbols_valid_weights(self):
        sizer = _sizer()
        rets = {"A": _returns(20, seed=0), "B": _returns(20, seed=1)}
        w = sizer.get_portfolio_weights(["A", "B"], rets)
        assert w is not None
        assert sum(w.values()) == pytest.approx(1.0, abs=0.001)

    def test_identical_returns_equal_weights(self):
        sizer = _sizer()
        base = _returns(20, seed=3)
        rets = {"A": base[:], "B": base[:]}
        w = sizer.get_portfolio_weights(["A", "B"], rets)
        assert w is not None
        # Identical series → equal weights
        assert abs(w["A"] - w["B"]) < 0.10

    def test_quasi_diag_returns_all_indices(self):
        sizer = _sizer()
        dist = np.array([[0, 0.5, 0.9], [0.5, 0, 0.4], [0.9, 0.4, 0]])
        order = sizer._quasi_diag(dist)
        assert sorted(order) == [0, 1, 2]

    def test_cluster_var_single_element(self):
        sizer = _sizer()
        cov = np.array([[0.01, 0.005], [0.005, 0.02]])
        var = sizer._cluster_var(cov, [0])
        assert var == pytest.approx(0.01, abs=0.001)

    def test_cov_to_corr_diagonal_is_one(self):
        sizer = _sizer()
        cov = np.array([[0.01, 0.003], [0.003, 0.04]])
        corr = sizer._cov_to_corr(cov)
        assert corr[0, 0] == pytest.approx(1.0)
        assert corr[1, 1] == pytest.approx(1.0)

    def test_cov_to_corr_bounded(self):
        sizer = _sizer()
        cov = np.array([[0.01, 0.003], [0.003, 0.04]])
        corr = sizer._cov_to_corr(cov)
        assert np.all(corr >= -1.0) and np.all(corr <= 1.0)
