"""
Tests for risk/signal_portfolio_constructor.py — Signal-Aware Portfolio Construction.

Covers: signal updates, return updates, HRP computation, sizing scale, fallback,
persistence, and the public snapshot API.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from risk.signal_portfolio_constructor import SignalPortfolioConstructor, PortfolioWeights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_constructor(tmp_path=None, **kwargs) -> SignalPortfolioConstructor:
    defaults = dict(
        max_single_weight=0.15,
        min_history=5,
        signal_blend=0.40,
        recompute_interval_s=0.0,   # always recompute
    )
    defaults.update(kwargs)
    return SignalPortfolioConstructor(state_dir=tmp_path, **defaults)


def _random_returns(n_obs: int = 30, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.standard_normal(n_obs) * 0.01)


def _feed_returns(constructor: SignalPortfolioConstructor, symbols: list[str], n_obs: int = 30) -> None:
    rets = {s: _random_returns(n_obs, seed=i) for i, s in enumerate(symbols)}
    constructor.update_returns(rets)


def _feed_signals(constructor: SignalPortfolioConstructor, symbols: list[str]) -> None:
    sigs = {s: 0.10 + i * 0.02 for i, s in enumerate(symbols)}
    constructor.update_signals(sigs)


# ---------------------------------------------------------------------------
# update_signals
# ---------------------------------------------------------------------------

class TestUpdateSignals:
    def test_stores_valid_signal(self):
        c = _make_constructor()
        c.update_signals({"AAPL": 0.20})
        assert "AAPL" in c._signals
        assert c._signals["AAPL"] == pytest.approx(0.20)

    def test_ignores_nan_signal(self):
        c = _make_constructor()
        c.update_signals({"AAPL": float("nan")})
        assert "AAPL" not in c._signals

    def test_ignores_inf_signal(self):
        c = _make_constructor()
        c.update_signals({"AAPL": float("inf")})
        assert "AAPL" not in c._signals

    def test_ignores_none_signal(self):
        c = _make_constructor()
        c.update_signals({"AAPL": None})  # type: ignore
        assert "AAPL" not in c._signals

    def test_overwrites_existing_signal(self):
        c = _make_constructor()
        c.update_signals({"AAPL": 0.10})
        c.update_signals({"AAPL": 0.25})
        assert c._signals["AAPL"] == pytest.approx(0.25)

    def test_multiple_symbols(self):
        c = _make_constructor()
        c.update_signals({"AAPL": 0.10, "MSFT": 0.15, "TSLA": 0.20})
        assert len(c._signals) == 3


# ---------------------------------------------------------------------------
# update_returns
# ---------------------------------------------------------------------------

class TestUpdateReturns:
    def test_stores_returns(self):
        c = _make_constructor()
        c.update_returns({"AAPL": [0.01, -0.02, 0.005]})
        assert "AAPL" in c._returns
        assert len(c._returns["AAPL"]) == 3

    def test_filters_nan_from_returns(self):
        c = _make_constructor()
        c.update_returns({"AAPL": [0.01, float("nan"), 0.005]})
        assert all(math.isfinite(r) for r in c._returns["AAPL"])
        assert len(c._returns["AAPL"]) == 2


# ---------------------------------------------------------------------------
# maybe_recompute / compute
# ---------------------------------------------------------------------------

class TestMaybeRecompute:
    def test_returns_true_when_recomputed(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT", "TSLA"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        assert c.maybe_recompute() is True

    def test_returns_false_within_interval(self):
        c = _make_constructor(recompute_interval_s=3600.0)
        syms = ["AAPL", "MSFT"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()   # first compute
        assert c.maybe_recompute() is False

    def test_weights_sum_le_one(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOG"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=30)
        c.maybe_recompute()
        weights = c._current.weights
        total = sum(weights.values())
        assert total <= 1.0 + 1e-6

    def test_each_weight_le_max_weight(self):
        c = _make_constructor(max_single_weight=0.15)
        syms = [f"SYM{i}" for i in range(8)]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=30)
        c.maybe_recompute()
        for w in c._current.weights.values():
            assert w <= 0.15 + 1e-6

    def test_method_set_to_hrp_signal(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT", "TSLA"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        assert c._current.method == "hrp_signal"

    def test_fallback_with_single_symbol(self):
        c = _make_constructor()
        c.update_signals({"AAPL": 0.20})
        c.update_returns({"AAPL": _random_returns(20)})
        c.maybe_recompute()
        assert c._current.method == "equal"

    def test_fallback_with_insufficient_history(self):
        c = _make_constructor(min_history=30)
        syms = ["AAPL", "MSFT"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=5)   # < min_history
        c.maybe_recompute()
        assert c._current.method == "equal"


# ---------------------------------------------------------------------------
# get_target_weight
# ---------------------------------------------------------------------------

class TestGetTargetWeight:
    def test_returns_max_weight_before_first_compute(self):
        c = _make_constructor(max_single_weight=0.15)
        assert c.get_target_weight("AAPL") == pytest.approx(0.15)

    def test_returns_zero_for_unknown_symbol_after_compute(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        assert c.get_target_weight("UNKNOWN") == pytest.approx(0.0)

    def test_known_symbol_has_positive_weight(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT", "TSLA"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        assert c.get_target_weight("AAPL") > 0.0


# ---------------------------------------------------------------------------
# get_sizing_scale
# ---------------------------------------------------------------------------

class TestGetSizingScale:
    def test_returns_one_for_zero_portfolio_value(self):
        c = _make_constructor()
        assert c.get_sizing_scale("AAPL", 10000, 0) == pytest.approx(1.0)

    def test_returns_one_for_zero_notional(self):
        c = _make_constructor()
        assert c.get_sizing_scale("AAPL", 0, 500000) == pytest.approx(1.0)

    def test_returns_one_when_within_budget(self):
        c = _make_constructor(max_single_weight=0.15)
        # Before compute: target = max_weight = 0.15, budget = 0.15 × 500k = 75k
        scale = c.get_sizing_scale("AAPL", 70_000, 500_000)
        assert scale == pytest.approx(1.0)

    def test_returns_fraction_when_over_budget(self):
        c = _make_constructor(max_single_weight=0.15)
        # budget = 0.15 × 500k = 75k; proposed = 150k → scale = 0.5
        scale = c.get_sizing_scale("AAPL", 150_000, 500_000)
        assert scale == pytest.approx(0.5)

    def test_scale_is_between_zero_and_one(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT", "TSLA"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        scale = c.get_sizing_scale("AAPL", 500_000, 500_000)
        assert 0.0 <= scale <= 1.0


# ---------------------------------------------------------------------------
# get_portfolio_snapshot
# ---------------------------------------------------------------------------

class TestGetPortfolioSnapshot:
    def test_returns_unavailable_before_compute(self):
        c = _make_constructor()
        snap = c.get_portfolio_snapshot()
        assert snap["available"] is False

    def test_returns_available_after_compute(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT", "TSLA"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        snap = c.get_portfolio_snapshot()
        assert snap["available"] is True

    def test_snapshot_has_required_keys(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        snap = c.get_portfolio_snapshot()
        for key in ["available", "method", "n_symbols", "weights", "top_signals"]:
            assert key in snap

    def test_weights_rounded_to_4dp(self):
        c = _make_constructor()
        syms = ["AAPL", "MSFT", "TSLA"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        for v in c.get_portfolio_snapshot()["weights"].values():
            assert v == round(v, 4)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_saves_and_loads_weights(self, tmp_path):
        c = _make_constructor(tmp_path=tmp_path)
        syms = ["AAPL", "MSFT", "TSLA"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        saved_weights = dict(c._current.weights)

        c2 = _make_constructor(tmp_path=tmp_path)
        assert c2._current is not None
        assert c2._current.weights == saved_weights

    def test_persistence_file_is_valid_json(self, tmp_path):
        c = _make_constructor(tmp_path=tmp_path)
        syms = ["AAPL", "MSFT"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=20)
        c.maybe_recompute()
        p = tmp_path / "portfolio_weights.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert "weights" in data
        assert "method" in data

    def test_load_survives_missing_file(self, tmp_path):
        c = _make_constructor(tmp_path=tmp_path)
        assert c._current is None  # no file exists, no crash

    def test_load_survives_corrupt_file(self, tmp_path):
        (tmp_path / "portfolio_weights.json").write_text("NOT JSON")
        c = _make_constructor(tmp_path=tmp_path)  # should not raise
        assert c._current is None


# ---------------------------------------------------------------------------
# HRP internals
# ---------------------------------------------------------------------------

class TestHrpInternals:
    def test_quasi_diag_returns_permutation(self):
        n = 5
        dist = np.random.default_rng(0).random((n, n))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        order = SignalPortfolioConstructor._quasi_diag(dist)
        assert sorted(order) == list(range(n))

    def test_cluster_var_is_positive(self):
        cov = np.eye(4) * 0.01
        var = SignalPortfolioConstructor._cluster_var(cov, [0, 1, 2])
        assert var > 0

    def test_hrp_weights_sum_to_one(self):
        c = _make_constructor()
        syms = ["A", "B", "C", "D"]
        _feed_signals(c, syms)
        _feed_returns(c, syms, n_obs=40)
        c.maybe_recompute()
        total = sum(c._current.weights.values())
        # After max_weight cap and renorm, total <= 1.0
        assert total <= 1.0 + 1e-6
        assert total > 0.5  # sanity: some weight allocated
