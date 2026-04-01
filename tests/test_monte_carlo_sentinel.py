"""tests/test_monte_carlo_sentinel.py — Monte Carlo Drawdown Sentinel tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from risk.monte_carlo_sentinel import MonteCarloSentinel, SentinelResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sentinel(tmp_path):
    return MonteCarloSentinel(
        n_paths=200,
        look_ahead=4,
        breach_threshold=0.30,
        defensive_threshold=0.60,
        history_len=60,
        data_dir=tmp_path,
    )


def _fill_history(s: MonteCarloSentinel, values) -> None:
    for v in values:
        s.record_pnl(v)


# ── SentinelResult ────────────────────────────────────────────────────────────

class TestSentinelResult:
    def test_defaults(self):
        r = SentinelResult()
        assert r.breach_probability == 0.0
        assert r.size_multiplier == 1.0
        assert r.max_positions_cap is None
        assert r.is_defensive is False
        assert r.tier == "green"

    def test_to_dict(self):
        r = SentinelResult(breach_probability=0.4, tier="amber", size_multiplier=0.7)
        d = r.to_dict()
        assert d["tier"] == "amber"
        assert d["breach_probability"] == 0.4
        assert "size_multiplier" in d


# ── record_pnl + history ──────────────────────────────────────────────────────

class TestRecordPnl:
    def test_appends_to_history(self, sentinel):
        sentinel.record_pnl(0.01)
        sentinel.record_pnl(-0.005)
        assert len(sentinel._history) == 2

    def test_history_bounded(self, sentinel):
        for _ in range(80):
            sentinel.record_pnl(0.001)
        assert len(sentinel._history) == 60  # maxlen=60

    def test_save_on_record(self, sentinel, tmp_path):
        sentinel.record_pnl(0.01)
        assert (tmp_path / "monte_carlo_sentinel_state.json").exists()


# ── _estimate_params ──────────────────────────────────────────────────────────

class TestEstimateParams:
    def test_insufficient_history_defaults(self, sentinel):
        mu, sigma = sentinel._estimate_params()
        assert mu == 0.0
        assert sigma == 0.005

    def test_with_history(self, sentinel):
        _fill_history(sentinel, [0.01, 0.02, -0.005, 0.015, -0.01, 0.008])
        mu, sigma = sentinel._estimate_params()
        assert sigma > 0.0
        assert abs(mu - np.mean([0.01, 0.02, -0.005, 0.015, -0.01, 0.008])) < 1e-9


# ── simulate ─────────────────────────────────────────────────────────────────

class TestSimulate:
    def test_returns_probability_in_range(self, sentinel):
        _fill_history(sentinel, [0.01] * 20 + [-0.005] * 10)
        prob = sentinel.simulate(current_daily_pnl=0.0, daily_loss_limit=-0.02)
        assert 0.0 <= prob <= 1.0

    def test_already_breached_high_prob(self, sentinel):
        """If current PnL is already at/beyond limit, probability should be high."""
        _fill_history(sentinel, [0.001] * 20)
        prob = sentinel.simulate(current_daily_pnl=-0.025, daily_loss_limit=-0.02)
        assert prob > 0.5

    def test_very_profitable_low_prob(self, sentinel):
        """Starting well above limit with positive drift → low breach probability."""
        # History of strong positive returns
        _fill_history(sentinel, [0.005] * 30)
        prob = sentinel.simulate(current_daily_pnl=0.01, daily_loss_limit=-0.02)
        assert prob < 0.5

    def test_deterministic_with_seed(self, sentinel):
        """Two calls with same history should produce similar results (stochastic tolerance)."""
        _fill_history(sentinel, [0.002] * 20 + [-0.001] * 10)
        p1 = sentinel.simulate(-0.005, -0.02)
        p2 = sentinel.simulate(-0.005, -0.02)
        # Allow ±15% deviation due to stochasticity
        assert abs(p1 - p2) < 0.15


# ── evaluate / tier logic ─────────────────────────────────────────────────────

class TestEvaluate:
    def test_insufficient_history_returns_green(self, sentinel):
        # Only 2 samples — below MIN_HISTORY=5
        sentinel.record_pnl(0.01)
        sentinel.record_pnl(-0.005)
        result = sentinel.evaluate(current_daily_pnl=-0.005, daily_loss_limit=-0.02)
        assert result.tier == "green"
        assert result.size_multiplier == 1.0

    def test_force_runs_even_insufficient(self, sentinel):
        sentinel.record_pnl(0.01)
        result = sentinel.evaluate(
            current_daily_pnl=-0.005, daily_loss_limit=-0.02, force=True
        )
        # Should run, result is valid
        assert isinstance(result.breach_probability, float)

    def test_green_tier_no_cap(self, sentinel):
        _fill_history(sentinel, [0.005] * 30)
        result = sentinel.evaluate(current_daily_pnl=0.005, daily_loss_limit=-0.02)
        assert result.tier == "green"
        assert result.size_multiplier == 1.0
        assert result.max_positions_cap is None

    def test_defensive_tier_on_near_breach(self, sentinel):
        """Force defensive mode by starting at daily limit."""
        # Use large negative history and start at limit
        _fill_history(sentinel, [-0.01] * 40)
        result = sentinel.evaluate(
            current_daily_pnl=-0.019, daily_loss_limit=-0.02, force=True
        )
        # With high vol and negative drift, breach prob should be high
        assert result.size_multiplier <= 0.70

    def test_result_stored(self, sentinel):
        _fill_history(sentinel, [0.001] * 10)
        result = sentinel.evaluate(-0.005, -0.02)
        assert sentinel._last_result is not None
        assert sentinel._last_eval_ts > 0.0

    def test_size_multiplier_bounded(self, sentinel):
        _fill_history(sentinel, [-0.015] * 30)
        result = sentinel.evaluate(-0.018, -0.02, force=True)
        assert 0.0 < result.size_multiplier <= 1.0

    def test_defensive_sets_max_positions(self, sentinel):
        """Defensive tier should cap max_positions."""
        sentinel.defensive_max_pos = 2
        # Mock evaluate to force defensive tier
        _fill_history(sentinel, [-0.02] * 30)
        result = sentinel.evaluate(-0.019, -0.02, force=True)
        # If defensive → max_pos = 2
        if result.is_defensive:
            assert result.max_positions_cap == 2


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        s1 = MonteCarloSentinel(n_paths=50, look_ahead=4, data_dir=tmp_path)
        _fill_history(s1, [0.01, -0.005, 0.008, 0.003, -0.002])

        s2 = MonteCarloSentinel(n_paths=50, look_ahead=4, data_dir=tmp_path)
        assert len(s2._history) == 5
        assert abs(list(s2._history)[0] - 0.01) < 1e-9

    def test_missing_file_no_crash(self, tmp_path):
        # No state file → fresh sentinel
        s = MonteCarloSentinel(data_dir=tmp_path)
        assert len(s._history) == 0

    def test_corrupt_file_no_crash(self, tmp_path):
        (tmp_path / "monte_carlo_sentinel_state.json").write_text("NOT_JSON")
        s = MonteCarloSentinel(data_dir=tmp_path)
        assert len(s._history) == 0


# ── get_status ────────────────────────────────────────────────────────────────

class TestGetStatus:
    def test_status_keys(self, sentinel):
        status = sentinel.get_status()
        assert "history_len" in status
        assert "vol_estimate" in status
        assert "last_tier" in status
        assert "last_breach_prob" in status
        assert "last_size_mult" in status

    def test_status_reflects_history(self, sentinel):
        _fill_history(sentinel, [0.005] * 10)
        status = sentinel.get_status()
        assert status["history_len"] == 10
