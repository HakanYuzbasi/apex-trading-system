"""
tests/test_portfolio_vol_target.py — Unit tests for risk/portfolio_vol_target.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from risk.portfolio_vol_target import (
    PortfolioVolTarget,
    VolTargetState,
    compute_realised_vol,
    compute_vol_target_multiplier,
    get_vol_target,
)


# ── compute_realised_vol ──────────────────────────────────────────────────────

class TestComputeRealisedVol:
    def test_empty_returns_zero(self):
        assert compute_realised_vol([]) == 0.0

    def test_single_value_returns_zero(self):
        assert compute_realised_vol([100.0]) == 0.0

    def test_constant_series_returns_near_zero(self):
        vol = compute_realised_vol([100.0] * 30)
        assert vol < 1e-8

    def test_volatile_series_higher_than_stable(self):
        stable = [100.0, 100.5, 99.8, 100.2, 100.0] * 6
        volatile = [100.0, 105.0, 95.0, 108.0, 92.0] * 6
        assert compute_realised_vol(volatile) > compute_realised_vol(stable)

    def test_annualisation_applied(self):
        # Daily vol of ~1% → annualised ~16% (1% × √252)
        np.random.seed(42)
        daily_returns = 1.0 + np.random.normal(0, 0.01, 50)
        equity = np.cumprod(daily_returns) * 100
        vol = compute_realised_vol(equity.tolist(), bars_per_year=252)
        assert 0.05 < vol < 0.30  # roughly correct range

    def test_zeros_filtered(self):
        series = [0.0, 100.0, 101.0, 102.0, 100.0]
        vol = compute_realised_vol(series)
        assert isinstance(vol, float)

    def test_result_non_negative(self):
        np.random.seed(0)
        equity = np.cumprod(1.0 + np.random.normal(0, 0.02, 30)) * 10000
        vol = compute_realised_vol(equity.tolist())
        assert vol >= 0.0


# ── compute_vol_target_multiplier ─────────────────────────────────────────────

class TestComputeVolTargetMultiplier:
    def test_higher_vol_gives_lower_mult(self):
        # Realised > target → scale down
        mult = compute_vol_target_multiplier(0.20, 0.12, 0.50, 1.50)
        assert mult < 1.0

    def test_lower_vol_gives_higher_mult(self):
        # Realised < target → scale up
        mult = compute_vol_target_multiplier(0.06, 0.12, 0.50, 1.50)
        assert mult > 1.0

    def test_equal_vol_returns_one(self):
        mult = compute_vol_target_multiplier(0.12, 0.12, 0.50, 1.50)
        assert mult == pytest.approx(1.0)

    def test_min_mult_respected(self):
        # Extreme high vol → floor
        mult = compute_vol_target_multiplier(2.0, 0.12, 0.50, 1.50)
        assert mult == pytest.approx(0.50)

    def test_max_mult_respected(self):
        # Very low vol → ceiling
        mult = compute_vol_target_multiplier(0.001, 0.12, 0.50, 1.50)
        assert mult == pytest.approx(1.50)

    def test_zero_vol_returns_one(self):
        mult = compute_vol_target_multiplier(0.0, 0.12, 0.50, 1.50)
        assert mult == pytest.approx(1.0)

    def test_near_zero_vol_returns_one(self):
        mult = compute_vol_target_multiplier(1e-10, 0.12, 0.50, 1.50)
        assert mult == pytest.approx(1.0)


# ── PortfolioVolTarget ────────────────────────────────────────────────────────

def _build_equity_series(n: int = 30, daily_vol: float = 0.02, seed: int = 42) -> list:
    np.random.seed(seed)
    rets = np.random.normal(0, daily_vol, n)
    equity = np.cumprod(1 + rets) * 100_000
    return equity.tolist()


class TestPortfolioVolTarget:
    def test_get_multiplier_returns_one_with_no_data(self):
        pvt = PortfolioVolTarget()
        assert pvt.get_multiplier() == pytest.approx(1.0)

    def test_get_multiplier_returns_one_below_min_obs(self):
        pvt = PortfolioVolTarget()
        for v in [100_000.0, 101_000.0]:  # only 2 obs, below min
            pvt.record_equity(v)
        assert pvt.get_multiplier() == pytest.approx(1.0)

    def test_get_multiplier_active_above_min_obs(self):
        pvt = PortfolioVolTarget()
        for v in _build_equity_series(30):
            pvt.record_equity(v)
        mult = pvt.get_multiplier()
        assert isinstance(mult, float)
        assert 0.5 <= mult <= 1.5

    def test_high_vol_reduces_mult_below_one(self):
        pvt = PortfolioVolTarget()
        # Very volatile series → mult < 1
        for v in _build_equity_series(30, daily_vol=0.05):  # 5% daily vol → ~80% annual
            pvt.record_equity(v)
        mult = pvt.get_multiplier()
        assert mult < 1.0

    def test_low_vol_increases_mult_above_one(self):
        pvt = PortfolioVolTarget()
        # Very stable series → mult > 1 (up to 1.5 cap)
        for v in _build_equity_series(30, daily_vol=0.001):  # 0.1% daily vol → ~1.6% annual
            pvt.record_equity(v)
        mult = pvt.get_multiplier()
        assert mult > 1.0

    def test_mult_bounded_by_min_max(self):
        pvt = PortfolioVolTarget()
        for v in _build_equity_series(30, daily_vol=0.10):  # extreme vol
            pvt.record_equity(v)
        mult = pvt.get_multiplier()
        assert 0.50 <= mult <= 1.50

    def test_disabled_returns_one(self, monkeypatch):
        import risk.portfolio_vol_target as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "VOL_TARGET_ENABLED" else mod._DEF.get(k))
        pvt = PortfolioVolTarget()
        for v in _build_equity_series(30):
            pvt.record_equity(v)
        assert pvt.get_multiplier() == pytest.approx(1.0)

    def test_get_state_returns_vol_target_state(self):
        pvt = PortfolioVolTarget()
        state = pvt.get_state()
        assert isinstance(state, VolTargetState)

    def test_get_state_active_after_min_obs(self):
        pvt = PortfolioVolTarget()
        for v in _build_equity_series(30):
            pvt.record_equity(v)
        state = pvt.get_state()
        assert state.active is True
        assert state.n_obs >= 10

    def test_get_state_not_active_below_min_obs(self):
        pvt = PortfolioVolTarget()
        pvt.record_equity(100_000.0)
        pvt.record_equity(101_000.0)
        state = pvt.get_state()
        assert state.active is False

    def test_get_report_has_expected_keys(self):
        pvt = PortfolioVolTarget()
        report = pvt.get_report()
        for key in ["realised_vol_pct", "target_vol_pct", "multiplier", "n_obs", "active", "timestamp"]:
            assert key in report

    def test_reset_clears_buffer(self):
        pvt = PortfolioVolTarget()
        for v in _build_equity_series(30):
            pvt.record_equity(v)
        pvt.reset()
        assert pvt.get_multiplier() == pytest.approx(1.0)
        assert pvt.get_state().active is False

    def test_negative_equity_ignored(self):
        pvt = PortfolioVolTarget()
        pvt.record_equity(-100.0)
        pvt.record_equity(0.0)
        assert pvt.get_multiplier() == pytest.approx(1.0)

    def test_vol_target_state_to_dict(self):
        s = VolTargetState(realised_vol=0.15, target_vol=0.12, multiplier=0.80, n_obs=25, active=True)
        d = s.to_dict()
        assert d["realised_vol_pct"] == pytest.approx(15.0)
        assert d["target_vol_pct"] == pytest.approx(12.0)
        assert d["multiplier"] == pytest.approx(0.80)
        assert d["active"] is True


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_instance(self):
        assert isinstance(get_vol_target(), PortfolioVolTarget)

    def test_same_object(self):
        a = get_vol_target()
        b = get_vol_target()
        assert a is b
