"""Tests for CapitalAllocator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from risk.capital_allocator import CapitalAllocator, AllocationResult


def _make_allocator(**kwargs) -> CapitalAllocator:
    defaults = dict(lookback_days=20, min_alloc=0.10, max_alloc=0.85,
                    momentum_tilt=0.05, corr_threshold=0.70, corr_dampen=0.50,
                    rebalance_threshold=0.03)
    defaults.update(kwargs)
    return CapitalAllocator(**defaults)


def _feed(ca: CapitalAllocator, eq_rets: list, cr_rets: list):
    for eq, cr in zip(eq_rets, cr_rets):
        ca.update_leg_pnl(equity_pnl_pct=eq, crypto_pnl_pct=cr)


class TestInitialState:

    def test_default_equal_weight(self):
        ca = _make_allocator()
        assert ca.current_equity_frac == pytest.approx(0.50, abs=0.01)
        assert ca.current_crypto_frac == pytest.approx(0.50, abs=0.01)

    def test_fractions_sum_to_one(self):
        ca = _make_allocator()
        assert ca.current_equity_frac + ca.current_crypto_frac == pytest.approx(1.0, abs=0.001)


class TestInsufficientHistory:

    def test_returns_unchanged_when_too_few_days(self):
        ca = _make_allocator()
        ca.update_leg_pnl(0.01, 0.02)
        result = ca.compute_allocation()
        assert result.rebalance_recommended is False
        assert "insufficient" in result.reason

    def test_equity_frac_unchanged_with_no_data(self):
        ca = _make_allocator()
        result = ca.compute_allocation()
        assert result.equity_frac == pytest.approx(0.50, abs=0.01)


class TestAllocationLogic:

    def test_strong_equity_gets_higher_allocation(self):
        ca = _make_allocator(rebalance_threshold=0.0)
        import random; rng = random.Random(1)
        eq_rets = [0.02 + rng.gauss(0, 0.003) for _ in range(20)]
        cr_rets = [-0.01 + rng.gauss(0, 0.003) for _ in range(20)]
        _feed(ca, eq_rets, cr_rets)
        result = ca.compute_allocation()
        assert result.equity_frac > result.crypto_frac

    def test_strong_crypto_gets_higher_allocation(self):
        ca = _make_allocator(rebalance_threshold=0.0)
        import random; rng = random.Random(2)
        eq_rets = [-0.01 + rng.gauss(0, 0.003) for _ in range(20)]
        cr_rets = [0.02 + rng.gauss(0, 0.003) for _ in range(20)]
        _feed(ca, eq_rets, cr_rets)
        result = ca.compute_allocation()
        assert result.crypto_frac > result.equity_frac

    def test_equal_performance_near_equal_weight(self):
        ca = _make_allocator(rebalance_threshold=0.0)
        returns = [0.01, -0.01, 0.01, -0.01, 0.01] * 4
        _feed(ca, returns, returns)
        result = ca.compute_allocation()
        assert abs(result.equity_frac - result.crypto_frac) < 0.10

    def test_fractions_always_sum_to_one(self):
        ca = _make_allocator(rebalance_threshold=0.0)
        eq_rets = [0.02, -0.01, 0.03, -0.005, 0.01] * 4
        cr_rets = [-0.01, 0.03, -0.02, 0.015, 0.005] * 4
        _feed(ca, eq_rets, cr_rets)
        result = ca.compute_allocation()
        assert result.equity_frac + result.crypto_frac == pytest.approx(1.0, abs=0.001)

    def test_allocations_respect_min_max(self):
        ca = _make_allocator(min_alloc=0.15, max_alloc=0.85, rebalance_threshold=0.0)
        import random; rng = random.Random(77)
        eq_rets = [0.05 + rng.gauss(0, 0.005) for _ in range(20)]
        cr_rets = [-0.05 + rng.gauss(0, 0.005) for _ in range(20)]
        _feed(ca, eq_rets, cr_rets)
        result = ca.compute_allocation()
        assert result.equity_frac <= 0.85
        assert result.crypto_frac >= 0.15


class TestCorrelationDampening:

    def test_high_correlation_compresses_toward_equal(self):
        ca = _make_allocator(corr_threshold=0.50, corr_dampen=0.80, rebalance_threshold=0.0)
        # Perfect correlation between legs
        shared = [0.02 if i % 3 else -0.01 for i in range(20)]
        extra_eq = [r + 0.005 for r in shared]  # equity slightly better
        _feed(ca, extra_eq, shared)
        result = ca.compute_allocation()
        # With high correlation, fractions should be close to equal
        assert abs(result.equity_frac - result.crypto_frac) < 0.20

    def test_zero_correlation_allows_full_tilt(self):
        ca = _make_allocator(corr_threshold=0.70, rebalance_threshold=0.0)
        # Uncorrelated: equity always up, crypto always flat
        eq_rets = [0.02] * 20
        cr_rets = [0.0, 0.01, -0.01] * 6 + [0.0, 0.0]
        _feed(ca, eq_rets, cr_rets)
        result = ca.compute_allocation()
        # Low correlation → equity should dominate more
        assert result.equity_frac > 0.55


class TestRebalanceTrigger:

    def test_no_rebalance_when_shift_small(self):
        ca = _make_allocator(rebalance_threshold=0.99)  # impossibly high threshold
        import random; rng = random.Random(10)
        eq_rets = [0.01 + rng.gauss(0, 0.003) for _ in range(20)]
        cr_rets = [0.009 + rng.gauss(0, 0.003) for _ in range(20)]
        _feed(ca, eq_rets, cr_rets)
        result = ca.compute_allocation()
        assert result.rebalance_recommended is False

    def test_rebalance_when_shift_large(self):
        ca = _make_allocator(rebalance_threshold=0.01)  # low threshold
        import random; rng = random.Random(11)
        eq_rets = [0.05 + rng.gauss(0, 0.005) for _ in range(20)]
        cr_rets = [-0.03 + rng.gauss(0, 0.005) for _ in range(20)]
        _feed(ca, eq_rets, cr_rets)
        result = ca.compute_allocation()
        assert result.rebalance_recommended is True

    def test_current_fracs_updated_on_rebalance(self):
        ca = _make_allocator(rebalance_threshold=0.01)
        import random; rng = random.Random(12)
        eq_rets = [0.05 + rng.gauss(0, 0.005) for _ in range(20)]
        cr_rets = [-0.03 + rng.gauss(0, 0.005) for _ in range(20)]
        _feed(ca, eq_rets, cr_rets)
        result = ca.compute_allocation()
        assert ca.current_equity_frac == pytest.approx(result.equity_frac, abs=0.001)
        assert ca.current_crypto_frac == pytest.approx(result.crypto_frac, abs=0.001)


class TestSharpeComputation:

    def test_positive_rets_positive_sharpe(self):
        ca = _make_allocator()
        rets = np.array([0.01] * 20)
        sharpe = ca._sharpe(rets)
        assert sharpe > 0

    def test_negative_rets_negative_sharpe(self):
        ca = _make_allocator()
        rets = np.array([-0.01] * 20)
        sharpe = ca._sharpe(rets)
        assert sharpe < 0

    def test_zero_vol_returns_zero_sharpe(self):
        ca = _make_allocator()
        rets = np.array([0.0] * 10)
        sharpe = ca._sharpe(rets)
        assert sharpe == 0.0


class TestCorrelationComputation:

    def test_identical_series_correlation_one(self):
        ca = _make_allocator()
        rets = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 4)
        corr = ca._correlation(rets, rets)
        assert corr == pytest.approx(1.0, abs=0.001)

    def test_opposite_series_correlation_neg_one(self):
        ca = _make_allocator()
        rets = np.array([0.01, -0.02, 0.03, -0.01, 0.02] * 4)
        corr = ca._correlation(rets, -rets)
        assert corr == pytest.approx(-1.0, abs=0.001)

    def test_short_series_returns_zero(self):
        ca = _make_allocator()
        corr = ca._correlation(np.array([0.01]), np.array([0.02]))
        assert corr == 0.0


class TestPersistence:

    def test_state_persisted_and_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ca = CapitalAllocator(data_dir=Path(tmp), rebalance_threshold=0.01)
            # Use varied returns so std > 0 → Sharpe computed; equity clearly wins
            import random; rng = random.Random(42)
            eq_rets = [0.03 + rng.gauss(0, 0.005) for _ in range(20)]
            cr_rets = [-0.02 + rng.gauss(0, 0.005) for _ in range(20)]
            _feed(ca, eq_rets, cr_rets)
            ca.compute_allocation()

            ca2 = CapitalAllocator(data_dir=Path(tmp))
            assert abs(ca2.current_equity_frac - ca.current_equity_frac) < 0.01
            assert len(ca2._equity_history) > 0

    def test_persist_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            ca = CapitalAllocator(data_dir=Path(tmp), rebalance_threshold=0.01)
            import random; rng = random.Random(99)
            eq_rets = [0.04 + rng.gauss(0, 0.005) for _ in range(20)]
            cr_rets = [-0.02 + rng.gauss(0, 0.005) for _ in range(20)]
            _feed(ca, eq_rets, cr_rets)
            ca.compute_allocation()
            path = Path(tmp) / "capital_allocation.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert "current_equity_frac" in data

    def test_load_with_no_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            ca = CapitalAllocator(data_dir=Path(tmp))
            assert ca.current_equity_frac == pytest.approx(0.50, abs=0.01)


class TestAllocationResult:

    def test_result_is_allocation_result_type(self):
        ca = _make_allocator()
        _feed(ca, [0.01] * 10, [0.01] * 10)
        result = ca.compute_allocation()
        assert isinstance(result, AllocationResult)

    def test_result_has_sharpe_fields(self):
        ca = _make_allocator(rebalance_threshold=0.0)
        _feed(ca, [0.01] * 20, [0.02] * 20)
        result = ca.compute_allocation()
        assert isinstance(result.equity_sharpe, float)
        assert isinstance(result.crypto_sharpe, float)

    def test_result_has_timestamp(self):
        ca = _make_allocator()
        result = ca.compute_allocation()
        assert result.timestamp.endswith("Z")
