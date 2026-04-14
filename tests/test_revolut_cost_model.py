"""
tests/test_revolut_cost_model.py
================================================================================
Unit tests for the Revolut-specific transaction cost model.
================================================================================
"""
from __future__ import annotations

import pytest
from quant_system.execution.revolut_cost_model import (
    RevolutCostModel,
    STABLECOIN_FREE_MONTHLY_EUR,
    STABLECOIN_FEE_ABOVE_CAP_BPS,
    CRYPTO_SPREAD_BPS,
    EQUITY_SPREAD_BPS,
    EQUITY_COMMISSION_BPS,
)


# ---------------------------------------------------------------------------
# Equity cost tests
# ---------------------------------------------------------------------------

class TestRevolutEquityCosts:
    def test_first_trades_are_commission_free(self):
        """First N trades within the monthly free allowance have no commission."""
        model = RevolutCostModel()
        model.monthly_equity_trade_count = 0

        result = model.calculate_cost(150.0, 1_000.0, "equity")
        assert result.commission_bps == 0.0
        assert result.spread_cost_bps == EQUITY_SPREAD_BPS
        assert result.total_fractional_cost == pytest.approx(EQUITY_SPREAD_BPS / 10_000.0)

    def test_commission_charged_when_allowance_exhausted(self):
        """Trades beyond the free monthly allowance incur the commission fee."""
        model = RevolutCostModel(monthly_equity_free_trades=3)
        model.monthly_equity_trade_count = 3  # already consumed all free trades

        result = model.calculate_cost(150.0, 1_000.0, "equity")
        assert result.commission_bps == EQUITY_COMMISSION_BPS
        expected = (EQUITY_SPREAD_BPS + EQUITY_COMMISSION_BPS) / 10_000.0
        assert result.total_fractional_cost == pytest.approx(expected)

    def test_equity_trade_count_increments(self):
        """Each equity trade advances the monthly counter."""
        model = RevolutCostModel(monthly_equity_free_trades=5)
        for i in range(3):
            model.calculate_cost(100.0, 500.0, "equity")
        assert model.monthly_equity_trade_count == 3


# ---------------------------------------------------------------------------
# Stablecoin cost tests
# ---------------------------------------------------------------------------

class TestRevolutStablecoinCosts:
    def test_no_tier_fee_below_monthly_cap(self):
        """Orders that keep cumulative volume below €500k pay only the spread."""
        model = RevolutCostModel(monthly_stablecoin_volume_eur=0.0)
        result = model.calculate_cost(
            1.0, 100_000.0, "crypto_stablecoin",
            monthly_stablecoin_vol=400_000.0,
        )
        assert result.stablecoin_tier_fee_bps == pytest.approx(0.0)
        assert result.fee_tier_activated is False

    def test_tier_fee_activates_above_cap(self):
        """Orders that push monthly volume above €500k trigger the 0.25% fee."""
        model = RevolutCostModel(monthly_stablecoin_volume_eur=490_000.0)
        # Order size 100k; 10k is free, 90k is above cap
        result = model.calculate_cost(
            1.0, 100_000.0, "crypto_stablecoin",
            monthly_stablecoin_vol=490_000.0,
        )
        assert result.fee_tier_activated is True
        # Proportional tier fee: 90k/100k * 25 bps = 22.5 bps
        expected_tier = STABLECOIN_FEE_ABOVE_CAP_BPS * (90_000.0 / 100_000.0)
        assert result.stablecoin_tier_fee_bps == pytest.approx(expected_tier, rel=1e-4)

    def test_fully_above_cap_order(self):
        """When entire order is above the cap, full 25 bps tier fee applies."""
        model = RevolutCostModel(
            monthly_stablecoin_volume_eur=STABLECOIN_FREE_MONTHLY_EUR
        )
        result = model.calculate_cost(
            1.0, 50_000.0, "crypto_stablecoin",
            monthly_stablecoin_vol=STABLECOIN_FREE_MONTHLY_EUR,
        )
        assert result.fee_tier_activated is True
        assert result.stablecoin_tier_fee_bps == pytest.approx(STABLECOIN_FEE_ABOVE_CAP_BPS, rel=1e-4)

    def test_zero_order_returns_zero_cost(self):
        model = RevolutCostModel()
        result = model.calculate_cost(1.0, 0.0, "crypto_stablecoin")
        assert result.total_fractional_cost == 0.0

    def test_stablecoin_volume_accumulates(self):
        """Monthly volume accumulator updates after each trade."""
        model = RevolutCostModel(monthly_stablecoin_volume_eur=100_000.0)
        model.calculate_cost(1.0, 50_000.0, "crypto_stablecoin")
        assert model.monthly_stablecoin_volume_eur == pytest.approx(150_000.0)


# ---------------------------------------------------------------------------
# Non-stablecoin crypto tests
# ---------------------------------------------------------------------------

class TestRevolutCryptoOtherCosts:
    def test_only_spread_applied(self):
        model = RevolutCostModel()
        result = model.calculate_cost(50_000.0, 5_000.0, "crypto_other")
        assert result.commission_bps == 0.0
        assert result.stablecoin_tier_fee_bps == 0.0
        assert result.spread_cost_bps == CRYPTO_SPREAD_BPS
        assert result.fee_tier_activated is False

    def test_total_fractional_cost_matches_spread(self):
        model = RevolutCostModel()
        result = model.calculate_cost(50_000.0, 5_000.0, "crypto_other")
        expected = CRYPTO_SPREAD_BPS / 10_000.0
        assert result.total_fractional_cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# State feature: stablecoin tier proximity
# ---------------------------------------------------------------------------

class TestStablecoinTierProximity:
    def test_zero_proximity_when_no_volume(self):
        model = RevolutCostModel(monthly_stablecoin_volume_eur=0.0)
        assert model.stablecoin_tier_proximity() == pytest.approx(0.0)

    def test_full_proximity_at_cap(self):
        model = RevolutCostModel(
            monthly_stablecoin_volume_eur=STABLECOIN_FREE_MONTHLY_EUR
        )
        assert model.stablecoin_tier_proximity() == pytest.approx(1.0)

    def test_clamped_at_one_above_cap(self):
        model = RevolutCostModel(
            monthly_stablecoin_volume_eur=STABLECOIN_FREE_MONTHLY_EUR * 2
        )
        assert model.stablecoin_tier_proximity() == pytest.approx(1.0)

    def test_proximity_half_at_half_cap(self):
        model = RevolutCostModel(
            monthly_stablecoin_volume_eur=STABLECOIN_FREE_MONTHLY_EUR / 2
        )
        assert model.stablecoin_tier_proximity() == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# to_dict() interface compatibility with downstream reward code
# ---------------------------------------------------------------------------

class TestRevolutCostBreakdownDict:
    def test_to_dict_has_required_keys(self):
        model = RevolutCostModel()
        result = model.calculate_cost(1.0, 10_000.0, "crypto_other")
        d = result.to_dict()
        for key in ("total_fractional_cost", "spread_bps", "commission_bps",
                    "stablecoin_tier_fee_bps", "multiplier"):
            assert key in d, f"Missing key: {key}"

    def test_cost_above_cap_exceeds_cost_below_cap(self):
        """Fundamental invariant: high monthly volume → higher cost."""
        model_below = RevolutCostModel(monthly_stablecoin_volume_eur=100_000.0)
        model_above = RevolutCostModel(
            monthly_stablecoin_volume_eur=STABLECOIN_FREE_MONTHLY_EUR
        )
        c_below = model_below.calculate_cost(
            1.0, 200_000.0, "crypto_stablecoin",
            monthly_stablecoin_vol=100_000.0,
        )
        c_above = model_above.calculate_cost(
            1.0, 200_000.0, "crypto_stablecoin",
            monthly_stablecoin_vol=STABLECOIN_FREE_MONTHLY_EUR,
        )
        assert c_above.total_fractional_cost > c_below.total_fractional_cost
