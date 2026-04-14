"""
quant_system/execution/revolut_cost_model.py
================================================================================
Revolut-Specific Transaction Cost Model
================================================================================

Encodes the exact fee structure documented in the Revolut Securities Europe UAB
and Revolut Digital Assets Europe Ltd terms (MiCA-compliant OTC model):

  Equities (mixed-capacity rAgent/Principal):
  - Commission-free up to MONTHLY_EQUITY_FREE_TRADES trades per month.
  - Beyond that, EQUITY_COMMISSION_BPS per trade.
  - Indicative quote spread estimated at EQUITY_SPREAD_BPS (markups vary).

  Crypto — non-stablecoin:
  - OTC spread markup: CRYPTO_SPREAD_BPS (broker acts as direct counterparty).
  - No per-trade commission, but spread is the cost.

  Crypto — stablecoin:
  - Free (OTC) up to STABLECOIN_FREE_MONTHLY_EUR of monthly volume.
  - Beyond that cap: 0.25% flat (STABLECOIN_FEE_ABOVE_CAP_BPS = 25 bps).
  - Spread markup still applies on top.

Monthly state resets on month rollover; caller is responsible for persisting
`monthly_equity_trade_count` and `monthly_stablecoin_volume_eur` across calls.
================================================================================
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Literal, Optional

import numpy as np

AssetType = Literal["equity", "crypto_stablecoin", "crypto_other"]

# ---------------------------------------------------------------------------
# Fee schedule constants (source: Revolut product terms, March 2026)
# ---------------------------------------------------------------------------
EQUITY_COMMISSION_BPS: float = 25.0      # ~0.25 % per trade when cap exceeded
MONTHLY_EQUITY_FREE_TRADES: int = 3       # commission-free trades / month (Standard plan)
EQUITY_SPREAD_BPS: float = 10.0          # indicative quote markup estimate

CRYPTO_SPREAD_BPS: float = 25.0          # OTC counterparty spread (non-stablecoin)

STABLECOIN_FREE_MONTHLY_EUR: float = 500_000.0   # €500 k/month threshold
STABLECOIN_FEE_ABOVE_CAP_BPS: float = 25.0       # 0.25 % fee above threshold
STABLECOIN_SPREAD_BPS: float = 5.0       # tighter spread for stablecoins


@dataclass
class RevolutCostBreakdown:
    """Full breakdown of execution costs for a single Revolut trade."""

    asset_type: AssetType
    order_eur: float
    spread_cost_bps: float
    commission_bps: float
    stablecoin_tier_fee_bps: float        # non-zero only for stablecoins above cap
    total_fractional_cost: float          # fraction of notional (not bps)
    monthly_stablecoin_volume_after: float
    monthly_equity_trades_after: int
    fee_tier_activated: bool              # True if the 0.25 % stablecoin tier fired

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_fractional_cost": self.total_fractional_cost,
            "spread_bps": self.spread_cost_bps,
            "commission_bps": self.commission_bps,
            "stablecoin_tier_fee_bps": self.stablecoin_tier_fee_bps,
            "multiplier": 1.0,           # compat with TransactionCostModel interface
            "fee_tier_activated": float(self.fee_tier_activated),
        }


@dataclass
class RevolutCostModel:
    """
    Revolut-aware transaction cost model for the RL simulator.

    Usage
    -----
    Instantiate once per episode (or share across episodes while resetting
    ``monthly_stablecoin_volume_eur`` and ``monthly_equity_trade_count`` each
    month via ``reset_monthly_state()``).

    Parameters
    ----------
    equity_commission_bps:
        Commission per equity trade once the free monthly allowance is consumed.
    monthly_equity_free_trades:
        Number of commission-free equity trades per month (plan-dependent).
    equity_spread_bps:
        Estimated indicative quote spread markup for equities.
    crypto_spread_bps:
        OTC counterparty spread for non-stablecoin crypto.
    stablecoin_free_monthly_eur:
        Monthly stablecoin volume threshold below which 0 % fee applies.
    stablecoin_fee_above_cap_bps:
        Fee in basis points applied to volume above the monthly threshold.
    stablecoin_spread_bps:
        OTC spread for stablecoin exchanges.
    monthly_stablecoin_volume_eur:
        Running total of stablecoin volume this calendar month (persist externally).
    monthly_equity_trade_count:
        Running count of equity trades this calendar month (persist externally).
    current_month:
        Current (year, month) tuple; used to auto-reset on month rollover.
    """

    equity_commission_bps: float = EQUITY_COMMISSION_BPS
    monthly_equity_free_trades: int = MONTHLY_EQUITY_FREE_TRADES
    equity_spread_bps: float = EQUITY_SPREAD_BPS

    crypto_spread_bps: float = CRYPTO_SPREAD_BPS

    stablecoin_free_monthly_eur: float = STABLECOIN_FREE_MONTHLY_EUR
    stablecoin_fee_above_cap_bps: float = STABLECOIN_FEE_ABOVE_CAP_BPS
    stablecoin_spread_bps: float = STABLECOIN_SPREAD_BPS

    # Mutable running state — reset via reset_monthly_state()
    monthly_stablecoin_volume_eur: float = field(default=0.0)
    monthly_equity_trade_count: int = field(default=0)
    current_month: Optional[tuple] = field(default=None)

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def calculate_cost(
        self,
        price: float,
        order_size_eur: float,
        asset_type: AssetType,
        *,
        monthly_stablecoin_vol: Optional[float] = None,
        monthly_equity_trades: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> RevolutCostBreakdown:
        """
        Compute the full cost of a single trade and update the running monthly
        state accumulators.

        Parameters
        ----------
        price:
            Current asset price (used only for validation / logging; cost is
            always computed on ``order_size_eur``).
        order_size_eur:
            Notional order value in EUR.
        asset_type:
            One of ``"equity"``, ``"crypto_stablecoin"``, ``"crypto_other"``.
        monthly_stablecoin_vol:
            Override current monthly stablecoin volume (EUR) instead of using
            the instance accumulator. Useful for simulation.
        monthly_equity_trades:
            Override current monthly equity trade count instead of using the
            instance accumulator. Useful for simulation.
        now:
            Override current datetime for month-rollover detection.

        Returns
        -------
        RevolutCostBreakdown
            Full cost breakdown; ``total_fractional_cost`` is the fraction of
            notional to subtract from PnL.
        """
        if order_size_eur <= 0.0:
            return RevolutCostBreakdown(
                asset_type=asset_type,
                order_eur=0.0,
                spread_cost_bps=0.0,
                commission_bps=0.0,
                stablecoin_tier_fee_bps=0.0,
                total_fractional_cost=0.0,
                monthly_stablecoin_volume_after=self.monthly_stablecoin_volume_eur,
                monthly_equity_trades_after=self.monthly_equity_trade_count,
                fee_tier_activated=False,
            )

        self._maybe_reset_month(now)

        # Use caller-supplied overrides when present (for simulation sweep).
        stablecoin_vol = (
            monthly_stablecoin_vol
            if monthly_stablecoin_vol is not None
            else self.monthly_stablecoin_volume_eur
        )
        equity_trades = (
            monthly_equity_trades
            if monthly_equity_trades is not None
            else self.monthly_equity_trade_count
        )

        if asset_type == "equity":
            return self._cost_equity(order_size_eur, equity_trades)

        if asset_type == "crypto_stablecoin":
            return self._cost_stablecoin(order_size_eur, stablecoin_vol)

        # crypto_other
        return self._cost_crypto_other(order_size_eur)

    def estimate_expected_impact(
        self,
        order_size_eur: float,
        asset_type: AssetType,
        monthly_stablecoin_vol: float = 0.0,
    ) -> float:
        """
        Proactive cost estimate before action selection (mirrors
        TransactionCostModel interface used by the RL state builder).

        Returns ``total_fractional_cost`` without updating internal state.
        """
        breakdown = self.calculate_cost(
            price=1.0,
            order_size_eur=max(0.0, order_size_eur),
            asset_type=asset_type,
            monthly_stablecoin_vol=monthly_stablecoin_vol,
        )
        # Roll back accumulators — estimate must be side-effect-free.
        if asset_type == "crypto_stablecoin":
            self.monthly_stablecoin_volume_eur -= max(0.0, order_size_eur)
        elif asset_type == "equity":
            self.monthly_equity_trade_count = max(
                0, self.monthly_equity_trade_count - 1
            )
        return breakdown.total_fractional_cost

    def stablecoin_tier_proximity(self) -> float:
        """
        Normalised proximity to the stablecoin fee-tier threshold.

        Returns 0.0 when far below the cap, 1.0 when at or above it.
        Designed to be included as a state feature in the RL observation so
        the agent can learn to slow trading as the threshold approaches.
        """
        if self.stablecoin_free_monthly_eur <= 0:
            return 1.0
        raw = self.monthly_stablecoin_volume_eur / self.stablecoin_free_monthly_eur
        return float(np.clip(raw, 0.0, 1.0))

    def reset_monthly_state(
        self,
        month: Optional[tuple] = None,
        stablecoin_volume_eur: float = 0.0,
        equity_trade_count: int = 0,
    ) -> None:
        """Manually reset monthly accumulators (call at start of each month)."""
        self.monthly_stablecoin_volume_eur = max(0.0, stablecoin_volume_eur)
        self.monthly_equity_trade_count = max(0, equity_trade_count)
        now = datetime.utcnow()
        self.current_month = month or (now.year, now.month)

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _maybe_reset_month(self, now: Optional[datetime] = None) -> None:
        dt = now or datetime.utcnow()
        this_month = (dt.year, dt.month)
        if self.current_month is None:
            self.current_month = this_month
        elif self.current_month != this_month:
            self.monthly_stablecoin_volume_eur = 0.0
            self.monthly_equity_trade_count = 0
            self.current_month = this_month

    def _cost_equity(
        self, order_size_eur: float, equity_trades: int
    ) -> RevolutCostBreakdown:
        spread_bps = self.equity_spread_bps
        commission_bps = (
            0.0
            if equity_trades < self.monthly_equity_free_trades
            else self.equity_commission_bps
        )
        total_bps = spread_bps + commission_bps
        total_frac = total_bps / 10_000.0

        self.monthly_equity_trade_count += 1
        return RevolutCostBreakdown(
            asset_type="equity",
            order_eur=order_size_eur,
            spread_cost_bps=spread_bps,
            commission_bps=commission_bps,
            stablecoin_tier_fee_bps=0.0,
            total_fractional_cost=total_frac,
            monthly_stablecoin_volume_after=self.monthly_stablecoin_volume_eur,
            monthly_equity_trades_after=self.monthly_equity_trade_count,
            fee_tier_activated=False,
        )

    def _cost_stablecoin(
        self, order_size_eur: float, stablecoin_vol: float
    ) -> RevolutCostBreakdown:
        spread_bps = self.stablecoin_spread_bps

        # Split the order at the threshold to compute blended tier fee.
        remaining_free = max(0.0, self.stablecoin_free_monthly_eur - stablecoin_vol)
        volume_above_cap = max(0.0, order_size_eur - remaining_free)

        tier_fee_bps = 0.0
        tier_activated = False
        if volume_above_cap > 0.0:
            tier_fee_bps = (
                self.stablecoin_fee_above_cap_bps * volume_above_cap / order_size_eur
            )
            tier_activated = True

        total_bps = spread_bps + tier_fee_bps
        total_frac = total_bps / 10_000.0

        # Update accumulator with actuals (not the override).
        self.monthly_stablecoin_volume_eur += order_size_eur
        return RevolutCostBreakdown(
            asset_type="crypto_stablecoin",
            order_eur=order_size_eur,
            spread_cost_bps=spread_bps,
            commission_bps=0.0,
            stablecoin_tier_fee_bps=tier_fee_bps,
            total_fractional_cost=total_frac,
            monthly_stablecoin_volume_after=self.monthly_stablecoin_volume_eur,
            monthly_equity_trades_after=self.monthly_equity_trade_count,
            fee_tier_activated=tier_activated,
        )

    def _cost_crypto_other(self, order_size_eur: float) -> RevolutCostBreakdown:
        spread_bps = self.crypto_spread_bps
        total_frac = spread_bps / 10_000.0
        return RevolutCostBreakdown(
            asset_type="crypto_other",
            order_eur=order_size_eur,
            spread_cost_bps=spread_bps,
            commission_bps=0.0,
            stablecoin_tier_fee_bps=0.0,
            total_fractional_cost=total_frac,
            monthly_stablecoin_volume_after=self.monthly_stablecoin_volume_eur,
            monthly_equity_trades_after=self.monthly_equity_trade_count,
            fee_tier_activated=False,
        )
