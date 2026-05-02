"""
risk/portfolio_heat.py

Portfolio "heat" tracker — aggregate dollar-risk-at-stake across all open positions.

Heat is defined as:
    heat_per_position = |entry_price - stop_price| / entry_price × notional
                      = fraction of notional that would be lost if stop is hit

This is strictly better than a raw notional cap because it accounts for how
close each position's stop is.  A $2,000 position with a 5% stop risks $100;
the same notional with a 0.5% stop risks $10.  Both are blocked the same way
by a pure notional cap, but only the first one actually threatens the book.

Usage:
    from risk.portfolio_heat import get_portfolio_heat

    # At entry — after computing stop_price
    heat = get_portfolio_heat()
    if not heat.can_open(entry_price, stop_price, notional):
        return   # too hot
    heat.register(instrument, entry_price, stop_price, notional)

    # At exit
    heat.deregister(instrument)
"""
from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

_HEAT_CAP: Final[float] = 1_500.0   # max aggregate dollar-risk across all positions


class PortfolioHeatTracker:
    """Module-level singleton tracking open dollar-risk per instrument."""

    def __init__(self) -> None:
        self._heat: dict[str, float] = {}   # instrument → dollar heat

    def register(
        self,
        instrument: str,
        entry_price: float,
        stop_price: float,
        notional: float,
    ) -> None:
        """Record open heat for a new position."""
        if entry_price <= 0:
            return
        stop_distance_pct = abs(entry_price - stop_price) / entry_price
        heat = stop_distance_pct * abs(notional)
        self._heat[instrument] = heat

    def deregister(self, instrument: str) -> None:
        """Remove heat when a position is closed."""
        self._heat.pop(instrument, None)

    def current_heat(self) -> float:
        return sum(self._heat.values())

    def can_open(
        self,
        instrument: str,
        entry_price: float,
        stop_price: float,
        notional: float,
    ) -> bool:
        """Return False if adding this position would exceed the heat cap."""
        if entry_price <= 0:
            return True
        stop_distance_pct = abs(entry_price - stop_price) / entry_price
        new_heat = stop_distance_pct * abs(notional)
        existing = self.current_heat() - self._heat.get(instrument, 0.0)
        if existing + new_heat > _HEAT_CAP:
            logger.warning(
                "HEAT CAP: skip %s — existing=$%.0f new=$%.0f total=$%.0f > cap=$%.0f",
                instrument, existing, new_heat, existing + new_heat, _HEAT_CAP,
            )
            return False
        return True

    def heat_breakdown(self) -> dict[str, float]:
        return dict(self._heat)

    def reset(self) -> None:
        """Clear all tracked heat (call at process startup to prevent stale state)."""
        self._heat.clear()


_PORTFOLIO_HEAT = PortfolioHeatTracker()


def get_portfolio_heat() -> PortfolioHeatTracker:
    return _PORTFOLIO_HEAT
