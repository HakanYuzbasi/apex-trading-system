"""
quant_system/execution/capital_allocator.py

Per-cycle capital allocator that prevents capital fragmentation when many
symbols fire signals simultaneously.  Instantiate once at the start of each
execution cycle; every order-size calculation must go through request() instead
of reading raw cash from the broker directly.
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

MIN_ORDER_USD: float = 50.0   # orders below this threshold are skipped


class CycleCapitalAllocator:
    """
    Tracks USD committed within a single execution cycle.

    All amounts are in USD notional.  The allocator is intentionally
    stateful and NOT thread-safe — it is designed to be used within a
    single async execution cycle.
    """

    def __init__(self, available_cash: float, max_per_symbol: float) -> None:
        if available_cash < 0:
            raise ValueError(f"available_cash must be >= 0, got {available_cash}")
        if max_per_symbol <= 0:
            raise ValueError(f"max_per_symbol must be > 0, got {max_per_symbol}")
        self._available: float = float(available_cash)
        self._max_per_symbol: float = float(max_per_symbol)
        self._committed: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request(self, symbol: str, requested_notional: float) -> float:
        """
        Approve and reserve USD for a symbol.

        Returns the approved notional (may be less than requested, or 0).
        Rules applied in order:
          1. Cap at _max_per_symbol per symbol.
          2. Cap at _available (remaining cycle budget).
          3. If approved < MIN_ORDER_USD, return 0 (skip — too small to fill).
          4. Deduct approved amount from _available.
          5. Accumulate in _committed[symbol].
        """
        requested_notional = float(requested_notional)
        if requested_notional <= 0:
            return 0.0

        # Cap 1: per-symbol maximum
        approved = min(requested_notional, self._max_per_symbol)

        # Cap 2: remaining cycle budget
        approved = min(approved, self._available)

        # Cap 3: minimum order size
        if approved < MIN_ORDER_USD:
            logger.warning(
                "Skipping %s — insufficient cycle capital "
                "(requested %.2f, remaining %.2f)",
                symbol, requested_notional, self._available,
            )
            return 0.0

        # Commit
        self._available -= approved
        self._committed[symbol] = self._committed.get(symbol, 0.0) + approved

        logger.debug(
            "CycleCapitalAllocator: approved %.2f for %s (remaining: %.2f)",
            approved, symbol, self._available,
        )
        return approved

    def committed(self) -> Dict[str, float]:
        """Return a snapshot of all committed notionals keyed by symbol."""
        return dict(self._committed)

    def remaining(self) -> float:
        """Return USD still available for this cycle."""
        return self._available
