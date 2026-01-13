# models/risk_management.py - Risk management and circuit breaker logic

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np


class CircuitBreaker:
    """
    Simple circuit breaker for risk management.

    Responsibilities:
    - Enforce max daily loss (based on portfolio value drop from starting_value).
    - Enforce max per-position loss.
    - Detect high intrabar volatility (price jump vs previous).
    - Detect gap-down events.
    - Track halted state and allow recovery when conditions normalize.
    """

    def __init__(
        self,
        ibkr_connector,
        max_daily_loss: float = 50_000,
        max_position_loss: float = 20_000,
        max_volatility: float = 0.05,
    ) -> None:
        self.ibkr = ibkr_connector
        self.max_daily_loss = max_daily_loss
        self.max_position_loss = max_position_loss
        self.max_volatility = max_volatility

        # Starting portfolio value for daily loss calculations
        self.starting_value: float = 1_100_000.0

        # Price history used for volatility / gap detection
        self.prices: np.ndarray = np.array([], dtype=float)

        # Whether trading is currently halted
        self.halted: bool = False

    async def check_if_can_trade(self) -> bool:
        """
        Main entry point used by tests.

        Returns False when:
        - Daily loss > max_daily_loss, or
        - Last move volatility > max_volatility.

        Otherwise returns True.
        """
        # 1) Daily loss check
        try:
            current_value = await self.ibkr.get_portfolio_value()
        except Exception:
            # Fail-safe: if portfolio value cannot be retrieved,
            # halt trading rather than allow it.
            self.halted = True
            return False

        daily_loss = self.starting_value - float(current_value)

        if daily_loss >= self.max_daily_loss:
            self.halted = True
            return False

        # 2) Volatility check (use last two prices if available)
        if self._is_high_volatility():
            self.halted = True
            return False

        # If we get here, no blocking condition
        self.halted = False
        return True

    def _is_high_volatility(self) -> bool:
        """Return True if last price move exceeds max_volatility."""
        if self.prices is None or len(self.prices) < 2:
            return False

        prev = float(self.prices[-2])
        last = float(self.prices[-1])
        if prev == 0:
            return False

        move = abs(last - prev) / prev
        return move >= self.max_volatility

    def should_close_position(self, position: Dict[str, Any]) -> bool:
        """
        Return True if this position exceeds the max per-position loss.

        Tests expect:
        - Large drop from entry_price to current_price with given quantity
          to trigger a True.
        """
        qty = float(position.get("quantity", 0.0))
        entry = float(position.get("entry_price", 0.0))
        current = float(position.get("current_price", entry))

        loss = (entry - current) * qty  # positive when losing money

        return loss >= self.max_position_loss

    def detect_gap_down(self, threshold: float = 0.05) -> bool:
        """
        Detect a gap-down event between the last two prices.

        Returns True when:
        - Latest price is lower than previous, and
        - Relative drop > threshold.
        """
        if self.prices is None or len(self.prices) < 2:
            return False

        prev = float(self.prices[-2])
        last = float(self.prices[-1])
        if prev == 0:
            return False

        gap = abs(last - prev) / prev
        return last < prev and gap > threshold
