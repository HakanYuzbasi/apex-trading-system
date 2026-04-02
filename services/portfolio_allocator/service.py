"""
Portfolio Allocator Service - wraps CapitalAllocator with configured data_dir.
"""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from risk.capital_allocator import CapitalAllocator


class PortfolioAllocatorService:
    """Thin wrapper around CapitalAllocator for the API layer."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir or Path(
            os.getenv("APEX_ALLOCATOR_DATA_DIR", "data/portfolio_allocator")
        )
        self._allocator = CapitalAllocator(data_dir=self._data_dir)

    def update_pnl(
        self,
        equity_pnl_pct: float,
        crypto_pnl_pct: float,
        equity_trades: int = 0,
        crypto_trades: int = 0,
        trade_date: Optional[str] = None,
    ) -> dict:
        """Feed one day of P&L and return the updated allocation."""
        self._allocator.update_leg_pnl(
            equity_pnl_pct=equity_pnl_pct,
            crypto_pnl_pct=crypto_pnl_pct,
            equity_trades=equity_trades,
            crypto_trades=crypto_trades,
            trade_date=trade_date,
        )
        result = self._allocator.compute_allocation()
        return asdict(result)

    def get_allocation(self) -> dict:
        """Compute and return the current allocation recommendation."""
        result = self._allocator.compute_allocation()
        return asdict(result)

    def get_state(self) -> dict:
        """Return the full allocator state for inspection."""
        last = self._allocator._last_result
        return {
            "current_equity_frac": self._allocator.current_equity_frac,
            "current_crypto_frac": self._allocator.current_crypto_frac,
            "equity_history": [
                {"date": p.date, "pnl_pct": p.pnl_pct, "trades": p.trades}
                for p in self._allocator._equity_history
            ],
            "crypto_history": [
                {"date": p.date, "pnl_pct": p.pnl_pct, "trades": p.trades}
                for p in self._allocator._crypto_history
            ],
            "last_result": asdict(last) if last else None,
        }
