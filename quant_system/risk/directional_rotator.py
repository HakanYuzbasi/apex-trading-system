"""
quant_system/risk/directional_rotator.py

DirectionalRotator — live Sharpe-based symbol rotation for directional slots.

Listens to ExecutionEvent fills, pairs them into round-trip trades per symbol,
computes a rolling Sharpe over the last N_TRADES trades, and identifies the
bottom performers from the directional slot universe.

Usage (called from harness at EOD):
    rotator = DirectionalRotator(event_bus)
    swaps = rotator.rotation_candidates(active_dir_slots, backbench_symbols)
    # swaps: list of (weak_slot_name, new_symbol)
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import ExecutionEvent

logger = logging.getLogger(__name__)

N_TRADES_MIN    = 15      # minimum closed trades before a slot can be rotated out
N_TRADES_WINDOW = 40      # rolling window for Sharpe calculation
SHARPE_FLOOR    = 0.20    # slots below this Sharpe are rotation candidates
N_ROTATE_MAX    = 3       # replace at most this many slots per rotation cycle


@dataclass
class _OpenLeg:
    side: str           # "buy" | "sell"
    price: float
    ts: float           # monotonic


@dataclass
class _ClosedTrade:
    pnl_pct: float      # (exit - entry) / entry, signed by side
    ts: float


class DirectionalRotator:
    """Tracks per-symbol directional trade outcomes and surfaces rotation swaps."""

    def __init__(self, event_bus: InMemoryEventBus) -> None:
        self._open_legs: dict[str, _OpenLeg]               = {}
        self._trades:    dict[str, deque[_ClosedTrade]]    = defaultdict(
            lambda: deque(maxlen=N_TRADES_WINDOW)
        )
        self._sub = event_bus.subscribe("execution", self._on_execution)

    def _on_execution(self, event: ExecutionEvent) -> None:
        sym   = event.instrument_id
        side  = event.side.lower()
        price = float(event.fill_price)
        ts    = time.monotonic()

        existing = self._open_legs.get(sym)
        if existing is None:
            # Opening a new directional leg
            if event.fill_qty and float(event.fill_qty) > 0:
                self._open_legs[sym] = _OpenLeg(side=side, price=price, ts=ts)
        else:
            # Closing — record the round-trip P&L
            if existing.price > 0:
                if existing.side == "buy":
                    pnl_pct = (price - existing.price) / existing.price
                else:
                    pnl_pct = (existing.price - price) / existing.price
                self._trades[sym].append(_ClosedTrade(pnl_pct=pnl_pct, ts=ts))
            del self._open_legs[sym]

    def rolling_sharpe(self, symbol: str) -> float | None:
        """Annualised Sharpe over last N_TRADES_WINDOW closed trades, or None."""
        trades = list(self._trades.get(symbol, []))
        if len(trades) < N_TRADES_MIN:
            return None
        rets = np.array([t.pnl_pct for t in trades])
        std  = float(np.std(rets))
        if std < 1e-9:
            return None
        # Approximate: assume ~4 round-trips per trading day → annualise
        return float(np.mean(rets) / std * np.sqrt(252 * 4))

    def rotation_candidates(
        self,
        active_dir_slots: list[tuple[str, str]],   # [(slot_name, symbol), ...]
        backbench_symbols: list[str],
    ) -> list[tuple[str, str]]:
        """
        Returns up to N_ROTATE_MAX pairs of (slot_name, replacement_symbol).
        Only considers slots with ≥ N_TRADES_MIN closed trades and Sharpe < SHARPE_FLOOR.
        Replacement symbols are drawn from backbench_symbols that are not already active.
        """
        active_symbols = {sym for _, sym in active_dir_slots}
        available      = [s for s in backbench_symbols if s not in active_symbols]
        if not available:
            return []

        # Score each active directional slot
        scored: list[tuple[float, str, str]] = []   # (sharpe, slot, symbol)
        for slot, sym in active_dir_slots:
            sh = self.rolling_sharpe(sym)
            if sh is not None and sh < SHARPE_FLOOR:
                scored.append((sh, slot, sym))

        # Sort weakest first
        scored.sort(key=lambda x: x[0])

        swaps: list[tuple[str, str]] = []
        for _, slot, _ in scored[:N_ROTATE_MAX]:
            if not available:
                break
            replacement = available.pop(0)
            swaps.append((slot, replacement))
            logger.info("DirectionalRotator: rotate %s → %s", slot, replacement)

        return swaps

    def close(self) -> None:
        self._sub.cancel()
