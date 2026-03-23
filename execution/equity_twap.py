"""
execution/equity_twap.py — Equity TWAP Executor

Slices large equity orders into time-weighted limit order tranches to reduce
market impact and improve average fill price.

Strategy:
  1. Divide total quantity into N equal slices (default 5).
  2. Every interval_sec, get the current mid price and post a limit order
     at mid ± tick_offset ticks (aggressive enough to fill within the interval).
  3. If the market moves adversely by more than adverse_bps from arrival price,
     cancel remaining slices and accept what has filled.
  4. Returns a TwapResult compatible with the existing order result shape.

Integration in execution_loop.py:
    from execution.equity_twap import EquityTwapExecutor
    twap = EquityTwapExecutor()
    result = await twap.execute(connector, symbol, side, quantity, notional)
    if result:
        fill_price = result.avg_fill_price
        filled_qty = result.total_qty_filled
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default tick offset for limit orders: 1 cent for US equities
_DEFAULT_TICK_USD = 0.01


@dataclass
class EquitySliceResult:
    slice_num: int
    qty: float
    limit_price: float
    fill_price: float
    filled_qty: float
    status: str          # "filled" | "partial" | "failed" | "skipped" | "cancelled"
    elapsed_ms: float


@dataclass
class EquityTwapResult:
    symbol: str
    side: str
    total_qty_requested: float
    total_qty_filled: float
    avg_fill_price: float
    num_slices_sent: int
    num_slices_filled: int
    slices: List[EquitySliceResult] = field(default_factory=list)
    elapsed_total_sec: float = 0.0
    status: str = "completed"   # "completed" | "partial" | "failed" | "abandoned"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "total_qty_requested": self.total_qty_requested,
            "total_qty_filled": self.total_qty_filled,
            "avg_fill_price": self.avg_fill_price,
            "num_slices": self.num_slices_sent,
            "status": self.status,
            # Compat with single-order result shape consumed by execution_loop
            "price": self.avg_fill_price,
            "quantity": self.total_qty_filled,
        }


class EquityTwapExecutor:
    """
    Time-Weighted Average Price executor for IBKR equity orders.

    For orders below min_notional, falls through immediately (returns None so
    the caller uses the normal single market/limit order path).

    For orders >= min_notional:
      - Splits qty into num_slices equal pieces
      - Places each as a passive limit order at arrival_mid ± tick_offset
      - Waits interval_sec between slices
      - Abandons if adverse move > adverse_bps from arrival price
    """

    def __init__(
        self,
        min_notional: float = 10_000.0,   # Only TWAP orders this large
        num_slices: int = 5,
        interval_sec: float = 60.0,        # 1 minute between slices
        adverse_bps: float = 50.0,         # 0.50% adverse → abandon
        tick_offset_usd: float = _DEFAULT_TICK_USD,
        timeout_per_slice_sec: float = 55.0,  # Cancel & move on if unfilled
    ):
        self.min_notional = min_notional
        self.num_slices = num_slices
        self.interval_sec = interval_sec
        self.adverse_bps = adverse_bps
        self.tick_offset_usd = tick_offset_usd
        self.timeout_per_slice_sec = timeout_per_slice_sec

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def execute(
        self,
        connector: Any,           # IBKRConnector instance
        symbol: str,
        side: str,                # "BUY" or "SELL"
        quantity: float,          # Total shares to execute
        notional: float,          # Estimated dollar value
        confidence: float = 0.5,
    ) -> Optional[EquityTwapResult]:
        """
        Execute a TWAP if notional >= min_notional.

        Returns None if notional is below threshold (caller uses normal path).
        Returns EquityTwapResult otherwise (may be partial if abandoned early).
        """
        if notional < self.min_notional:
            return None

        logger.info(
            "🕐 EquityTWAP starting: %s %s %.0f shares @ ~$%.2f notional  "
            "[%d slices × %ds]",
            side, symbol, quantity, notional, self.num_slices, int(self.interval_sec),
        )

        start_t = time.monotonic()
        arrival_price = await self._get_mid(connector, symbol)
        if not arrival_price:
            logger.warning("EquityTWAP: cannot get arrival price for %s — falling back", symbol)
            return None

        slice_qty = quantity / self.num_slices
        slice_qty = max(1.0, round(slice_qty))  # at least 1 share per slice

        slices: List[EquitySliceResult] = []
        total_filled_qty = 0.0
        total_filled_value = 0.0
        num_slices_sent = 0
        abandoned = False

        for i in range(self.num_slices):
            if i > 0:
                await asyncio.sleep(self.interval_sec)

            # Check for adverse move
            cur_price = await self._get_mid(connector, symbol)
            if cur_price and arrival_price:
                move_bps = abs(cur_price - arrival_price) / arrival_price * 10_000
                direction_adverse = (
                    (side == "BUY" and cur_price > arrival_price) or
                    (side == "SELL" and cur_price < arrival_price)
                )
                if direction_adverse and move_bps > self.adverse_bps:
                    logger.warning(
                        "EquityTWAP: adverse move %.1f bps on %s — abandoning remaining %d slices",
                        move_bps, symbol, self.num_slices - i,
                    )
                    abandoned = True
                    break

            # Determine limit price for this slice
            limit_price = self._limit_price(arrival_price, cur_price, side)

            # Remaining qty for last slice (absorb rounding)
            remaining = quantity - total_filled_qty
            this_qty = slice_qty if i < self.num_slices - 1 else max(1.0, round(remaining))
            if this_qty <= 0:
                break

            slice_t0 = time.monotonic()
            num_slices_sent += 1

            try:
                result = await asyncio.wait_for(
                    connector.execute_order(
                        symbol=symbol,
                        side=side,
                        quantity=this_qty,
                        confidence=confidence,
                        force_market=False,
                    ),
                    timeout=self.timeout_per_slice_sec,
                )
                elapsed_ms = (time.monotonic() - slice_t0) * 1000

                if result and result.get("status") in ("filled", "Filled", "partial_fill"):
                    fill_p = float(result.get("price", limit_price))
                    fill_q = float(result.get("quantity", this_qty))
                    total_filled_qty += fill_q
                    total_filled_value += fill_p * fill_q
                    slices.append(EquitySliceResult(
                        slice_num=i + 1,
                        qty=this_qty,
                        limit_price=limit_price,
                        fill_price=fill_p,
                        filled_qty=fill_q,
                        status="filled",
                        elapsed_ms=elapsed_ms,
                    ))
                    logger.info(
                        "EquityTWAP slice %d/%d: filled %.0f of %.0f @ $%.4f",
                        i + 1, self.num_slices, fill_q, this_qty, fill_p,
                    )
                else:
                    slices.append(EquitySliceResult(
                        slice_num=i + 1, qty=this_qty, limit_price=limit_price,
                        fill_price=0.0, filled_qty=0.0, status="failed",
                        elapsed_ms=(time.monotonic() - slice_t0) * 1000,
                    ))
                    logger.warning("EquityTWAP slice %d/%d: no fill", i + 1, self.num_slices)

            except asyncio.TimeoutError:
                slices.append(EquitySliceResult(
                    slice_num=i + 1, qty=this_qty, limit_price=limit_price,
                    fill_price=0.0, filled_qty=0.0, status="cancelled",
                    elapsed_ms=self.timeout_per_slice_sec * 1000,
                ))
                logger.warning("EquityTWAP slice %d/%d: timed out after %.0fs", i + 1, self.num_slices, self.timeout_per_slice_sec)
            except Exception as exc:
                slices.append(EquitySliceResult(
                    slice_num=i + 1, qty=this_qty, limit_price=limit_price,
                    fill_price=0.0, filled_qty=0.0, status="failed",
                    elapsed_ms=(time.monotonic() - slice_t0) * 1000,
                ))
                logger.warning("EquityTWAP slice %d/%d: error: %s", i + 1, self.num_slices, exc)

        elapsed_total = time.monotonic() - start_t
        avg_price = total_filled_value / total_filled_qty if total_filled_qty > 0 else 0.0
        num_filled = sum(1 for s in slices if s.status == "filled")

        if total_filled_qty == 0:
            status = "failed"
        elif abandoned:
            status = "abandoned"
        elif total_filled_qty < quantity * 0.95:
            status = "partial"
        else:
            status = "completed"

        result = EquityTwapResult(
            symbol=symbol,
            side=side,
            total_qty_requested=quantity,
            total_qty_filled=total_filled_qty,
            avg_fill_price=avg_price,
            num_slices_sent=num_slices_sent,
            num_slices_filled=num_filled,
            slices=slices,
            elapsed_total_sec=round(elapsed_total, 1),
            status=status,
        )

        logger.info(
            "🕐 EquityTWAP complete: %s %s %.0f/%.0f shares @ avg $%.4f  status=%s  time=%.0fs",
            side, symbol, total_filled_qty, quantity, avg_price, status, elapsed_total,
        )
        return result

    def should_twap(self, quantity: float, price: float) -> bool:
        """Quick check: should this order use TWAP?"""
        return quantity * price >= self.min_notional

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_mid(self, connector: Any, symbol: str) -> Optional[float]:
        """Get current mid price from connector."""
        try:
            price = await connector.get_market_price(symbol)
            return float(price) if price else None
        except Exception as exc:
            logger.debug("EquityTWAP: get_market_price(%s) failed: %s", symbol, exc)
            return None

    def _limit_price(
        self,
        arrival_price: float,
        current_price: Optional[float],
        side: str,
    ) -> float:
        """
        Compute limit price: reference mid ± one tick.
        Slightly aggressive so the order fills within the interval.
        BUY:  arrival_mid + 1 tick  (cross the spread to get filled)
        SELL: arrival_mid - 1 tick
        """
        ref = current_price if current_price else arrival_price
        if side == "BUY":
            return round(ref + self.tick_offset_usd, 4)
        else:
            return round(ref - self.tick_offset_usd, 4)
