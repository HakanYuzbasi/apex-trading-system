"""
execution/crypto_twap.py — Crypto TWAP Executor

Breaks large fractional crypto orders into time-weighted slices to reduce
market impact on thin crypto books.

Integration in execution_loop.py:
    from execution.crypto_twap import CryptoTwapExecutor
    _twap = CryptoTwapExecutor()
    trade = await _twap.execute(connector, symbol, side, total_qty, num_slices=4, interval_sec=30)
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TwapSliceResult:
    slice_num: int
    qty: float
    fill_price: float
    status: str  # "filled" | "failed" | "skipped"
    elapsed_ms: float


@dataclass
class TwapResult:
    symbol: str
    side: str
    total_qty_requested: float
    total_qty_filled: float
    avg_fill_price: float
    num_slices_sent: int
    num_slices_filled: int
    slices: List[TwapSliceResult] = field(default_factory=list)
    elapsed_total_sec: float = 0.0
    status: str = "completed"  # "completed" | "partial" | "failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "total_qty_requested": self.total_qty_requested,
            "total_qty_filled": self.total_qty_filled,
            "avg_fill_price": self.avg_fill_price,
            "num_slices": self.num_slices_sent,
            "status": self.status,
            "price": self.avg_fill_price,  # compat with single-order result shape
        }


class CryptoTwapExecutor:
    """
    Time-Weighted Average Price executor for crypto fractional orders.

    Splits a large order into `num_slices` equal-sized child orders placed
    `interval_sec` apart. Adaptive: stops early if market moves against the
    trade by more than `abandon_pct` from arrival price.

    Suitable for Alpaca crypto connector (fractional qty, async execute_order).
    """

    def __init__(
        self,
        default_num_slices: int = 4,
        default_interval_sec: float = 30.0,
        abandon_pct: float = 0.005,   # 0.5% adverse move → abandon remaining slices
        min_slice_qty: float = 1e-6,  # fractional floor
    ):
        self.default_num_slices = default_num_slices
        self.default_interval_sec = default_interval_sec
        self.abandon_pct = abandon_pct
        self.min_slice_qty = min_slice_qty

    def _n_slices(self, notional: float) -> int:
        """Scale slice count by order size."""
        if notional < 5_000:
            return 2
        elif notional < 20_000:
            return 4
        elif notional < 50_000:
            return 6
        else:
            return 8

    async def execute(
        self,
        connector: Any,
        symbol: str,
        side: str,
        total_qty: float,
        current_price: float,
        num_slices: Optional[int] = None,
        interval_sec: Optional[float] = None,
        confidence: float = 0.5,
    ) -> TwapResult:
        """
        Execute a crypto order using TWAP slicing.

        Args:
            connector: Alpaca connector with async `execute_order(symbol, side, quantity, confidence)`.
            symbol: Crypto symbol e.g. "BTC/USD".
            side: "BUY" or "SELL".
            total_qty: Total fractional quantity to execute.
            current_price: Arrival price for adverse-move check.
            num_slices: Override slice count (auto-computed from notional if None).
            interval_sec: Seconds between slices (default: self.default_interval_sec).
            confidence: Passed to each slice's execute_order call.

        Returns:
            TwapResult with fill summary.
        """
        notional = total_qty * max(current_price, 1.0)
        n = num_slices or self._n_slices(notional)
        interval = interval_sec if interval_sec is not None else self.default_interval_sec

        slice_qty = total_qty / n
        if slice_qty < self.min_slice_qty:
            # Too small to slice — execute whole order at once
            n = 1
            slice_qty = total_qty

        result = TwapResult(
            symbol=symbol,
            side=side,
            total_qty_requested=total_qty,
            total_qty_filled=0.0,
            avg_fill_price=0.0,
            num_slices_sent=0,
            num_slices_filled=0,
        )

        logger.info(
            "TWAP: %s %s %.6f @ ~%.4f in %d slices × %.0fs",
            side, symbol, total_qty, current_price, n, interval,
        )

        _t0 = time.time()
        weighted_price_sum = 0.0

        for i in range(n):
            # Adverse move check: if market has moved against us by > abandon_pct, stop
            if i > 0 and current_price > 0:
                try:
                    _live_price = await self._fetch_live_price(connector, symbol)
                    if _live_price > 0:
                        _move = (_live_price - current_price) / current_price
                        _adverse = _move < -self.abandon_pct if side.upper() == "BUY" else _move > self.abandon_pct
                        if _adverse:
                            logger.warning(
                                "TWAP: %s adverse move %.2f%% — abandoning %d remaining slices",
                                symbol, _move * 100, n - i,
                            )
                            result.status = "partial"
                            break
                except Exception:
                    pass  # live price fetch failure is non-fatal

            # Execute slice
            _slice_t0 = time.time()
            try:
                trade = await connector.execute_order(
                    symbol=symbol,
                    side=side,
                    quantity=slice_qty,
                    confidence=confidence,
                )
                _elapsed_ms = (time.time() - _slice_t0) * 1000

                if trade:
                    _fill_price = float(
                        trade.get("price", 0) if isinstance(trade, dict)
                        else getattr(trade, "avgFillPrice", 0) or 0
                    )
                    _status = "filled" if _fill_price > 0 else "skipped"
                    if _fill_price > 0:
                        result.total_qty_filled += slice_qty
                        weighted_price_sum += slice_qty * _fill_price
                        result.num_slices_filled += 1
                else:
                    _fill_price = 0.0
                    _status = "failed"

                result.slices.append(TwapSliceResult(
                    slice_num=i + 1,
                    qty=slice_qty,
                    fill_price=_fill_price,
                    status=_status,
                    elapsed_ms=_elapsed_ms,
                ))
                result.num_slices_sent += 1

                logger.debug(
                    "  TWAP slice %d/%d: %.6f @ %.4f (%s, %.0fms)",
                    i + 1, n, slice_qty, _fill_price, _status, _elapsed_ms,
                )

            except Exception as _e:
                logger.warning("TWAP slice %d failed for %s: %s", i + 1, symbol, _e)
                result.slices.append(TwapSliceResult(
                    slice_num=i + 1, qty=slice_qty, fill_price=0.0,
                    status="failed", elapsed_ms=0.0,
                ))
                result.num_slices_sent += 1

            # Wait between slices (skip after last slice)
            if i < n - 1:
                await asyncio.sleep(interval)

        # Compute average fill price
        if result.total_qty_filled > 0:
            result.avg_fill_price = weighted_price_sum / result.total_qty_filled
        else:
            result.status = "failed"

        result.elapsed_total_sec = time.time() - _t0
        logger.info(
            "TWAP done: %s %s filled=%.6f/%.6f avg=%.4f status=%s",
            side, symbol, result.total_qty_filled, total_qty,
            result.avg_fill_price, result.status,
        )
        return result

    async def _fetch_live_price(self, connector: Any, symbol: str) -> float:
        """Try to get a live quote price from the connector."""
        try:
            if hasattr(connector, 'get_current_price'):
                return float(await connector.get_current_price(symbol) or 0)
            if hasattr(connector, 'get_quote'):
                q = await connector.get_quote(symbol)
                if q:
                    return float(q.get('ask', 0) or q.get('last', 0) or 0)
        except Exception:
            pass
        return 0.0
