"""Tests for the Equity TWAP Executor."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.equity_twap import EquityTwapExecutor, EquityTwapResult


def _make_connector(fill_price: float = 150.0, status: str = "filled") -> MagicMock:
    connector = MagicMock()
    connector.get_market_price = AsyncMock(return_value=fill_price)
    connector.execute_order = AsyncMock(return_value={
        "status": status,
        "price": fill_price,
        "quantity": None,  # will be set per call
    })
    return connector


class TestEquityTwapExecutor:

    @pytest.mark.asyncio
    async def test_below_min_notional_returns_none(self):
        """Orders below min_notional should fall through (return None)."""
        twap = EquityTwapExecutor(min_notional=10_000.0)
        connector = _make_connector()
        result = await twap.execute(connector, "AAPL", "BUY", quantity=5, notional=750.0)
        assert result is None
        connector.execute_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_twap_gate(self):
        twap = EquityTwapExecutor(min_notional=10_000.0)
        assert twap.should_twap(quantity=100, price=200.0) is True   # $20k
        assert twap.should_twap(quantity=10, price=50.0) is False     # $500

    @pytest.mark.asyncio
    async def test_happy_path_all_slices_fill(self):
        """All slices fill: completed status, correct qty."""
        twap = EquityTwapExecutor(
            min_notional=10_000.0,
            num_slices=3,
            interval_sec=0.0,   # no delay in tests
        )
        connector = MagicMock()
        connector.get_market_price = AsyncMock(return_value=150.0)
        call_count = [0]

        async def fake_order(symbol, side, quantity, confidence, force_market=False):
            call_count[0] += 1
            return {"status": "filled", "price": 150.0, "quantity": quantity}

        connector.execute_order = fake_order

        result = await twap.execute(connector, "AAPL", "BUY", quantity=30, notional=4500.0)
        # notional < min_notional → should return None (test setup issue)
        # Let's use a large enough notional
        result = await twap.execute(connector, "AAPL", "BUY", quantity=100, notional=15_000.0)

        assert result is not None
        assert result.status == "completed"
        assert result.total_qty_filled == pytest.approx(100.0, abs=2.0)
        assert result.avg_fill_price == pytest.approx(150.0, abs=0.01)
        assert result.num_slices_sent == 3

    @pytest.mark.asyncio
    async def test_all_slices_fail_returns_failed(self):
        """If every slice fails, status should be 'failed'."""
        twap = EquityTwapExecutor(min_notional=5_000.0, num_slices=3, interval_sec=0.0)
        connector = MagicMock()
        connector.get_market_price = AsyncMock(return_value=100.0)
        connector.execute_order = AsyncMock(return_value={"status": "cancelled", "price": 0, "quantity": 0})

        result = await twap.execute(connector, "MSFT", "BUY", quantity=50, notional=5_000.0)
        assert result is not None
        assert result.status == "failed"
        assert result.total_qty_filled == 0.0

    @pytest.mark.asyncio
    async def test_adverse_move_buy_abandons_remaining_slices(self):
        """BUY: price rises > adverse_bps from arrival → abandon remaining slices."""
        twap = EquityTwapExecutor(
            min_notional=5_000.0,
            num_slices=5,
            interval_sec=0.0,
            adverse_bps=50.0,   # 0.50%
        )
        connector = MagicMock()
        prices = iter([100.0, 100.0, 100.6, 100.6, 100.6])  # +0.6% on 3rd call
        connector.get_market_price = AsyncMock(side_effect=prices)

        fills = [0]

        async def fake_order(symbol, side, quantity, confidence, force_market=False):
            fills[0] += 1
            return {"status": "filled", "price": 100.0, "quantity": quantity}

        connector.execute_order = fake_order

        result = await twap.execute(connector, "TSLA", "BUY", quantity=50, notional=5_000.0)
        assert result is not None
        assert result.status == "abandoned"
        # Should have filled fewer than all 5 slices
        assert result.num_slices_sent < 5

    @pytest.mark.asyncio
    async def test_sell_adverse_move_price_drop(self):
        """SELL: price falls > adverse_bps from arrival → abandon."""
        twap = EquityTwapExecutor(
            min_notional=5_000.0,
            num_slices=4,
            interval_sec=0.0,
            adverse_bps=50.0,
        )
        connector = MagicMock()
        # For SELL, adverse = price falls. Arrival 100.0, then drops to 99.4 (-0.6%)
        prices = iter([100.0, 100.0, 99.4, 99.4])
        connector.get_market_price = AsyncMock(side_effect=prices)

        async def fake_order(symbol, side, quantity, confidence, force_market=False):
            return {"status": "filled", "price": 100.0, "quantity": quantity}

        connector.execute_order = fake_order

        result = await twap.execute(connector, "AAPL", "SELL", quantity=40, notional=10_000.0)
        assert result is not None
        assert result.status == "abandoned"

    @pytest.mark.asyncio
    async def test_timeout_per_slice_handled(self):
        """If a slice times out, it's marked cancelled and execution continues."""
        twap = EquityTwapExecutor(
            min_notional=5_000.0,
            num_slices=2,
            interval_sec=0.0,
            timeout_per_slice_sec=0.01,  # very short timeout
        )
        connector = MagicMock()
        connector.get_market_price = AsyncMock(return_value=100.0)

        async def slow_order(symbol, side, quantity, confidence, force_market=False):
            await asyncio.sleep(1.0)   # will timeout
            return {"status": "filled", "price": 100.0, "quantity": quantity}

        connector.execute_order = slow_order

        result = await twap.execute(connector, "AAPL", "BUY", quantity=20, notional=5_000.0)
        assert result is not None
        # All slices timed out → failed
        assert result.status in ("failed", "partial")
        cancelled = [s for s in result.slices if s.status == "cancelled"]
        assert len(cancelled) > 0

    @pytest.mark.asyncio
    async def test_partial_fill_status(self):
        """If only some slices fill, status = partial."""
        twap = EquityTwapExecutor(min_notional=5_000.0, num_slices=4, interval_sec=0.0)
        connector = MagicMock()
        connector.get_market_price = AsyncMock(return_value=100.0)
        call_count = [0]

        async def mixed_order(symbol, side, quantity, confidence, force_market=False):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"status": "filled", "price": 100.0, "quantity": quantity}
            return {"status": "cancelled", "price": 0, "quantity": 0}

        connector.execute_order = mixed_order

        result = await twap.execute(connector, "AAPL", "BUY", quantity=40, notional=10_000.0)
        assert result is not None
        assert result.status == "partial"
        assert 0 < result.total_qty_filled < 40

    @pytest.mark.asyncio
    async def test_to_dict_has_required_keys(self):
        """Result.to_dict() must have keys consumed by execution_loop."""
        twap = EquityTwapExecutor(min_notional=5_000.0, num_slices=2, interval_sec=0.0)
        connector = MagicMock()
        connector.get_market_price = AsyncMock(return_value=200.0)

        async def fake_order(symbol, side, quantity, confidence, force_market=False):
            return {"status": "filled", "price": 200.0, "quantity": quantity}

        connector.execute_order = fake_order

        result = await twap.execute(connector, "AAPL", "BUY", quantity=25, notional=5_000.0)
        assert result is not None
        d = result.to_dict()
        assert "price" in d           # used as fill_price in execution_loop
        assert "quantity" in d        # used as filled quantity
        assert "status" in d
        assert "avg_fill_price" in d

    @pytest.mark.asyncio
    async def test_limit_price_buy_above_mid(self):
        """BUY limit prices should be slightly above arrival mid."""
        twap = EquityTwapExecutor(min_notional=5_000.0, num_slices=1, interval_sec=0.0, tick_offset_usd=0.01)
        lp = twap._limit_price(arrival_price=100.0, current_price=100.0, side="BUY")
        assert lp == pytest.approx(100.01, abs=0.001)

    @pytest.mark.asyncio
    async def test_limit_price_sell_below_mid(self):
        """SELL limit prices should be slightly below arrival mid."""
        twap = EquityTwapExecutor(min_notional=5_000.0, num_slices=1, interval_sec=0.0, tick_offset_usd=0.01)
        lp = twap._limit_price(arrival_price=200.0, current_price=200.0, side="SELL")
        assert lp == pytest.approx(199.99, abs=0.001)

    @pytest.mark.asyncio
    async def test_get_mid_failure_returns_none(self):
        """If price fetch fails, _get_mid returns None gracefully."""
        twap = EquityTwapExecutor()
        connector = MagicMock()
        connector.get_market_price = AsyncMock(side_effect=RuntimeError("no price"))
        result = await twap._get_mid(connector, "XYZ")
        assert result is None

    @pytest.mark.asyncio
    async def test_cannot_get_arrival_price_falls_back(self):
        """If arrival price unavailable, execute() returns None (fall back to normal order)."""
        twap = EquityTwapExecutor(min_notional=5_000.0)
        connector = MagicMock()
        connector.get_market_price = AsyncMock(return_value=None)
        result = await twap.execute(connector, "AAPL", "BUY", quantity=50, notional=10_000.0)
        assert result is None
