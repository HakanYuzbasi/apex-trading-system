"""
tests/test_wash_trade_dedup.py

Tests for the wash-trade pending-cancel registry added to AlpacaBroker.
Uses __new__ to bypass constructor dependencies and injects minimal state.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from quant_system.execution.brokers.alpaca_broker import AlpacaBroker
from quant_system.events.order import OrderEvent
from datetime import datetime, timezone


def _make_broker() -> AlpacaBroker:
    """Return a bare AlpacaBroker with mocked internals."""
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._trading_client = MagicMock()
    broker._pending_cancels: dict[str, float] = {}
    broker._cycle_cancel_ids: set[str] = set()
    broker._orders_by_client_order_id = {}
    broker._orders_by_venue_order_id = {}
    broker._filled_qty_by_client_order_id = {}
    broker._limit_chaser = None
    return broker


def _wash_exc(existing_order_id: str) -> Exception:
    """Simulate Alpaca's wash-trade HTTP error body."""
    body = (
        f'{{"code": 40310000, "message": "wash trade", '
        f'"existing_order_id": "{existing_order_id}"}}'
    )
    return Exception(f"422 Unprocessable Entity: {body}")


def _make_order_event(symbol: str = "CRYPTO:BTC/USD") -> OrderEvent:
    now = datetime.now(timezone.utc)
    return OrderEvent(
        instrument_id=symbol,
        exchange_ts=now, received_ts=now, processed_ts=now,
        sequence_id=0, source="test",
        order_id="oid-001", order_action="submit", order_scope="parent",
        side="buy", order_type="limit", quantity=0.01,
        time_in_force="gtc", execution_algo="direct", limit_price=50_000.0,
    )


# ── test_cancel_not_sent_twice_same_cycle ──────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_not_sent_twice_same_cycle():
    """Second wash-trade rejection with same conflict ID in same cycle must not cancel again."""
    broker = _make_broker()
    conflict_id = "conf-aaa"
    broker._trading_client.submit_order = MagicMock(side_effect=_wash_exc(conflict_id))
    broker._trading_client.cancel_order_by_id = MagicMock()
    broker._orders_by_client_order_id = {}

    event = _make_order_event()

    # First call — cancel should fire
    with patch.object(broker, "_publish_rejection", new=AsyncMock()):
        with patch.object(broker, "_build_order_request", return_value=MagicMock()):
            await broker._submit_direct_order(event)

    assert broker._trading_client.cancel_order_by_id.call_count == 1
    assert conflict_id in broker._pending_cancels

    # Second call same cycle — cancel must NOT fire again
    with patch.object(broker, "_publish_rejection", new=AsyncMock()):
        with patch.object(broker, "_build_order_request", return_value=MagicMock()):
            await broker._submit_direct_order(event)

    # Still only 1 call (same cycle dedup via _cycle_cancel_ids)
    assert broker._trading_client.cancel_order_by_id.call_count == 1


# ── test_cancel_not_retried_within_90s ────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_not_retried_within_90s():
    """A conflict ID already in _pending_cancels within 90s window must not be cancelled again."""
    broker = _make_broker()
    conflict_id = "conf-bbb"
    broker._pending_cancels[conflict_id] = time.monotonic()   # just sent

    broker._trading_client.submit_order = MagicMock(side_effect=_wash_exc(conflict_id))
    broker._trading_client.cancel_order_by_id = MagicMock()
    event = _make_order_event()

    with patch.object(broker, "_publish_rejection", new=AsyncMock()):
        with patch.object(broker, "_build_order_request", return_value=MagicMock()):
            await broker._submit_direct_order(event)

    broker._trading_client.cancel_order_by_id.assert_not_called()


# ── test_already_filled_suppresses_retry ─────────────────────────────────────

@pytest.mark.asyncio
async def test_already_filled_suppresses_retry():
    """When cancel raises 'already filled' (42210000) the ID must be removed from pending."""
    broker = _make_broker()
    conflict_id = "conf-ccc"
    broker._pending_cancels[conflict_id] = time.monotonic() - 200.0  # expired entry

    fill_exc = Exception("422: 42210000 order already filled")
    broker._trading_client.submit_order = MagicMock(side_effect=_wash_exc(conflict_id))
    broker._trading_client.cancel_order_by_id = MagicMock(side_effect=fill_exc)
    event = _make_order_event()

    with patch.object(broker, "_publish_rejection", new=AsyncMock()):
        with patch.object(broker, "_build_order_request", return_value=MagicMock()):
            await broker._submit_direct_order(event)

    # Entry must be removed — filled orders don't need retry
    assert conflict_id not in broker._pending_cancels


# ── test_pending_cancel_purged_after_120s ────────────────────────────────────

@pytest.mark.asyncio
async def test_pending_cancel_purged_after_120s():
    """Entries older than 120s are purged from _pending_cancels at the start of the next wash-trade handling."""
    broker = _make_broker()
    old_id = "conf-old"
    new_id = "conf-new"

    # Seed one stale entry (> 120s) and one fresh entry
    broker._pending_cancels[old_id] = time.monotonic() - 130.0
    broker._pending_cancels[new_id] = time.monotonic() - 10.0

    # Trigger any wash-trade rejection so the purge logic runs
    some_conflict = "conf-trigger"
    broker._trading_client.submit_order = MagicMock(side_effect=_wash_exc(some_conflict))
    broker._trading_client.cancel_order_by_id = MagicMock()
    event = _make_order_event()

    with patch.object(broker, "_publish_rejection", new=AsyncMock()):
        with patch.object(broker, "_build_order_request", return_value=MagicMock()):
            await broker._submit_direct_order(event)

    # The old entry must be gone; the fresh one must survive
    assert old_id not in broker._pending_cancels
    assert new_id in broker._pending_cancels


# ── test fill/cancel confirmation clears pending_cancels ─────────────────────

@pytest.mark.asyncio
async def test_fill_confirmation_clears_pending_cancel():
    """Receiving a fill trade update removes the venue_order_id from _pending_cancels."""
    broker = _make_broker()
    venue_id = "venue-fff"
    broker._pending_cancels[venue_id] = time.monotonic()

    # Minimal trade-update payload simulating a fill
    payload = {
        "event": "fill",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "qty": "0.01",
        "price": "50000.00",
        "order": {
            "id": venue_id,
            "client_order_id": "coid-x",
            "symbol": "BTCUSD",
            "side": "buy",
            "filled_qty": "0.01",
            "filled_avg_price": "50000.00",
            "qty": "0.01",
        },
    }
    broker._event_bus = MagicMock()
    broker._event_bus.publish_async = AsyncMock()
    from itertools import count as _count
    broker._sequence = _count()
    broker._filled_qty_by_client_order_id = {}
    broker._latest_market = {}
    broker._orders_by_client_order_id = {}
    broker._orders_by_venue_order_id = {}

    # Call _on_trade_update directly
    await broker._on_trade_update(payload)

    assert venue_id not in broker._pending_cancels
