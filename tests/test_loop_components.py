from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, ExecutionEvent, OrderEvent, SignalEvent, TradeTick
from quant_system.execution.brokers.paper import PaperBroker
from quant_system.instruments.instrument import Instrument
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.manager import RiskManager
from quant_system.strategies.base import BaseStrategy
from core.symbols import AssetClass


class DummyStrategy(BaseStrategy):
    def __init__(self, event_bus: InMemoryEventBus) -> None:
        super().__init__(event_bus)
        self.bar_calls = 0
        self.tick_calls = 0

    def on_bar(self, event: BarEvent) -> None:
        self.bar_calls += 1

    def on_tick(self, event: TradeTick) -> None:
        self.tick_calls += 1


def test_base_strategy_subscribes_and_emits_signal_synchronously() -> None:
    bus = InMemoryEventBus()
    strategy = DummyStrategy(bus)
    captured: list[SignalEvent] = []

    def on_signal(event) -> None:
        captured.append(event)

    bus.subscribe("signal", on_signal)

    now = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    bus.publish(
        BarEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            volume=1_000_000,
        )
    )
    bus.publish(
        TradeTick(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="test",
            last_price=100.5,
            last_size=100.0,
            aggressor_side="buy",
            trade_id="t-1",
        )
    )
    signal = strategy.emit_signal("AAPL", "units", 10.0, 0.8)

    assert strategy.bar_calls == 1
    assert strategy.tick_calls == 1
    assert captured[0] == signal
    strategy.close()


def test_portfolio_ledger_updates_cash_and_positions() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=10_000.0)
    now = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)

    bus.publish(
        ExecutionEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            order_id="ord-1",
            side="buy",
            execution_status="filled",
            fill_qty=10.0,
            fill_price=100.0,
            fees=1.0,
            slippage=0.01,
            remaining_qty=0.0,
        )
    )

    position = ledger.get_position("AAPL")
    assert ledger.cash == pytest.approx(8_999.0)
    assert position.quantity == pytest.approx(10.0)
    assert position.avg_price == pytest.approx(100.0)


def test_portfolio_ledger_tracks_short_liability_and_cover_pnl() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=10_000.0)
    now = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)

    bus.publish(
        ExecutionEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            order_id="ord-short",
            side="sell",
            execution_status="filled",
            fill_qty=10.0,
            fill_price=100.0,
            fees=1.0,
            slippage=0.01,
            remaining_qty=0.0,
        )
    )
    bus.publish(
        BarEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="test",
            open_price=90.0,
            high_price=90.0,
            low_price=90.0,
            close_price=90.0,
            volume=100.0,
        )
    )

    position = ledger.get_position("AAPL")
    assert position.quantity == pytest.approx(-10.0)
    assert ledger.cash == pytest.approx(10_999.0)
    assert ledger.total_short_liability() == pytest.approx(900.0)
    assert ledger.total_unrealized_pnl() == pytest.approx(100.0)
    assert ledger.total_equity() == pytest.approx(10_099.0)

    bus.publish(
        ExecutionEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=3,
            source="test",
            order_id="ord-cover",
            side="buy",
            execution_status="filled",
            fill_qty=10.0,
            fill_price=90.0,
            fees=1.0,
            slippage=0.01,
            remaining_qty=0.0,
        )
    )

    assert position.quantity == pytest.approx(0.0)
    assert position.realized_pnl == pytest.approx(98.0)
    assert ledger.cash == pytest.approx(10_098.0)
    assert ledger.total_equity() == pytest.approx(10_098.0)


def test_risk_manager_maps_notional_signal_to_quantity_using_market_price() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=10_000.0)
    risk_manager = RiskManager(ledger, bus)
    orders: list[OrderEvent] = []
    now = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)

    def on_order(event) -> None:
        orders.append(event)

    bus.subscribe("order", on_order)
    bus.publish(
        BarEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.0,
            volume=1_000_000,
        )
    )
    bus.publish(
        SignalEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="strategy.test",
            strategy_id="Demo",
            side="buy",
            target_type="notional",
            target_value=1_000.0,
            confidence=0.9,
            stop_model="none",
            stop_params={},
        )
    )

    assert len(orders) == 1
    assert orders[0].quantity == pytest.approx(10.0)

    risk_manager.close()
    ledger.close()


def test_risk_manager_allows_short_when_margin_is_available() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=10_000.0)
    risk_manager = RiskManager(ledger, bus)
    orders: list[OrderEvent] = []
    now = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)

    bus.subscribe("order", lambda event: orders.append(event))
    bus.publish(
        BarEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.0,
            volume=1_000_000,
        )
    )
    bus.publish(
        SignalEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="strategy.test",
            strategy_id="Demo",
            side="sell",
            target_type="notional",
            target_value=-1_000.0,
            confidence=0.9,
            stop_model="none",
            stop_params={},
        )
    )

    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].quantity == pytest.approx(10.0)

    risk_manager.close()
    ledger.close()


def test_risk_manager_rejects_non_shortable_instrument() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=10_000.0)
    risk_manager = RiskManager(
        ledger,
        bus,
        instruments={
            "AAPL": Instrument(
                instrument_id="AAPL",
                asset_class=AssetClass.EQUITY,
                shortable=False,
            )
        },
    )
    orders: list[OrderEvent] = []
    now = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)

    bus.subscribe("order", lambda event: orders.append(event))
    bus.publish(
        BarEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.0,
            volume=1_000_000,
        )
    )
    bus.publish(
        SignalEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="strategy.test",
            strategy_id="Demo",
            side="sell",
            target_type="notional",
            target_value=-1_000.0,
            confidence=0.9,
            stop_model="none",
            stop_params={},
        )
    )

    assert orders == []

    risk_manager.close()
    ledger.close()


def test_risk_manager_enforces_asset_class_margin_rates() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=1_000.0)
    risk_manager = RiskManager(
        ledger,
        bus,
        instruments={
            "CRYPTO:BTC/USD": Instrument(
                instrument_id="CRYPTO:BTC/USD",
                asset_class=AssetClass.CRYPTO,
            )
        },
    )
    orders: list[OrderEvent] = []
    now = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)

    bus.subscribe("order", lambda event: orders.append(event))
    bus.publish(
        BarEvent(
            instrument_id="CRYPTO:BTC/USD",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.0,
            volume=1_000_000,
        )
    )
    bus.publish(
        SignalEvent(
            instrument_id="CRYPTO:BTC/USD",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="strategy.test",
            strategy_id="Demo",
            side="buy",
            target_type="notional",
            target_value=2_000.0,
            confidence=0.9,
            stop_model="none",
            stop_params={},
        )
    )

    assert orders == []

    risk_manager.close()
    ledger.close()


@pytest.mark.asyncio
async def test_risk_manager_closes_loop_into_paper_broker_and_ledger() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=5_000.0)
    risk_manager = RiskManager(ledger, bus)
    broker = PaperBroker(bus, simulated_latency_ms=0.0)
    executions: list[ExecutionEvent] = []

    def on_execution(event) -> None:
        executions.append(event)

    bus.subscribe("execution", on_execution)

    now = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    await bus.publish_async(
        TradeTick(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=1,
            source="test",
            last_price=100.0,
            last_size=100.0,
            aggressor_side="buy",
            trade_id="t-1",
        )
    )

    signal = SignalEvent(
        instrument_id="AAPL",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=2,
        source="strategy.test",
        strategy_id="Demo",
        side="buy",
        target_type="units",
        target_value=10.0,
        confidence=0.9,
        stop_model="none",
        stop_params={},
    )

    await bus.publish_async(signal)
    await asyncio.sleep(0.01)

    position = ledger.get_position("AAPL")
    assert len(executions) == 1
    assert executions[0].execution_status == "filled"
    assert position.quantity == pytest.approx(10.0)
    assert ledger.cash < 5_000.0

    broker.close()
    risk_manager.close()
    ledger.close()
