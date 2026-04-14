from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BarEvent, ExecutionEvent, QuoteTick, TradeTick


@dataclass(slots=True)
class Position:
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0


class PortfolioLedger:
    def __init__(self, event_bus: InMemoryEventBus, starting_cash: float) -> None:
        if starting_cash < 0:
            raise ValueError("starting_cash must be non-negative")
        self.event_bus = event_bus
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.positions: Dict[str, Position] = {}
        self.last_price_by_instrument: Dict[str, float] = {}
        self._subscriptions: tuple[Subscription, ...] = (
            self.event_bus.subscribe("bar", self._on_bar),
            self.event_bus.subscribe("quote_tick", self._on_quote_tick),
            self.event_bus.subscribe("trade_tick", self._on_trade_tick),
            self.event_bus.subscribe("execution", self._on_execution),
        )

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return self._subscriptions

    def close(self) -> None:
        for subscription in self._subscriptions:
            self.event_bus.unsubscribe(subscription.token)

    def get_position(self, instrument_id: str) -> Position:
        return self.positions.setdefault(instrument_id, Position())

    def get_unrealized_pnl(self, instrument_id: str) -> float:
        pos = self.positions.get(instrument_id)
        if not pos or abs(pos.quantity) < 1e-12:
            return 0.0
        ref = self.get_reference_price(instrument_id)
        if ref is None:
            return 0.0
        if pos.quantity > 0:
            return pos.quantity * (ref - pos.avg_price)
        return abs(pos.quantity) * (pos.avg_price - ref)

    def get_reference_price(self, instrument_id: str) -> float | None:
        price = self.last_price_by_instrument.get(instrument_id)
        if price is None:
            position = self.positions.get(instrument_id)
            if position and position.avg_price > 0:
                return position.avg_price
            return None
        return price

    def total_realized_pnl(self) -> float:
        return sum(position.realized_pnl for position in self.positions.values())

    def total_market_value(self) -> float:
        market_value = 0.0
        for instrument_id, position in self.positions.items():
            if abs(position.quantity) < 1e-12:
                continue
            reference_price = self.get_reference_price(instrument_id)
            if reference_price is None:
                continue
            market_value += position.quantity * reference_price
        return market_value

    def total_long_market_value(self) -> float:
        market_value = 0.0
        for instrument_id, position in self.positions.items():
            if position.quantity <= 1e-12:
                continue
            reference_price = self.get_reference_price(instrument_id)
            if reference_price is None:
                continue
            market_value += position.quantity * reference_price
        return market_value

    def total_short_liability(self) -> float:
        liability = 0.0
        for instrument_id, position in self.positions.items():
            if position.quantity >= -1e-12:
                continue
            reference_price = self.get_reference_price(instrument_id)
            if reference_price is None:
                continue
            liability += abs(position.quantity) * reference_price
        return liability

    def total_gross_market_value(self) -> float:
        gross_market_value = 0.0
        for instrument_id, position in self.positions.items():
            if abs(position.quantity) < 1e-12:
                continue
            reference_price = self.get_reference_price(instrument_id)
            if reference_price is None:
                continue
            gross_market_value += abs(position.quantity) * reference_price
        return gross_market_value

    def total_unrealized_pnl(self) -> float:
        unrealized = 0.0
        for instrument_id, position in self.positions.items():
            if abs(position.quantity) < 1e-12:
                continue
            reference_price = self.get_reference_price(instrument_id)
            if reference_price is None:
                continue
            if position.quantity > 0:
                unrealized += position.quantity * (reference_price - position.avg_price)
            else:
                unrealized += abs(position.quantity) * (position.avg_price - reference_price)
        return unrealized

    def total_equity(self) -> float:
        return self.cash + self.total_market_value()

    def net_liquidation_value(self) -> float:
        return self.total_equity()

    def _on_bar(self, event: BarEvent) -> None:
        self.last_price_by_instrument[event.instrument_id] = event.close_price

    def _on_quote_tick(self, event: QuoteTick) -> None:
        self.last_price_by_instrument[event.instrument_id] = (event.bid + event.ask) / 2.0

    def _on_trade_tick(self, event: TradeTick) -> None:
        self.last_price_by_instrument[event.instrument_id] = event.last_price

    def _on_execution(self, event: ExecutionEvent) -> None:
        if event.execution_status not in {"partial_fill", "filled"}:
            return

        position = self.get_position(event.instrument_id)
        fill_qty = float(event.fill_qty)
        fill_price = float(event.fill_price)
        fees = float(event.fees)
        signed_qty = fill_qty if event.side == "buy" else -fill_qty

        if event.side == "buy":
            self.cash -= (fill_qty * fill_price) + fees
        else:
            self.cash += (fill_qty * fill_price) - fees

        self._apply_fill(position, signed_qty, fill_price, fees)
        self.last_price_by_instrument[event.instrument_id] = fill_price

    @staticmethod
    def _apply_fill(position: Position, signed_qty: float, fill_price: float, fees: float) -> None:
        if signed_qty == 0:
            position.realized_pnl -= fees
            return

        current_qty = position.quantity
        if current_qty == 0:
            position.quantity = signed_qty
            position.avg_price = fill_price
            position.realized_pnl -= fees
            return

        same_direction = (current_qty > 0 and signed_qty > 0) or (current_qty < 0 and signed_qty < 0)
        if same_direction:
            total_qty = abs(current_qty) + abs(signed_qty)
            position.avg_price = ((abs(current_qty) * position.avg_price) + (abs(signed_qty) * fill_price)) / total_qty
            position.quantity = current_qty + signed_qty
            position.realized_pnl -= fees
            return

        closing_qty = min(abs(current_qty), abs(signed_qty))
        if current_qty > 0:
            realized = closing_qty * (fill_price - position.avg_price)
        else:
            realized = closing_qty * (position.avg_price - fill_price)
        position.realized_pnl += realized - fees

        remaining_qty = current_qty + signed_qty
        if remaining_qty == 0:
            position.quantity = 0.0
            position.avg_price = 0.0
            return

        if (current_qty > 0 and remaining_qty > 0) or (current_qty < 0 and remaining_qty < 0):
            position.quantity = remaining_qty
            return

        position.quantity = remaining_qty
        position.avg_price = fill_price
