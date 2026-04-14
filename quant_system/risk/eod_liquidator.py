from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

from core.symbols import AssetClass, parse_symbol
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.core.calendars import SessionManager
from quant_system.events import BarEvent, SignalEvent
from quant_system.portfolio.ledger import PortfolioLedger


class EODLiquidator:
    """
    Flatten open equity/option risk near the regular US market close.

    Crypto is explicitly ignored and can continue to run overnight.
    """

    def __init__(
        self,
        portfolio_ledger: PortfolioLedger,
        event_bus: InMemoryEventBus,
        *,
        session_manager: SessionManager | None = None,
        flatten_threshold_minutes: float = 5.0,
    ) -> None:
        self._portfolio_ledger = portfolio_ledger
        self._event_bus = event_bus
        self._session_manager = session_manager or SessionManager()
        self._flatten_threshold_minutes = max(0.0, float(flatten_threshold_minutes))
        self._emitted_state: dict[tuple[date, str], tuple[int, float]] = {}
        self._subscription: Subscription = self._event_bus.subscribe("bar", self._on_bar)

    @property
    def subscription(self) -> Subscription:
        return self._subscription

    def close(self) -> None:
        self._event_bus.unsubscribe(self._subscription.token)

    def _on_bar(self, event: BarEvent) -> None:
        trigger_asset_class = self._asset_class(event.instrument_id)
        if trigger_asset_class not in {AssetClass.EQUITY, AssetClass.OPTION}:
            return

        minutes_until_close = self._session_manager.minutes_until_close(event.instrument_id, event.exchange_ts)
        if minutes_until_close is None or minutes_until_close > self._flatten_threshold_minutes:
            return

        trading_date = event.exchange_ts.date()
        open_positions = [
            (instrument_id, position.quantity)
            for instrument_id, position in self._portfolio_ledger.positions.items()
            if abs(position.quantity) > 1e-12
            and self._asset_class(instrument_id) in {AssetClass.EQUITY, AssetClass.OPTION}
        ]
        if not open_positions:
            self._clear_zero_states(trading_date)
            return

        for instrument_id, quantity in open_positions:
            state_key = (trading_date, instrument_id)
            position_signature = self._position_signature(quantity)
            if self._emitted_state.get(state_key) == position_signature:
                continue

            now = datetime.now(timezone.utc)
            received_ts = max(event.exchange_ts, now)
            processed_ts = max(received_ts, event.processed_ts, now)

            signal = SignalEvent(
                instrument_id=instrument_id,
                exchange_ts=event.exchange_ts,
                received_ts=received_ts,
                processed_ts=processed_ts,
                sequence_id=event.sequence_id,
                source="risk.eod_liquidator",
                strategy_id="EODLiquidator",
                side="flatten",
                target_type="notional",
                target_value=0.0,
                confidence=1.0,
                stop_model="eod_flatten",
                stop_params={"minutes_until_close": minutes_until_close},
                metadata={"trigger_bar_id": event.bar_id},
            )
            self._dispatch_signal(signal)
            self._emitted_state[state_key] = position_signature

        self._clear_zero_states(trading_date)

    def _clear_zero_states(self, trading_date: date) -> None:
        stale_keys = [
            (trade_date, instrument_id)
            for (trade_date, instrument_id) in self._emitted_state
            if trade_date == trading_date
            and abs(self._portfolio_ledger.get_position(instrument_id).quantity) <= 1e-12
        ]
        for key in stale_keys:
            self._emitted_state.pop(key, None)

    def _dispatch_signal(self, event: SignalEvent) -> None:
        subscriptions = self._event_bus.subscriptions_for(event.event_type)
        has_async_subscribers = any(subscription.is_async for subscription in subscriptions)
        if not has_async_subscribers:
            self._event_bus.publish(event)
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._event_bus.publish_async(event))
            return

        loop.create_task(self._event_bus.publish_async(event))

    @staticmethod
    def _position_signature(quantity: float) -> tuple[int, float]:
        side = 1 if quantity > 0 else -1
        return side, round(abs(quantity), 8)

    @staticmethod
    def _asset_class(instrument_id: str) -> AssetClass:
        return parse_symbol(instrument_id).asset_class
