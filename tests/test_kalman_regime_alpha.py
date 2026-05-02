from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_system.analytics.alpha_monitor import AlphaDecayMonitor
from quant_system.analytics.notifier import TelegramNotifier
from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, SignalEvent
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.manager import RiskManager
from quant_system.risk.regime_detector import RegimeDetector
from quant_system.strategies.kalman_pairs import KalmanPairsStrategy


def _bar(instrument_id: str, when: datetime, close_price: float) -> BarEvent:
    return BarEvent(
        instrument_id=instrument_id,
        exchange_ts=when,
        received_ts=when,
        processed_ts=when,
        sequence_id=1,
        source="test",
        open_price=close_price,
        high_price=close_price,
        low_price=close_price,
        close_price=close_price,
        volume=100.0,
    )


def test_kalman_pairs_strategy_emits_entry_signals_after_warmup() -> None:
    bus = InMemoryEventBus()
    strategy = KalmanPairsStrategy(
        bus,
        instrument_a="AAPL",
        instrument_b="MSFT",
        entry_z_score=1.0,
        exit_z_score=0.5,
        warmup_bars=5,
        leg_notional=1_000.0,
    )
    signals: list[SignalEvent] = []
    bus.subscribe("signal", lambda event: signals.append(event))

    start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    # 5 warmup bars with ~3-unit variation so OLS is non-degenerate and
    # the post-warm-start z-gate doesn't fire on a pathological theta.
    # Followed by a large gap bar that drives entry signals for both legs.
    paired_prices = [
        (100.0, 98.0),
        (102.0, 100.0),
        (98.0, 96.0),
        (101.0, 99.0),
        (99.0, 97.0),
        (115.0, 97.0),   # A spikes well above the spread → entry
    ]

    for index, (price_a, price_b) in enumerate(paired_prices):
        ts = start + timedelta(hours=index)
        bus.publish(_bar("AAPL", ts, price_a))
        bus.publish(_bar("MSFT", ts, price_b))

    assert len(signals) >= 2
    instruments = {s.instrument_id for s in signals}
    assert "AAPL" in instruments
    assert "MSFT" in instruments
    strategy.close()


def test_risk_manager_vetoes_entries_during_extreme_volatility() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=10_000.0)
    detector = RegimeDetector(bus, high_vol_threshold=0.01, breakout_return_threshold=0.03, breakout_window=2)
    risk_manager = RiskManager(ledger, bus, regime_detector=detector)
    orders = []

    bus.subscribe("order", lambda event: orders.append(event))

    start = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    bus.publish(_bar("AAPL", start, 100.0))
    bus.publish(_bar("AAPL", start + timedelta(hours=1), 110.0))
    bus.publish(
        SignalEvent(
            instrument_id="AAPL",
            exchange_ts=start + timedelta(hours=1),
            received_ts=start + timedelta(hours=1),
            processed_ts=start + timedelta(hours=1),
            sequence_id=2,
            source="strategy.test",
            strategy_id="Demo",
            side="buy",
            target_type="notional",
            target_value=1_000.0,
            confidence=0.8,
            stop_model="none",
            stop_params={},
        )
    )

    assert detector.is_extreme_volatility("AAPL") is True
    assert orders == []

    risk_manager.close()
    detector.close()
    ledger.close()


@pytest.mark.asyncio
async def test_alpha_decay_monitor_sends_critical_alert() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=10_000.0)
    notifier = TelegramNotifier(bus)
    messages: list[str] = []

    async def capture_message(text: str) -> None:
        messages.append(text)

    notifier.notify_text = capture_message  # type: ignore[method-assign]
    monitor = AlphaDecayMonitor(
        ledger,
        bus,
        notifier,
        expected_oos_sharpe=2.0,
        decay_threshold=0.50,
        evaluation_window_days=30,
        alert_cooldown_hours=1,
    )

    start = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    prices = [100.0, 95.0, 90.0, 85.0]
    position = ledger.get_position("AAPL")
    position.quantity = 10.0
    position.avg_price = 100.0

    for index, price in enumerate(prices):
        ts = start + timedelta(days=10 * index)
        await bus.publish_async(_bar("AAPL", ts, price))

    assert messages == ["🚨 ALPHA DECAY DETECTED: Strategy performance deviating from historical norm."]

    await monitor.close()
    await notifier.close()
    ledger.close()
