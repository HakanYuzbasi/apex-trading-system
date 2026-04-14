from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quant_system.analytics.performance import PerformanceAnalyzer
from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, ExecutionEvent
from quant_system.portfolio.ledger import PortfolioLedger


def test_portfolio_ledger_mark_to_market_helpers() -> None:
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
            slippage=0.0,
            remaining_qty=0.0,
        )
    )
    bus.publish(
        BarEvent(
            instrument_id="AAPL",
            exchange_ts=now + timedelta(minutes=1),
            received_ts=now + timedelta(minutes=1),
            processed_ts=now + timedelta(minutes=1),
            sequence_id=2,
            source="test",
            open_price=101.0,
            high_price=101.0,
            low_price=101.0,
            close_price=101.0,
            volume=1_000.0,
        )
    )

    assert ledger.total_market_value() == pytest.approx(1_010.0)
    assert ledger.total_unrealized_pnl() == pytest.approx(10.0)
    assert ledger.total_equity() == pytest.approx(10_009.0)


def test_performance_analyzer_generates_expected_metrics() -> None:
    bus = InMemoryEventBus()
    ledger = PortfolioLedger(bus, starting_cash=10_000.0)
    analyzer = PerformanceAnalyzer(ledger, bus)
    base_ts = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)

    equity_targets = [10_000.0, 10_100.0, 9_900.0, 10_200.0]
    for idx, equity in enumerate(equity_targets):
        ledger.cash = equity
        bus.publish(
            BarEvent(
                instrument_id="TEST-EQ",
                exchange_ts=base_ts + timedelta(days=idx),
                received_ts=base_ts + timedelta(days=idx),
                processed_ts=base_ts + timedelta(days=idx),
                sequence_id=idx,
                source="test",
                open_price=100.0,
                high_price=100.0,
                low_price=100.0,
                close_price=100.0,
                volume=1_000.0,
            )
        )

    metrics = analyzer.generate_tearsheet()

    assert metrics["total_return_pct"] == pytest.approx(2.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(-1.9801980198)
    assert metrics["annualized_sharpe"] != 0.0
    assert metrics["annualized_sortino"] != 0.0
