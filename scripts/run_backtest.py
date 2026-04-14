from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.core.bus import InMemoryEventBus
from quant_system.core.clock import SimulatedClock
from quant_system.analytics.performance import PerformanceAnalyzer
from quant_system.data.replay.engine import ReplayEngine
from quant_system.data.replay.source import HistoricalReplaySource
from quant_system.events import ExecutionEvent
from quant_system.execution.brokers.paper import PaperBroker
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.manager import RiskManager
from quant_system.strategies.sma_crossover import SMACrossoverStrategy


def build_dummy_ohlcv(rows: int = 100) -> pd.DataFrame:
    start = datetime(2026, 1, 1, 14, 30, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=i) for i in range(rows)]

    close_prices: list[float] = []
    for i in range(rows):
        trend = 100.0 + (0.08 * i)
        oscillation = 4.0 * math.sin(i / 6.0)
        close_prices.append(trend + oscillation)

    open_prices = [close_prices[0]] + close_prices[:-1]
    high_prices = [max(open_px, close_px) + 0.35 for open_px, close_px in zip(open_prices, close_prices)]
    low_prices = [min(open_px, close_px) - 0.35 for open_px, close_px in zip(open_prices, close_prices)]
    volumes = [100_000 + ((i % 10) * 5_000) for i in range(rows)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        }
    )


def main() -> None:
    instrument_id = "TEST-EQ"
    event_bus = InMemoryEventBus()
    market_data = build_dummy_ohlcv(100)
    datasets = {instrument_id: market_data}

    clock = SimulatedClock(current_time=market_data["timestamp"].iloc[0].to_pydatetime())
    source = HistoricalReplaySource(datasets)
    ledger = PortfolioLedger(event_bus, starting_cash=10_000.0)
    risk_manager = RiskManager(ledger, event_bus)
    paper_broker = PaperBroker(event_bus, simulated_latency_ms=0.0, synchronous=True)
    strategy = SMACrossoverStrategy(
        event_bus,
        instrument_id=instrument_id,
        short_window=8,
        long_window=21,
        long_notional=1_000.0,
    )
    analyzer = PerformanceAnalyzer(ledger, event_bus)

    executed_trades = {"count": 0}

    def on_execution(event: ExecutionEvent) -> None:
        if event.execution_status in {"partial_fill", "filled"}:
            executed_trades["count"] += 1

    event_bus.subscribe("execution", on_execution)

    engine = ReplayEngine(clock, source, event_bus)
    processed_events = engine.run()

    realized_pnl = ledger.total_realized_pnl()
    analyzer.generate_tearsheet()

    print()
    print("=" * 68)
    print("INSTITUTIONAL BACKTEST SUMMARY")
    print("=" * 68)
    print(f"Instrument:           {instrument_id}")
    print(f"Bars Processed:       {processed_events}")
    print(f"Final Sim Time:       {clock.current_time.isoformat()}")
    print(f"Final Cash:           ${ledger.cash:,.2f}")
    print(f"Realized PnL:         ${realized_pnl:,.2f}")
    print(f"Unrealized PnL:       ${ledger.total_unrealized_pnl():,.2f}")
    print(f"Total Equity:         ${ledger.total_equity():,.2f}")
    print(f"Executed Trades:      {executed_trades['count']}")
    print("=" * 68)
    print("Open Positions")
    print("-" * 68)
    for symbol, position in ledger.positions.items():
        if abs(position.quantity) < 1e-12:
            continue
        print(
            f"{symbol:<20} qty={position.quantity:>10.4f}  "
            f"avg_price=${position.avg_price:>10.4f}  "
            f"realized=${position.realized_pnl:>10.2f}"
        )
    print("=" * 68)

    strategy.close()
    analyzer.close()
    paper_broker.close()
    risk_manager.close()
    ledger.close()


if __name__ == "__main__":
    main()
