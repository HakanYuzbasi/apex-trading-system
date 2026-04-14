from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.analytics.performance import PerformanceAnalyzer
from quant_system.core.bus import InMemoryEventBus
from quant_system.core.clock import SimulatedClock
from quant_system.data.replay.engine import ReplayEngine
from quant_system.data.replay.source import HistoricalReplaySource
from quant_system.data.stores.client import TimescaleDBClient
from quant_system.events import ExecutionEvent
from quant_system.execution.brokers.paper import PaperBroker
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.manager import RiskManager
from quant_system.strategies.sma_crossover import SMACrossoverStrategy


def main() -> None:
    instrument_ids = ["AAPL"]
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=182)

    db_client = TimescaleDBClient()
    event_bus = InMemoryEventBus()
    clock = SimulatedClock(current_time=start_ts)
    source = HistoricalReplaySource(
        client=db_client,
        start_ts=start_ts,
        end_ts=end_ts,
        instrument_ids=instrument_ids,
        include_bars=True,
        include_trade_ticks=False,
        chunk_size=10_000,
    )

    ledger = PortfolioLedger(event_bus, starting_cash=50_000.0)
    risk_manager = RiskManager(ledger, event_bus)
    paper_broker = PaperBroker(event_bus, simulated_latency_ms=0.0, synchronous=True)
    strategy = SMACrossoverStrategy(
        event_bus,
        instrument_id="AAPL",
        short_window=20,
        long_window=50,
        long_notional=5_000.0,
    )
    analyzer = PerformanceAnalyzer(ledger, event_bus)

    executed_trades = {"count": 0}

    def on_execution(event: ExecutionEvent) -> None:
        if event.execution_status in {"partial_fill", "filled"}:
            executed_trades["count"] += 1

    event_bus.subscribe("execution", on_execution)

    engine = ReplayEngine(clock, source, event_bus)
    processed_events = engine.run()
    analyzer.generate_tearsheet()

    print()
    print("=" * 68)
    print("DATABASE-BACKED BACKTEST SUMMARY")
    print("=" * 68)
    print(f"Instrument:           {', '.join(instrument_ids)}")
    print(f"Start:                {start_ts.isoformat()}")
    print(f"End:                  {end_ts.isoformat()}")
    print(f"Bars Processed:       {processed_events}")
    print(f"Final Sim Time:       {clock.current_time.isoformat()}")
    print(f"Final Cash:           ${ledger.cash:,.2f}")
    print(f"Realized PnL:         ${ledger.total_realized_pnl():,.2f}")
    print(f"Unrealized PnL:       ${ledger.total_unrealized_pnl():,.2f}")
    print(f"Total Equity:         ${ledger.total_equity():,.2f}")
    print(f"Executed Trades:      {executed_trades['count']}")
    print("=" * 68)

    strategy.close()
    analyzer.close()
    paper_broker.close()
    risk_manager.close()
    ledger.close()


if __name__ == "__main__":
    main()
