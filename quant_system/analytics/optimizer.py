from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Type

import pandas as pd

from quant_system.analytics.performance import PerformanceAnalyzer
from quant_system.core.bus import InMemoryEventBus
from quant_system.core.clock import SimulatedClock
from quant_system.data.replay.engine import ReplayEngine
from quant_system.data.replay.source import HistoricalReplaySource
from quant_system.data.stores.client import DatabaseConfig, TimescaleDBClient
from quant_system.execution.brokers.paper import PaperBroker
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.manager import RiskManager
from quant_system.strategies.base import BaseStrategy


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    instrument_ids: tuple[str, ...]
    start_ts: datetime
    end_ts: datetime
    include_bars: bool = True
    include_trade_ticks: bool = False
    chunk_size: int = 10_000
    starting_cash: float = 50_000.0
    simulated_latency_ms: float = 0.0
    synchronous_broker: bool = True


class GridSearchOptimizer:
    def __init__(
        self,
        strategy_cls: Type[BaseStrategy],
        parameter_grid: dict[str, list[Any]],
        *,
        db_config: DatabaseConfig | None = None,
        optimization_config: OptimizationConfig,
    ) -> None:
        self.strategy_cls = strategy_cls
        self.parameter_grid = parameter_grid
        self.db_config = db_config or DatabaseConfig.from_env()
        self.optimization_config = optimization_config

    def run(self) -> pd.DataFrame:
        return self.run_grid(verbose=True)

    def run_grid(self, *, verbose: bool = True) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for params in self._parameter_combinations():
            metrics, _ = self.run_backtest(params)
            rows.append({**params, **metrics})

        results = pd.DataFrame(rows)
        if results.empty:
            return results

        results = results.sort_values("annualized_sharpe", ascending=False, kind="stable").reset_index(drop=True)
        if verbose:
            print()
            print("=" * 88)
            print("GRID SEARCH TOP 5 BY SHARPE")
            print("=" * 88)
            print(results.head(5).to_string(index=False))
            print("=" * 88)
        return results

    def _parameter_combinations(self) -> list[dict[str, Any]]:
        keys = list(self.parameter_grid.keys())
        values = [self.parameter_grid[key] for key in keys]
        return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]

    def run_backtest(
        self,
        strategy_params: dict[str, Any],
        *,
        optimization_config: OptimizationConfig | None = None,
        capture_equity_curve: bool = False,
    ) -> tuple[dict[str, Any], pd.Series | None]:
        config = optimization_config or self.optimization_config
        db_client = TimescaleDBClient(self.db_config)
        event_bus = InMemoryEventBus()
        clock = SimulatedClock(current_time=config.start_ts)
        source = HistoricalReplaySource(
            client=db_client,
            start_ts=config.start_ts,
            end_ts=config.end_ts,
            instrument_ids=config.instrument_ids,
            include_bars=config.include_bars,
            include_trade_ticks=config.include_trade_ticks,
            chunk_size=config.chunk_size,
        )
        ledger = PortfolioLedger(event_bus, starting_cash=config.starting_cash)
        risk_manager = RiskManager(ledger, event_bus)
        paper_broker = PaperBroker(
            event_bus,
            simulated_latency_ms=config.simulated_latency_ms,
            synchronous=config.synchronous_broker,
        )
        analyzer = PerformanceAnalyzer(ledger, event_bus)

        strategy = self.strategy_cls(event_bus, **strategy_params)
        engine = ReplayEngine(clock, source, event_bus)
        processed_events = engine.run()
        metrics = analyzer.compute_metrics()
        metrics["bars_processed"] = processed_events
        metrics["final_cash"] = ledger.cash
        metrics["final_equity"] = ledger.total_equity()
        metrics["realized_pnl"] = ledger.total_realized_pnl()
        metrics["unrealized_pnl"] = ledger.total_unrealized_pnl()
        equity_curve = analyzer.equity_curve() if capture_equity_curve else None

        strategy.close()
        analyzer.close()
        paper_broker.close()
        risk_manager.close()
        ledger.close()
        return metrics, equity_curve
