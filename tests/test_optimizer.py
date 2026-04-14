from __future__ import annotations

from datetime import datetime, timezone

from quant_system.analytics.optimizer import GridSearchOptimizer, OptimizationConfig
from quant_system.strategies.base import BaseStrategy


class FakeStrategy(BaseStrategy):
    def __init__(self, event_bus, *, instrument_id: str, alpha: int) -> None:
        self.instrument_id = instrument_id
        self.alpha = alpha
        super().__init__(event_bus)

    def on_bar(self, event) -> None:
        return

    def on_tick(self, event) -> None:
        return


class FakeClient:
    def stream_event_rows(self, **kwargs):
        return iter(())


def test_optimizer_runs_isolated_backtests_per_parameter_combination(monkeypatch) -> None:
    created_ledgers = []
    created_buses = []

    class FakeBus:
        def __init__(self) -> None:
            created_buses.append(self)

        def subscribe(self, event_type, handler, *, is_async=None):
            class Subscription:
                token = f"{event_type}-sub"
                is_async = False

            return Subscription()

        def unsubscribe(self, token):
            return True

        def subscriptions_for(self, event_type):
            return []

        def publish(self, event):
            return None

    class FakeLedger:
        def __init__(self, event_bus, starting_cash):
            self.event_bus = event_bus
            self.cash = float(starting_cash)
            created_ledgers.append(self)

        def total_equity(self):
            return self.cash

        def total_realized_pnl(self):
            return 0.0

        def total_unrealized_pnl(self):
            return 0.0

        def close(self):
            return None

    class FakeAnalyzer:
        def __init__(self, ledger, event_bus):
            self.ledger = ledger

        def compute_metrics(self):
            return {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "annualized_sharpe": 1.0,
                "annualized_sortino": 1.0,
            }

        def close(self):
            return None

    class FakeRiskManager:
        def __init__(self, ledger, event_bus):
            self.ledger = ledger

        def close(self):
            return None

    class FakePaperBroker:
        def __init__(self, event_bus, **kwargs):
            self.event_bus = event_bus

        def close(self):
            return None

    class FakeReplayEngine:
        def __init__(self, clock, source, event_bus):
            self.clock = clock

        def run(self):
            return 0

    monkeypatch.setattr("quant_system.analytics.optimizer.InMemoryEventBus", FakeBus)
    monkeypatch.setattr("quant_system.analytics.optimizer.PortfolioLedger", FakeLedger)
    monkeypatch.setattr("quant_system.analytics.optimizer.PerformanceAnalyzer", FakeAnalyzer)
    monkeypatch.setattr("quant_system.analytics.optimizer.RiskManager", FakeRiskManager)
    monkeypatch.setattr("quant_system.analytics.optimizer.PaperBroker", FakePaperBroker)
    monkeypatch.setattr("quant_system.analytics.optimizer.ReplayEngine", FakeReplayEngine)
    monkeypatch.setattr("quant_system.analytics.optimizer.TimescaleDBClient", lambda config: FakeClient())

    optimizer = GridSearchOptimizer(
        FakeStrategy,
        {"instrument_id": ["AAPL"], "alpha": [1, 2, 3]},
        optimization_config=OptimizationConfig(
            instrument_ids=("AAPL",),
            start_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_ts=datetime(2026, 2, 1, tzinfo=timezone.utc),
        ),
    )

    results = optimizer.run()

    assert len(results) == 3
    assert len(created_ledgers) == 3
    assert len(created_buses) == 3
    assert len({id(obj) for obj in created_ledgers}) == 3
    assert len({id(obj) for obj in created_buses}) == 3
