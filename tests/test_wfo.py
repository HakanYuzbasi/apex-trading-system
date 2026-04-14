from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_system.analytics.optimizer import OptimizationConfig
from quant_system.analytics.wfo import WalkForwardOptimizer
from quant_system.strategies.base import BaseStrategy


class FakeStrategy(BaseStrategy):
    def on_bar(self, event) -> None:
        return

    def on_tick(self, event) -> None:
        return


class FakeGridSearchOptimizer:
    instances = []

    def __init__(self, strategy_cls, parameter_grid, *, db_config=None, optimization_config=None) -> None:
        self.optimization_config = optimization_config
        self.parameter_grid = parameter_grid
        FakeGridSearchOptimizer.instances.append(self)

    def run_grid(self, *, verbose: bool = True) -> pd.DataFrame:
        short = 10 if self.optimization_config.start_ts.day % 2 == 0 else 20
        return pd.DataFrame(
            [
                {
                    "instrument_id": "AAPL",
                    "short_window": short,
                    "long_window": 50,
                    "long_notional": 5_000.0,
                    "annualized_sharpe": 2.0,
                    "max_drawdown_pct": -5.0,
                }
            ]
        )

    def run_backtest(self, strategy_params, *, optimization_config=None, capture_equity_curve=False):
        idx = pd.date_range(
            optimization_config.start_ts,
            periods=3,
            freq="D",
            tz="UTC",
        )
        if strategy_params["short_window"] == 10:
            curve = pd.Series([50_000.0, 50_500.0, 51_000.0], index=idx, name="equity")
        else:
            curve = pd.Series([50_000.0, 49_500.0, 49_750.0], index=idx, name="equity")
        metrics = {
            "total_return_pct": float((curve.iloc[-1] / curve.iloc[0] - 1.0) * 100.0),
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": -1.0,
            "annualized_sharpe": 1.0,
            "annualized_sortino": 1.0,
        }
        return metrics, curve


def test_walk_forward_optimizer_rolls_windows_and_stitches_oos_curve() -> None:
    start_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end_ts = start_ts + timedelta(days=100)
    wfo = WalkForwardOptimizer(
        FakeGridSearchOptimizer,
        FakeStrategy,
        {
            "instrument_id": ["AAPL"],
            "short_window": [10, 20],
            "long_window": [50],
            "long_notional": [5_000.0],
        },
        base_config=OptimizationConfig(
            instrument_ids=("AAPL",),
            start_ts=start_ts,
            end_ts=end_ts,
        ),
        is_window_days=60,
        oos_window_days=14,
    )

    result = wfo.run()

    assert not result.window_results.empty
    assert list(result.window_results.columns[:4]) == ["is_start", "is_end", "oos_start", "oos_end"]
    assert "best_short_window" in result.window_results.columns
    assert len(result.stitched_oos_equity_curve) >= 3
    assert result.oos_metrics["max_drawdown_pct"] <= 0.0
