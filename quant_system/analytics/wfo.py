from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Type

import pandas as pd

from quant_system.analytics.optimizer import GridSearchOptimizer, OptimizationConfig
from quant_system.analytics.performance import PerformanceAnalyzer
from quant_system.data.stores.client import DatabaseConfig
from quant_system.strategies.base import BaseStrategy


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    stitched_oos_equity_curve: pd.Series
    oos_metrics: dict[str, float]
    window_results: pd.DataFrame


class WalkForwardOptimizer:
    def __init__(
        self,
        optimizer_cls: Type[GridSearchOptimizer],
        strategy_cls: Type[BaseStrategy],
        parameter_grid: dict[str, list[Any]],
        *,
        db_config: DatabaseConfig | None = None,
        base_config: OptimizationConfig,
        is_window_days: int,
        oos_window_days: int,
    ) -> None:
        if is_window_days <= 0 or oos_window_days <= 0:
            raise ValueError("window sizes must be positive")
        self.optimizer_cls = optimizer_cls
        self.strategy_cls = strategy_cls
        self.parameter_grid = parameter_grid
        self.db_config = db_config or DatabaseConfig.from_env()
        self.base_config = base_config
        self.is_window_days = is_window_days
        self.oos_window_days = oos_window_days

    def run(self) -> WalkForwardResult:
        window_rows: list[dict[str, Any]] = []
        stitched_curve: pd.Series | None = None
        current_is_start = self.base_config.start_ts
        timeline_end = self.base_config.end_ts
        parameter_keys = list(self.parameter_grid.keys())

        while True:
            current_is_end = current_is_start + timedelta(days=self.is_window_days)
            current_oos_start = current_is_end
            current_oos_end = min(current_oos_start + timedelta(days=self.oos_window_days), timeline_end)

            if current_oos_start >= timeline_end:
                break

            is_config = self._with_window(current_is_start, current_is_end)
            is_optimizer = self.optimizer_cls(
                self.strategy_cls,
                self.parameter_grid,
                db_config=self.db_config,
                optimization_config=is_config,
            )
            is_results = is_optimizer.run_grid(verbose=False)
            if is_results.empty:
                break

            best_row = is_results.iloc[0]
            best_params = {key: best_row[key] for key in parameter_keys}

            oos_config = self._with_window(current_oos_start, current_oos_end)
            oos_optimizer = self.optimizer_cls(
                self.strategy_cls,
                self.parameter_grid,
                db_config=self.db_config,
                optimization_config=oos_config,
            )
            oos_metrics, oos_curve = oos_optimizer.run_backtest(
                best_params,
                optimization_config=oos_config,
                capture_equity_curve=True,
            )

            if oos_curve is not None and not oos_curve.empty:
                stitched_curve = self._stitch_curve(stitched_curve, oos_curve)

            window_rows.append(
                {
                    "is_start": current_is_start,
                    "is_end": current_is_end,
                    "oos_start": current_oos_start,
                    "oos_end": current_oos_end,
                    **{f"best_{key}": value for key, value in best_params.items()},
                    "is_annualized_sharpe": float(best_row["annualized_sharpe"]),
                    "is_max_drawdown_pct": float(best_row["max_drawdown_pct"]),
                    "oos_total_return_pct": float(oos_metrics["total_return_pct"]),
                    "oos_annualized_sharpe": float(oos_metrics["annualized_sharpe"]),
                    "oos_max_drawdown_pct": float(oos_metrics["max_drawdown_pct"]),
                    "oos_annualized_sortino": float(oos_metrics["annualized_sortino"]),
                }
            )

            current_is_start = current_is_start + timedelta(days=self.oos_window_days)

        stitched_curve = stitched_curve if stitched_curve is not None else pd.Series(dtype=float, name="equity")
        oos_metrics = PerformanceAnalyzer.compute_metrics_from_equity_curve(stitched_curve)
        PerformanceAnalyzer.print_metrics(oos_metrics)

        window_results = pd.DataFrame(window_rows)
        return WalkForwardResult(
            stitched_oos_equity_curve=stitched_curve,
            oos_metrics=oos_metrics,
            window_results=window_results,
        )

    def _with_window(self, start_ts, end_ts) -> OptimizationConfig:
        return OptimizationConfig(
            instrument_ids=self.base_config.instrument_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            include_bars=self.base_config.include_bars,
            include_trade_ticks=self.base_config.include_trade_ticks,
            chunk_size=self.base_config.chunk_size,
            starting_cash=self.base_config.starting_cash,
            simulated_latency_ms=self.base_config.simulated_latency_ms,
            synchronous_broker=self.base_config.synchronous_broker,
        )

    @staticmethod
    def _stitch_curve(existing: pd.Series | None, segment: pd.Series) -> pd.Series:
        segment = segment.sort_index()
        if existing is None or existing.empty:
            return segment.copy()

        base_existing = float(existing.iloc[-1])
        base_segment = float(segment.iloc[0])
        if base_segment == 0.0:
            scaled = segment.copy()
        else:
            scaled = base_existing * (segment / base_segment)

        if existing.index[-1] == scaled.index[0]:
            scaled = scaled.iloc[1:]
        return pd.concat([existing, scaled]).sort_index()
