from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BarEvent
from quant_system.portfolio.ledger import PortfolioLedger


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    timestamp: datetime
    equity: float


class PerformanceAnalyzer:
    def __init__(self, portfolio_ledger: PortfolioLedger, event_bus: InMemoryEventBus) -> None:
        self.portfolio_ledger = portfolio_ledger
        self.event_bus = event_bus
        self._snapshots: list[PerformanceSnapshot] = []
        self._subscription: Subscription = self.event_bus.subscribe("bar", self._on_bar)

    @property
    def snapshots(self) -> tuple[PerformanceSnapshot, ...]:
        return tuple(self._snapshots)

    def close(self) -> None:
        self.event_bus.unsubscribe(self._subscription.token)

    def _on_bar(self, event: BarEvent) -> None:
        equity = self.portfolio_ledger.total_equity()
        self._snapshots.append(PerformanceSnapshot(timestamp=event.exchange_ts, equity=equity))

    def equity_curve(self) -> pd.Series:
        if not self._snapshots:
            return pd.Series(dtype=float, name="equity")
        index = pd.DatetimeIndex([snapshot.timestamp for snapshot in self._snapshots], tz="UTC")
        values = [snapshot.equity for snapshot in self._snapshots]
        return pd.Series(values, index=index, name="equity", dtype=float)

    def generate_tearsheet(self) -> dict[str, float]:
        equity_curve = self.equity_curve()
        metrics = self.compute_metrics_from_equity_curve(equity_curve)
        self.print_metrics(metrics)
        return metrics

    def compute_metrics(self) -> dict[str, float]:
        return self.compute_metrics_from_equity_curve(self.equity_curve())

    @classmethod
    def compute_metrics_from_equity_curve(cls, equity_curve: pd.Series) -> dict[str, float]:
        return cls._compute_metrics(equity_curve)

    @classmethod
    def print_metrics(cls, metrics: dict[str, float]) -> None:
        cls._print_tearsheet(metrics)

    @staticmethod
    def _compute_metrics(equity_curve: pd.Series) -> dict[str, float]:
        if equity_curve.empty:
            return {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "annualized_sharpe": 0.0,
                "annualized_sortino": 0.0,
                "omega_ratio": 0.0,
                "ulcer_index": 0.0,
            }

        starting_equity = float(equity_curve.iloc[0])
        ending_equity = float(equity_curve.iloc[-1])
        total_return = (ending_equity / starting_equity) - 1.0 if starting_equity > 0 else 0.0

        returns = equity_curve.pct_change().dropna()
        periods_per_year = PerformanceAnalyzer._infer_periods_per_year(equity_curve.index)
        annualized_return = PerformanceAnalyzer._annualized_return(total_return, len(returns), periods_per_year)
        max_drawdown = PerformanceAnalyzer._max_drawdown(equity_curve)
        annualized_sharpe = PerformanceAnalyzer._annualized_sharpe(returns, periods_per_year)
        annualized_sortino = PerformanceAnalyzer._annualized_sortino(returns, periods_per_year)
        omega_ratio = PerformanceAnalyzer._omega_ratio(returns, threshold=0.0)
        ulcer_index = PerformanceAnalyzer._ulcer_index(equity_curve)

        return {
            "total_return_pct": total_return * 100.0,
            "annualized_return_pct": annualized_return * 100.0,
            "max_drawdown_pct": max_drawdown * 100.0,
            "annualized_sharpe": annualized_sharpe,
            "annualized_sortino": annualized_sortino,
            "omega_ratio": omega_ratio,
            "ulcer_index": ulcer_index,
        }

    @staticmethod
    def _annualized_return(total_return: float, periods: int, periods_per_year: float) -> float:
        if periods <= 0 or periods_per_year <= 0:
            return 0.0
        if total_return <= -1.0:
            return -1.0
        years = periods / periods_per_year
        if years <= 0:
            return 0.0
        exponent = 1.0 / years
        # Guard against overflow: exp(exponent * log(1+r)) must stay within float range.
        # log(float max) ≈ 709, so clamp exponent × |log(1+r)| ≤ 700.
        try:
            result = (1.0 + total_return) ** exponent - 1.0
        except (OverflowError, ZeroDivisionError):
            result = float("inf") if total_return > 0 else float("-inf")
        if not math.isfinite(result):
            return result
        return result

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        running_peak = equity_curve.cummax()
        drawdowns = (equity_curve / running_peak) - 1.0
        return float(drawdowns.min()) if not drawdowns.empty else 0.0

    @staticmethod
    def _annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
        if returns.empty or periods_per_year <= 0:
            return 0.0
        volatility = float(returns.std(ddof=0))
        if math.isclose(volatility, 0.0, abs_tol=1e-12):
            return 0.0
        return float((returns.mean() / volatility) * math.sqrt(periods_per_year))

    @staticmethod
    def _annualized_sortino(returns: pd.Series, periods_per_year: float) -> float:
        if returns.empty or periods_per_year <= 0:
            return 0.0
        downside = returns[returns < 0.0]
        downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if not downside.empty else 0.0
        if math.isclose(downside_deviation, 0.0, abs_tol=1e-12):
            return 0.0
        return float((returns.mean() / downside_deviation) * math.sqrt(periods_per_year))

    @staticmethod
    def _omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculates the Omega Ratio which is the weighted ratio of gains to losses
        given a threshold (MAR). Unlike Sharpe, it captures all moments of the distribution.
        """
        if returns.empty:
            return 0.0
        diff = returns - threshold
        gains = diff[diff > 0].sum()
        losses = -diff[diff < 0].sum()
        if math.isclose(losses, 0.0, abs_tol=1e-12):
            return 10.0 if gains > 0 else 0.0
        return float(gains / losses)

    @staticmethod
    def _ulcer_index(equity_curve: pd.Series) -> float:
        """
        Calculates the Ulcer Index which measures the depth and duration of drawdowns.
        UI = sqrt(mean(drawdown_squared))
        """
        if equity_curve.empty:
            return 0.0
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve / rolling_max) - 1.0
        # Ulcer Index uses drawdown percentages (not negative)
        squared_drawdowns = np.square(drawdowns * 100.0)
        return float(np.sqrt(np.mean(squared_drawdowns)))

    @staticmethod
    def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
        if len(index) < 2:
            return 252.0
        deltas = index.to_series().diff().dropna()
        if deltas.empty:
            return 252.0

        median_delta = deltas.median()
        delta_seconds = median_delta.total_seconds()
        if delta_seconds <= 0:
            return 252.0

        delta_days = delta_seconds / 86_400.0
        if delta_days >= 1.0:
            return 252.0 / delta_days

        trading_minutes_per_year = 252.0 * 390.0
        delta_minutes = delta_seconds / 60.0
        if delta_minutes <= 0:
            return 252.0
        return trading_minutes_per_year / delta_minutes

    @staticmethod
    def _print_tearsheet(metrics: dict[str, float]) -> None:
        print()
        print("=" * 68)
        print("PERFORMANCE TEARSHEET")
        print("=" * 68)
        print(f"{'Total Return':<28}{metrics['total_return_pct']:>12.2f}%")
        print(f"{'Annualized Return':<28}{metrics['annualized_return_pct']:>12.2f}%")
        print(f"{'Maximum Drawdown':<28}{metrics['max_drawdown_pct']:>12.2f}%")
        print(f"{'Annualized Sharpe':<28}{metrics['annualized_sharpe']:>12.3f}")
        print(f"{'Annualized Sortino':<28}{metrics['annualized_sortino']:>12.3f}")
        print(f"{'Omega Ratio':<28}{metrics['omega_ratio']:>12.3f}")
        print(f"{'Ulcer Index':<28}{metrics['ulcer_index']:>12.3f}")
        print("=" * 68)
