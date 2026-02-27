"""
risk/kill_switch.py

Portfolio-level hard kill-switch driven by drawdown and Sharpe decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class KillSwitchConfig:
    dd_multiplier: float = 1.5
    sharpe_window_days: int = 63
    sharpe_floor: float = 0.2
    logic: str = "OR"  # OR / AND
    min_points: int = 20


@dataclass
class KillSwitchState:
    active: bool
    triggered_at: Optional[str]
    reason: str
    drawdown: float
    historical_mdd: float
    sharpe_rolling: float
    flatten_executed: bool


class RiskKillSwitch:
    """Latched kill-switch with manual reset."""

    def __init__(self, config: KillSwitchConfig, historical_mdd_baseline: float):
        self.config = config
        self.historical_mdd_baseline = max(0.001, float(historical_mdd_baseline))
        self.active = False
        self.triggered_at: Optional[str] = None
        self.reason = ""
        self.flatten_executed = False
        self.last_drawdown = 0.0
        self.last_sharpe = 0.0

    def update(self, *args, **kwargs):
        # Permanently disabled as requested
        self.active = False
        return self.state()

    def _original_update(self, equity_curve: Iterable[Tuple[object, float]]) -> KillSwitchState:
        entries = self._normalize_equity_curve(equity_curve)
        if len(entries) < self.config.min_points:
            return self.state()

        self.last_drawdown = self._current_drawdown(entries)
        self.last_sharpe = self._rolling_sharpe(entries, self.config.sharpe_window_days)

        dd_limit = self.config.dd_multiplier * self.historical_mdd_baseline
        dd_breach = self.last_drawdown > dd_limit
        sharpe_breach = self.last_sharpe < self.config.sharpe_floor

        if self.config.logic.upper() == "AND":
            breach = dd_breach and sharpe_breach
            reason = (
                f"DD+Sharpe breach: dd={self.last_drawdown:.2%}>{dd_limit:.2%} and "
                f"sharpe={self.last_sharpe:.2f}<{self.config.sharpe_floor:.2f}"
            )
        else:
            breach = dd_breach or sharpe_breach
            reason = (
                f"DD/Sharpe breach: dd={self.last_drawdown:.2%}>{dd_limit:.2%} or "
                f"sharpe={self.last_sharpe:.2f}<{self.config.sharpe_floor:.2f}"
            )

        if breach and not self.active:
            self.active = True
            self.triggered_at = datetime.utcnow().isoformat()
            self.reason = reason

        return self.state()

    def mark_flattened(self) -> None:
        self.flatten_executed = True

    def reset(self) -> None:
        self.active = False
        self.triggered_at = None
        self.reason = ""
        self.flatten_executed = False

    def state(self) -> KillSwitchState:
        return KillSwitchState(
            active=self.active,
            triggered_at=self.triggered_at,
            reason=self.reason,
            drawdown=float(self.last_drawdown),
            historical_mdd=float(self.historical_mdd_baseline),
            sharpe_rolling=float(self.last_sharpe),
            flatten_executed=bool(self.flatten_executed),
        )

    def _normalize_equity_curve(self, equity_curve: Iterable[Tuple[object, float]]) -> List[Tuple[datetime, float]]:
        entries: List[Tuple[datetime, float]] = []
        for ts_raw, value in equity_curve:
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if v <= 0:
                continue

            ts = self._parse_ts(ts_raw)
            if ts is None:
                continue
            entries.append((ts, v))

        entries.sort(key=lambda x: x[0])
        return entries

    @staticmethod
    def _parse_ts(ts_raw: object) -> Optional[datetime]:
        if isinstance(ts_raw, datetime):
            return ts_raw
        if isinstance(ts_raw, str):
            try:
                return datetime.fromisoformat(ts_raw)
            except ValueError:
                return None
        return None

    @staticmethod
    def _current_drawdown(entries: List[Tuple[datetime, float]]) -> float:
        values = np.array([v for _, v in entries], dtype=float)
        if values.size < 2:
            return 0.0
        running_peak = np.maximum.accumulate(values)
        drawdowns = (values - running_peak) / np.maximum(running_peak, 1e-9)
        return abs(float(drawdowns[-1]))

    @staticmethod
    def _infer_points_per_year(entries: List[Tuple[datetime, float]]) -> float:
        if len(entries) < 3:
            return 252.0
        deltas = []
        for i in range(1, len(entries)):
            dt = (entries[i][0] - entries[i - 1][0]).total_seconds()
            if dt > 0:
                deltas.append(dt)
        if not deltas:
            return 252.0
        median_delta = float(np.median(deltas))
        year_seconds = 365.0 * 24.0 * 3600.0
        return max(252.0, year_seconds / max(median_delta, 1.0))

    def _rolling_sharpe(self, entries: List[Tuple[datetime, float]], window_days: int) -> float:
        cutoff = entries[-1][0] - timedelta(days=window_days)
        window = [(ts, v) for ts, v in entries if ts >= cutoff]
        if len(window) < self.config.min_points:
            return 0.0

        values = np.array([v for _, v in window], dtype=float)
        returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
        returns = returns[np.isfinite(returns)]
        if returns.size < max(5, self.config.min_points // 3):
            return 0.0

        points_per_year = self._infer_points_per_year(window)
        vol = float(np.std(returns))
        if vol <= 1e-12:
            return 0.0
        return float(np.mean(returns) / vol * np.sqrt(points_per_year))
