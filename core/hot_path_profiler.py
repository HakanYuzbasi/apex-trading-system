import json
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Deque, Dict


@dataclass
class _HotPathStats:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    recent_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=256))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[rank]


class HotPathProfiler:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        enabled: bool,
        slow_threshold_ms: float = 250.0,
        summary_interval_cycles: int = 20,
    ) -> None:
        self.logger = logger
        self.enabled = bool(enabled)
        self.slow_threshold_ms = float(slow_threshold_ms)
        self.summary_interval_cycles = max(0, int(summary_interval_cycles))
        self._stats: Dict[str, _HotPathStats] = {}

    @contextmanager
    def track(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record_duration(name, elapsed_ms)

    def record_duration(self, name: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        stats = self._stats.setdefault(name, _HotPathStats())
        stats.count += 1
        stats.total_ms += float(elapsed_ms)
        stats.max_ms = max(stats.max_ms, float(elapsed_ms))
        stats.recent_ms.append(float(elapsed_ms))
        if elapsed_ms >= self.slow_threshold_ms:
            self.logger.info(
                "HOT_PATH_SLOW %s",
                json.dumps(
                    {
                        "path": name,
                        "elapsed_ms": round(float(elapsed_ms), 2),
                        "threshold_ms": round(self.slow_threshold_ms, 2),
                    },
                    sort_keys=True,
                ),
            )

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for name, stats in self._stats.items():
            recent = list(stats.recent_ms)
            summary[name] = {
                "count": float(stats.count),
                "avg_ms": round(stats.total_ms / max(stats.count, 1), 2),
                "max_ms": round(stats.max_ms, 2),
                "p95_ms": round(_percentile(recent, 0.95), 2),
            }
        return summary

    def maybe_log_summary(self, cycle: int) -> None:
        if (
            not self.enabled
            or self.summary_interval_cycles <= 0
            or cycle <= 0
            or cycle % self.summary_interval_cycles != 0
        ):
            return
        summary = self.snapshot()
        if not summary:
            return
        self.logger.info(
            "HOT_PATH_PROFILE %s",
            json.dumps(
                {
                    "cycle": int(cycle),
                    "paths": summary,
                },
                sort_keys=True,
            ),
        )


def profile_hot_path(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            profiler = getattr(self, "_hot_path_profiler", None)
            if profiler is None:
                return await func(self, *args, **kwargs)
            with profiler.track(name):
                return await func(self, *args, **kwargs)

        return async_wrapper

    return decorator
