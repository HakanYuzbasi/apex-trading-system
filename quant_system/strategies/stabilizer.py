from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)

# Config defaults
STABILIZER_FAILURE_THRESHOLD: int = 8
STABILIZER_MIN_READONLY_MINUTES: float = 15.0
STABILIZER_FAILURE_RATE_WINDOW: int = 20
STABILIZER_FAILURE_RATE_THRESHOLD: float = 0.75

# Exponential backoff: attempt 1 → 15 min, attempt 2 → 30 min, attempt 3+ → 60 min
_BACKOFF_SCHEDULE_MINUTES = (15.0, 30.0, 60.0)


@dataclass(slots=True)
class CointegrationResult:
    is_cointegrated: bool
    rank: int
    confidence_level: float
    max_eigen_stat: float
    critical_value: float


class JohansenStabilizer:
    """
    Cointegration guard using the Johansen Trace Test.

    Improvements over original:
    - fail_streak raised to 8 (was 3) to avoid false Read-Only on normal crypto volatility.
    - Failure-rate trigger: also enters Read-Only when >75% of last 20 tests fail.
    - Minimum 15-min Read-Only duration before exit is considered.
    - Exponential backoff between re-evaluation attempts: 15 / 30 / 60 min.
    - WARN logged only once per Read-Only period; INFO on re-evaluation and exit.
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
        min_window: int = 60,
        fail_streak: int = STABILIZER_FAILURE_THRESHOLD,
        pass_streak: int = 2,
        failure_rate_window: int = STABILIZER_FAILURE_RATE_WINDOW,
        failure_rate_threshold: float = STABILIZER_FAILURE_RATE_THRESHOLD,
        min_readonly_minutes: float = STABILIZER_MIN_READONLY_MINUTES,
    ) -> None:
        self.confidence_level = confidence_level
        self.min_window = min_window
        self.fail_streak = fail_streak
        self.pass_streak = pass_streak
        self.failure_rate_window = failure_rate_window
        self.failure_rate_threshold = failure_rate_threshold
        self.min_readonly_seconds = min_readonly_minutes * 60.0

        self._level_idx = {0.90: 0, 0.95: 1, 0.99: 2}.get(confidence_level, 0)
        self._consecutive_fails: int = 0
        self._consecutive_passes: int = 0
        self._history: Deque[bool] = deque(maxlen=failure_rate_window)
        self.is_read_only: bool = False
        self._readonly_since: float | None = None
        self._readonly_warn_logged: bool = False
        self._backoff_attempt: int = 0
        self._next_eval_at: float = 0.0

    # ------------------------------------------------------------------

    def test(
        self,
        prices_a: list[float] | np.ndarray,
        prices_b: list[float] | np.ndarray,
    ) -> CointegrationResult:
        """Run the Johansen test and update debounced read-only state."""
        if len(prices_a) != len(prices_b) or len(prices_a) < self.min_window:
            return CointegrationResult(True, 1, self.confidence_level, 0.0, 0.0)

        # Skip expensive test during backoff window
        if self.is_read_only and time.monotonic() < self._next_eval_at:
            return CointegrationResult(False, 0, self.confidence_level, 0.0, 0.0)

        data = pd.DataFrame({"a": prices_a, "b": prices_b})
        trace_stat: float = 0.0
        trace_crit: float = 0.0

        try:
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            trace_stat = float(result.lr1[0])
            trace_crit = float(result.cvt[0, self._level_idx])
            is_rank_1 = trace_stat > trace_crit
        except Exception as e:
            logger.error("JohansenStabilizer: Error running test: %s", e)
            is_rank_1 = True  # don't block on test error

        self._history.append(is_rank_1)
        self._update_state(is_rank_1)

        return CointegrationResult(
            is_cointegrated=not self.is_read_only,
            rank=1 if is_rank_1 else 0,
            confidence_level=self.confidence_level,
            max_eigen_stat=trace_stat,
            critical_value=trace_crit,
        )

    # ------------------------------------------------------------------

    def _failure_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for x in self._history if not x) / len(self._history)

    def _update_state(self, passed: bool) -> None:
        now = time.monotonic()

        if passed:
            self._consecutive_fails = 0
            self._consecutive_passes += 1
            if self.is_read_only:
                elapsed = now - self._readonly_since if self._readonly_since is not None else 0.0
                if elapsed >= self.min_readonly_seconds and self._consecutive_passes >= self.pass_streak:
                    logger.info(
                        "JohansenStabilizer: Cointegration restored after %.0fs — exiting Read-Only mode.",
                        elapsed,
                    )
                    self.is_read_only = False
                    self._readonly_since = None
                    self._readonly_warn_logged = False
                    self._backoff_attempt = 0
                    self._next_eval_at = 0.0
                else:
                    logger.info(
                        "JohansenStabilizer: Re-evaluation pass %d/%d (elapsed %.0fs / min %.0fs).",
                        self._consecutive_passes, self.pass_streak,
                        elapsed, self.min_readonly_seconds,
                    )
        else:
            self._consecutive_passes = 0
            self._consecutive_fails += 1

            if not self.is_read_only:
                high_failure_rate = (
                    len(self._history) >= self.failure_rate_window
                    and self._failure_rate() > self.failure_rate_threshold
                )
                if self._consecutive_fails >= self.fail_streak or high_failure_rate:
                    self.is_read_only = True
                    self._readonly_since = now
                    self._readonly_warn_logged = False
                    self._backoff_attempt = 0

            if self.is_read_only:
                if not self._readonly_warn_logged:
                    logger.warning(
                        "JohansenStabilizer: Entering Read-Only — %d consecutive failures, "
                        "failure_rate=%.0f%% over last %d tests.",
                        self._consecutive_fails,
                        self._failure_rate() * 100,
                        len(self._history),
                    )
                    self._readonly_warn_logged = True

                backoff_idx = min(self._backoff_attempt, len(_BACKOFF_SCHEDULE_MINUTES) - 1)
                backoff_min = _BACKOFF_SCHEDULE_MINUTES[backoff_idx]
                self._next_eval_at = now + backoff_min * 60.0
                self._backoff_attempt += 1
                logger.info(
                    "JohansenStabilizer: Next re-evaluation in %.0f min (attempt %d).",
                    backoff_min, self._backoff_attempt,
                )
