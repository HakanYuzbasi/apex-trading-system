from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)

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

    Changes vs original:
    - Minimum window of 60 bars before the test runs (20 bars is too short
      for the Johansen test to have any statistical power).
    - Default confidence lowered to 0.90 (standard for intraday pairs).
    - Debounce: requires `fail_streak` consecutive failures before locking
      Read-Only, and `pass_streak` consecutive passes before unlocking.
      This prevents a single noisy bar from blocking all entries.
    """

    def __init__(
        self,
        confidence_level: float = 0.90,
        min_window: int = 60,
        fail_streak: int = 3,
        pass_streak: int = 2,
    ) -> None:
        self.confidence_level = confidence_level
        self.min_window = min_window
        self.fail_streak = fail_streak
        self.pass_streak = pass_streak
        self._level_idx = {0.90: 0, 0.95: 1, 0.99: 2}.get(confidence_level, 0)
        self._consecutive_fails: int = 0
        self._consecutive_passes: int = 0
        self.is_read_only: bool = False

    def test(
        self,
        prices_a: list[float] | np.ndarray,
        prices_b: list[float] | np.ndarray,
    ) -> CointegrationResult:
        """Run the Johansen test and update debounced read-only state."""
        if len(prices_a) != len(prices_b) or len(prices_a) < self.min_window:
            # Not enough data yet — don't block trading
            return CointegrationResult(True, 1, self.confidence_level, 0.0, 0.0)

        data = pd.DataFrame({"a": prices_a, "b": prices_b})

        try:
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            trace_stat = result.lr1[0]
            trace_crit = result.cvt[0, self._level_idx]
            is_rank_1 = trace_stat > trace_crit
        except Exception as e:
            logger.error("JohansenStabilizer: Error running test: %s", e)
            # On error, don't block — treat as pass
            is_rank_1 = True

        self._update_state(is_rank_1)

        return CointegrationResult(
            is_cointegrated=not self.is_read_only,
            rank=1 if is_rank_1 else 0,
            confidence_level=self.confidence_level,
            max_eigen_stat=trace_stat if 'trace_stat' in dir() else 0.0,
            critical_value=trace_crit if 'trace_crit' in dir() else 0.0,
        )

    def _update_state(self, passed: bool) -> None:
        if passed:
            self._consecutive_fails = 0
            self._consecutive_passes += 1
            if self.is_read_only and self._consecutive_passes >= self.pass_streak:
                logger.info("JohansenStabilizer: Cointegration restored — exiting Read-Only mode.")
                self.is_read_only = False
        else:
            self._consecutive_passes = 0
            self._consecutive_fails += 1
            if not self.is_read_only and self._consecutive_fails >= self.fail_streak:
                logger.warning(
                    "JohansenStabilizer: %d consecutive cointegration failures — entering Read-Only mode.",
                    self._consecutive_fails,
                )
                self.is_read_only = True
