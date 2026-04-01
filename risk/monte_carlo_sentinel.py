"""
risk/monte_carlo_sentinel.py
─────────────────────────────
Monte Carlo Drawdown Sentinel.

Runs fast in-process Monte Carlo simulation to estimate:
  P(daily loss limit hit within the next look-ahead period)

When the probability exceeds a configurable threshold the sentinel:
  • tightens the effective max-positions limit
  • reduces the position-size multiplier
  • (optionally) moves to defensive-only mode

The simulation is intentionally fast (numpy-vectorised over paths) and
operates entirely on recent P&L history stored in the execution loop.

Usage
─────
    sentinel = MonteCarloSentinel()
    sentinel.record_pnl(daily_pnl_fraction)     # call every EOD cycle

    result = sentinel.evaluate(
        current_daily_pnl=-0.008,               # fraction of session equity
        daily_loss_limit=-0.02,                 # config threshold
        positions_open=5,
        session_equity=100_000.0,
    )

    result.breach_probability   → float [0, 1]
    result.size_multiplier      → float applied to new position sizes
    result.max_positions_cap    → int | None (override max_positions)
    result.is_defensive         → bool
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_N_PATHS               = 1_000    # Monte Carlo paths
_LOOK_AHEAD_PERIODS    = 8        # number of intraday periods to simulate
_DEFAULT_BREACH_THRESH = 0.30     # P(breach) >= 30% → tighten
_DEFAULT_DEFENSIVE_THRESH = 0.60  # P(breach) >= 60% → defensive mode
_DEFAULT_HISTORY_LEN   = 60       # P&L history depth (days/sessions)
_MIN_HISTORY           = 5        # need at least this many samples to simulate

_STATE_FILE = "monte_carlo_sentinel_state.json"

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SentinelResult:
    breach_probability:  float = 0.0
    size_multiplier:     float = 1.0
    max_positions_cap:   Optional[int] = None
    is_defensive:        bool = False
    tier:                str = "green"    # green / amber / red / defensive
    simulated_paths:     int = 0
    vol_estimate:        float = 0.0
    mean_estimate:       float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Main class ─────────────────────────────────────────────────────────────────

class MonteCarloSentinel:
    """
    Monte Carlo Drawdown Sentinel.

    Parameters
    ──────────
    n_paths             : Monte Carlo path count
    look_ahead          : number of forward periods to simulate
    breach_threshold    : P(loss) >= this → amber tier (tighten sizing)
    defensive_threshold : P(loss) >= this → defensive tier (halt new entries)
    history_len         : rolling session P&L history depth
    data_dir            : directory for persistence
    """

    def __init__(
        self,
        n_paths: int                = _N_PATHS,
        look_ahead: int             = _LOOK_AHEAD_PERIODS,
        breach_threshold: float     = _DEFAULT_BREACH_THRESH,
        defensive_threshold: float  = _DEFAULT_DEFENSIVE_THRESH,
        history_len: int            = _DEFAULT_HISTORY_LEN,
        data_dir: str | Path        = "data",
        amber_size_mult: float      = 0.70,
        red_size_mult: float        = 0.45,
        defensive_size_mult: float  = 0.20,
        amber_max_pos: Optional[int] = None,
        red_max_pos: Optional[int]   = None,
        defensive_max_pos: int       = 2,
    ) -> None:
        self.n_paths             = n_paths
        self.look_ahead          = look_ahead
        self.breach_threshold    = breach_threshold
        self.defensive_threshold = defensive_threshold
        self.amber_size_mult     = amber_size_mult
        self.red_size_mult       = red_size_mult
        self.defensive_size_mult = defensive_size_mult
        self.amber_max_pos       = amber_max_pos
        self.red_max_pos         = red_max_pos
        self.defensive_max_pos   = defensive_max_pos

        self._history: deque = deque(maxlen=history_len)
        self._data_dir = Path(data_dir)
        self._state_path = self._data_dir / _STATE_FILE

        # Cached last result to avoid re-simulation every cycle
        self._last_result: Optional[SentinelResult] = None
        self._last_eval_ts: float = 0.0

        self._load_state()

    # ── P&L recording ──────────────────────────────────────────────────────────

    def record_pnl(self, daily_pnl_fraction: float) -> None:
        """Record one session's P&L as a fraction of session equity."""
        self._history.append(float(daily_pnl_fraction))
        self._save_state()

    # ── Core simulation ────────────────────────────────────────────────────────

    def _estimate_params(self) -> tuple[float, float]:
        """Return (mean, std) of historical P&L fractions."""
        if len(self._history) < _MIN_HISTORY:
            return 0.0, 0.005   # conservative default
        arr = np.array(list(self._history))
        return float(arr.mean()), float(arr.std() + 1e-9)

    def simulate(
        self,
        current_daily_pnl: float,
        daily_loss_limit: float,
    ) -> float:
        """
        Monte Carlo: estimate P(session loss exceeds daily_loss_limit
        by end of look_ahead periods), starting from current_daily_pnl.

        Returns breach_probability in [0, 1].
        """
        mu, sigma = self._estimate_params()
        # Per-period (intraday) drift and vol: assume sqrt(look_ahead) scaling
        period_mu    = mu / self._look_ahead_periods_for_day()
        period_sigma = sigma / math.sqrt(self._look_ahead_periods_for_day())

        np.random.seed(None)  # stochastic
        shocks = np.random.normal(
            loc=period_mu,
            scale=period_sigma,
            size=(self.n_paths, self.look_ahead),
        )
        cumulative = current_daily_pnl + shocks.cumsum(axis=1)
        # Check if any period in the path hits the loss limit
        breached = (cumulative < daily_loss_limit).any(axis=1)
        return float(breached.mean())

    def _look_ahead_periods_for_day(self) -> int:
        """Approximate total intraday periods (prevents div-by-zero)."""
        return max(1, self.look_ahead)

    # ── Evaluate ───────────────────────────────────────────────────────────────

    def evaluate(
        self,
        current_daily_pnl: float,
        daily_loss_limit: float = -0.02,
        positions_open: int = 0,
        session_equity: float = 100_000.0,
        force: bool = False,
    ) -> SentinelResult:
        """
        Run Monte Carlo and return a SentinelResult with sizing guidance.

        Parameters
        ──────────
        current_daily_pnl : today's realised P&L as fraction (e.g. -0.008 = -0.8%)
        daily_loss_limit  : session loss limit (negative fraction, e.g. -0.02)
        positions_open    : current number of open positions
        session_equity    : total session equity for log context
        force             : skip cache and always re-simulate
        """
        if len(self._history) < _MIN_HISTORY and not force:
            # Not enough data → neutral result
            return SentinelResult()

        prob = self.simulate(current_daily_pnl, daily_loss_limit)
        mu, sigma = self._estimate_params()

        # Tier logic
        if prob >= self.defensive_threshold:
            tier             = "defensive"
            size_mult        = self.defensive_size_mult
            max_pos          = self.defensive_max_pos
            is_defensive     = True
        elif prob >= (self.breach_threshold + self.defensive_threshold) / 2:
            tier             = "red"
            size_mult        = self.red_size_mult
            max_pos          = self.red_max_pos
            is_defensive     = False
        elif prob >= self.breach_threshold:
            tier             = "amber"
            size_mult        = self.amber_size_mult
            max_pos          = self.amber_max_pos
            is_defensive     = False
        else:
            tier             = "green"
            size_mult        = 1.0
            max_pos          = None
            is_defensive     = False

        result = SentinelResult(
            breach_probability = prob,
            size_multiplier    = size_mult,
            max_positions_cap  = max_pos,
            is_defensive       = is_defensive,
            tier               = tier,
            simulated_paths    = self.n_paths,
            vol_estimate       = sigma,
            mean_estimate      = mu,
        )

        self._last_result  = result
        self._last_eval_ts = time.time()

        if tier != "green":
            logger.info(
                "MonteCarloSentinel [%s]: P(breach)=%.1f%%  size×=%.2f  "
                "positions=%d  equity=$%.0f",
                tier.upper(), prob * 100, size_mult, positions_open, session_equity,
            )

        return result

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        state = {
            "history": list(self._history),
            "last_eval_ts": self._last_eval_ts,
        }
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(self._state_path)
        except Exception as exc:
            logger.debug("MonteCarloSentinel state save error: %s", exc)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            state = json.loads(self._state_path.read_text())
            for v in state.get("history", []):
                self._history.append(float(v))
            self._last_eval_ts = float(state.get("last_eval_ts", 0.0))
            logger.info(
                "MonteCarloSentinel loaded (history=%d samples)", len(self._history)
            )
        except Exception as exc:
            logger.warning("MonteCarloSentinel state load error: %s", exc)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        mu, sigma = self._estimate_params()
        last = self._last_result
        return {
            "history_len":      len(self._history),
            "vol_estimate":     sigma,
            "mean_estimate":    mu,
            "last_tier":        last.tier if last else "unknown",
            "last_breach_prob": last.breach_probability if last else None,
            "last_size_mult":   last.size_multiplier if last else 1.0,
            "last_eval_ts":     self._last_eval_ts,
        }
