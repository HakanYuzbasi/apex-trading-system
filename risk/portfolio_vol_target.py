"""
risk/portfolio_vol_target.py — Portfolio Volatility Targeting

Dynamically adjusts aggregate position sizing to target a specific portfolio
annualised volatility. This prevents the system from taking full-size positions
when realised volatility is elevated (naturally reduces risk in choppy markets).

Algorithm:
  1. Compute realised vol from the equity curve (log-returns of portfolio value)
     over the last LOOKBACK_DAYS bars.
  2. Compute scaling multiplier = TARGET_VOL / realised_vol, clamped to [MIN_MULT, MAX_MULT].
  3. Multiply every new position's shares by this multiplier.
  4. Update every VOL_UPDATE_INTERVAL_CYCLES cycles (lightweight EMA update).

Equity curve is maintained by the execution_loop via record_equity().

Usage:
    vt = PortfolioVolTarget()
    vt.record_equity(current_portfolio_value)
    mult = vt.get_multiplier()   # apply to position sizing

Config keys:
    VOL_TARGET_ENABLED              = True
    VOL_TARGET_ANNUALISED           = 0.12   # 12% target annual vol
    VOL_TARGET_LOOKBACK_DAYS        = 30     # bars for realised vol estimation
    VOL_TARGET_MIN_MULT             = 0.50   # never scale down more than 50%
    VOL_TARGET_MAX_MULT             = 1.50   # never scale up more than 150%
    VOL_TARGET_MIN_OBS              = 10     # minimum obs before multiplier kicks in
    VOL_TARGET_BARS_PER_YEAR        = 252    # trading days per year for annualisation
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "VOL_TARGET_ENABLED":       True,
    "VOL_TARGET_ANNUALISED":    0.12,
    "VOL_TARGET_LOOKBACK_DAYS": 30,
    "VOL_TARGET_MIN_MULT":      0.50,
    "VOL_TARGET_MAX_MULT":      1.50,
    "VOL_TARGET_MIN_OBS":       10,
    "VOL_TARGET_BARS_PER_YEAR": 252,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


# ── Core maths ─────────────────────────────────────────────────────────────────

def compute_realised_vol(equity_series: list, bars_per_year: int = 252) -> float:
    """
    Compute annualised realised volatility from an equity series.

    Args:
        equity_series: List of portfolio values (newest last).
        bars_per_year: Annualisation factor (252 for daily bars).

    Returns:
        Annualised volatility as decimal (e.g. 0.18 = 18%).
        Returns 0.0 if insufficient data or zero variance.
    """
    if len(equity_series) < 2:
        return 0.0

    arr = np.array(equity_series, dtype=float)
    # Filter out zeros/negatives
    arr = arr[arr > 0]
    if len(arr) < 2:
        return 0.0

    log_rets = np.log(arr[1:] / arr[:-1])
    if len(log_rets) < 2:
        return 0.0

    daily_vol = float(np.std(log_rets, ddof=1))
    return float(daily_vol * math.sqrt(bars_per_year))


def compute_vol_target_multiplier(
    realised_vol: float,
    target_vol: float,
    min_mult: float,
    max_mult: float,
) -> float:
    """
    Compute sizing multiplier to achieve target vol.

    mult = target_vol / realised_vol, clamped to [min_mult, max_mult].
    Returns 1.0 if realised vol is near zero (no data).
    """
    if realised_vol < 1e-6:
        return 1.0

    mult = target_vol / realised_vol
    return float(np.clip(mult, min_mult, max_mult))


# ── PortfolioVolTarget ────────────────────────────────────────────────────────

@dataclass
class VolTargetState:
    """Snapshot of vol targeting state."""
    realised_vol: float = 0.0
    target_vol: float = 0.12
    multiplier: float = 1.0
    n_obs: int = 0
    active: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "realised_vol_pct": round(self.realised_vol * 100, 2),
            "target_vol_pct": round(self.target_vol * 100, 2),
            "multiplier": round(self.multiplier, 4),
            "n_obs": self.n_obs,
            "active": self.active,
            "timestamp": self.timestamp,
        }


class PortfolioVolTarget:
    """
    Portfolio volatility targeting.

    Maintains a rolling equity curve buffer and computes a dynamic position
    sizing multiplier to target annualised portfolio volatility.
    """

    def __init__(self):
        lookback = int(_cfg("VOL_TARGET_LOOKBACK_DAYS"))
        self._equity_buf: Deque[float] = deque(maxlen=lookback + 10)
        self._current_mult: float = 1.0
        self._last_state: Optional[VolTargetState] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_equity(self, portfolio_value: float) -> None:
        """Add a new portfolio value observation to the rolling buffer."""
        if portfolio_value > 0:
            self._equity_buf.append(float(portfolio_value))

    def get_multiplier(self) -> float:
        """Return the current position sizing multiplier."""
        if not _cfg("VOL_TARGET_ENABLED"):
            return 1.0
        self._update()
        return self._current_mult

    def get_state(self) -> VolTargetState:
        """Return current vol targeting state snapshot."""
        self._update()
        return self._last_state or VolTargetState()

    def get_report(self) -> Dict:
        """Return JSON-serialisable state."""
        return self.get_state().to_dict()

    def reset(self) -> None:
        """Clear equity buffer and reset multiplier."""
        self._equity_buf.clear()
        self._current_mult = 1.0
        self._last_state = None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _update(self) -> None:
        if not _cfg("VOL_TARGET_ENABLED"):
            self._current_mult = 1.0
            return

        n_obs = len(self._equity_buf)
        min_obs = int(_cfg("VOL_TARGET_MIN_OBS"))
        target = float(_cfg("VOL_TARGET_ANNUALISED"))
        min_mult = float(_cfg("VOL_TARGET_MIN_MULT"))
        max_mult = float(_cfg("VOL_TARGET_MAX_MULT"))
        bars_per_year = int(_cfg("VOL_TARGET_BARS_PER_YEAR"))

        if n_obs < min_obs:
            self._current_mult = 1.0
            self._last_state = VolTargetState(
                target_vol=target,
                multiplier=1.0,
                n_obs=n_obs,
                active=False,
            )
            return

        series = list(self._equity_buf)
        realised = compute_realised_vol(series, bars_per_year)
        mult = compute_vol_target_multiplier(realised, target, min_mult, max_mult)
        self._current_mult = mult

        if abs(mult - 1.0) > 0.05:
            logger.debug(
                "VolTarget: realised=%.1f%% target=%.1f%% → mult=%.3f",
                realised * 100, target * 100, mult,
            )

        self._last_state = VolTargetState(
            realised_vol=realised,
            target_vol=target,
            multiplier=mult,
            n_obs=n_obs,
            active=True,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_vol_target: Optional[PortfolioVolTarget] = None


def get_vol_target() -> PortfolioVolTarget:
    global _vol_target
    if _vol_target is None:
        _vol_target = PortfolioVolTarget()
    return _vol_target
