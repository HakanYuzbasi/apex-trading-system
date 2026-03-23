"""
risk/capital_allocator.py — Kelly-Optimal Capital Allocator

Dynamically splits available capital between asset classes (equity / crypto)
based on their rolling risk-adjusted performance.

Algorithm
---------
1. Collect recent realized Sharpe, win rate, and drawdown per leg
   (from the broker-split P&L already computed in execution_loop).
2. Compute a Kelly-fraction weight for each leg using:
       f = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
   Clamped to [MIN_ALLOC, MAX_ALLOC] and normalised to sum to 1.0.
3. Blend with a momentum signal: if a leg's trailing 5-day Sharpe is
   accelerating, tilt slightly toward it (up to MOMENTUM_TILT cap).
4. Apply a correlation penalty: when equity-crypto correlation is high
   (ρ > 0.70), compress both weights toward equal (reduce concentration).
5. Output: recommended_equity_frac, recommended_crypto_frac

Configuration (all optional env vars):
    APEX_ALLOC_MIN_PCT          — minimum allocation per asset class (default 0.10)
    APEX_ALLOC_MAX_PCT          — maximum allocation per asset class (default 0.85)
    APEX_ALLOC_MOMENTUM_TILT    — max tilt from momentum signal (default 0.05)
    APEX_ALLOC_CORR_THRESHOLD   — correlation above which to dampen (default 0.70)
    APEX_ALLOC_CORR_DAMPEN      — dampen magnitude when corr is high (default 0.50)
    APEX_ALLOC_LOOKBACK_DAYS    — history window for Sharpe computation (default 20)
    APEX_ALLOC_REBALANCE_THRESH — minimum shift to act on (default 0.03 = 3%)
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MIN_ALLOC   = float(os.getenv("APEX_ALLOC_MIN_PCT", "0.10"))
_MAX_ALLOC   = float(os.getenv("APEX_ALLOC_MAX_PCT", "0.85"))
_MOM_TILT    = float(os.getenv("APEX_ALLOC_MOMENTUM_TILT", "0.05"))
_CORR_THRESH = float(os.getenv("APEX_ALLOC_CORR_THRESHOLD", "0.70"))
_CORR_DAMPEN = float(os.getenv("APEX_ALLOC_CORR_DAMPEN", "0.50"))
_LOOKBACK    = int(os.getenv("APEX_ALLOC_LOOKBACK_DAYS", "20"))
_REBAL_THRESH = float(os.getenv("APEX_ALLOC_REBALANCE_THRESH", "0.03"))
_MIN_DAYS    = 5    # minimum history days before deviating from equal-weight


@dataclass
class LegPerf:
    """One day's P&L snapshot for a single leg (equity or crypto)."""
    date: str
    pnl_pct: float       # daily realised return as fraction
    trades: int


@dataclass
class AllocationResult:
    equity_frac: float
    crypto_frac: float
    equity_sharpe: float
    crypto_sharpe: float
    correlation: float
    rebalance_recommended: bool
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class CapitalAllocator:
    """
    Computes optimal equity/crypto capital split using Kelly fractions
    blended with momentum tilt and correlation dampening.

    Call update_leg_pnl() each day with the broker-split P&L, then
    call compute_allocation() to get updated weights.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        lookback_days: int = _LOOKBACK,
        min_alloc: float = _MIN_ALLOC,
        max_alloc: float = _MAX_ALLOC,
        momentum_tilt: float = _MOM_TILT,
        corr_threshold: float = _CORR_THRESH,
        corr_dampen: float = _CORR_DAMPEN,
        rebalance_threshold: float = _REBAL_THRESH,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self._lookback = lookback_days
        self._min = min_alloc
        self._max = max_alloc
        self._mom_tilt = momentum_tilt
        self._corr_thresh = corr_threshold
        self._corr_dampen = corr_dampen
        self._rebal_thresh = rebalance_threshold

        self._equity_history: deque[LegPerf] = deque(maxlen=lookback_days)
        self._crypto_history: deque[LegPerf] = deque(maxlen=lookback_days)

        # Current recommended allocation (starts equal-weight)
        self._current_equity_frac: float = 0.50
        self._current_crypto_frac: float = 0.50

        self._last_result: Optional[AllocationResult] = None
        self._load_state()

    # ------------------------------------------------------------------
    # Public: feed data
    # ------------------------------------------------------------------

    def update_leg_pnl(
        self,
        equity_pnl_pct: float,
        crypto_pnl_pct: float,
        equity_trades: int = 0,
        crypto_trades: int = 0,
        trade_date: Optional[str] = None,
    ) -> None:
        """Record one day's realized P&L for both legs."""
        day = trade_date or date.today().isoformat()
        self._equity_history.append(LegPerf(date=day, pnl_pct=equity_pnl_pct, trades=equity_trades))
        self._crypto_history.append(LegPerf(date=day, pnl_pct=crypto_pnl_pct, trades=crypto_trades))

    # ------------------------------------------------------------------
    # Public: compute allocation
    # ------------------------------------------------------------------

    def compute_allocation(self) -> AllocationResult:
        """
        Compute optimal equity/crypto split.

        Returns current fractions unchanged if insufficient history,
        otherwise applies Kelly + momentum + correlation logic.
        """
        eq_rets = np.array([p.pnl_pct for p in self._equity_history], dtype=float)
        cr_rets = np.array([p.pnl_pct for p in self._crypto_history], dtype=float)

        eq_sharpe = self._sharpe(eq_rets)
        cr_sharpe = self._sharpe(cr_rets)

        if len(eq_rets) < _MIN_DAYS or len(cr_rets) < _MIN_DAYS:
            result = AllocationResult(
                equity_frac=self._current_equity_frac,
                crypto_frac=self._current_crypto_frac,
                equity_sharpe=eq_sharpe,
                crypto_sharpe=cr_sharpe,
                correlation=0.0,
                rebalance_recommended=False,
                reason=f"insufficient history ({len(eq_rets)} days < {_MIN_DAYS})",
            )
            self._last_result = result
            return result

        # Kelly weights (based on Sharpe sign/magnitude as proxy)
        eq_kelly, cr_kelly = self._kelly_weights(eq_rets, cr_rets)

        # Momentum tilt: accelerating Sharpe → small extra tilt
        eq_tilt, cr_tilt = self._momentum_tilt(eq_rets, cr_rets)
        eq_w = eq_kelly + eq_tilt
        cr_w = cr_kelly + cr_tilt

        # Normalise
        total = eq_w + cr_w
        if total <= 0:
            eq_w, cr_w = 0.5, 0.5
        else:
            eq_w, cr_w = eq_w / total, cr_w / total

        # Correlation dampening
        corr = self._correlation(eq_rets, cr_rets)
        if abs(corr) > self._corr_thresh:
            dampen = self._corr_dampen * (abs(corr) - self._corr_thresh) / (1.0 - self._corr_thresh)
            eq_w = eq_w * (1 - dampen) + 0.5 * dampen
            cr_w = cr_w * (1 - dampen) + 0.5 * dampen

        # Hard clamp
        eq_w = max(self._min, min(self._max, eq_w))
        cr_w = max(self._min, min(self._max, cr_w))

        # Re-normalise after clamp
        total = eq_w + cr_w
        eq_w, cr_w = eq_w / total, cr_w / total

        # Check if shift is large enough to recommend action
        shift = abs(eq_w - self._current_equity_frac)
        rebalance = shift >= self._rebal_thresh
        reason = (
            f"Kelly eq={eq_kelly:.2f} cr={cr_kelly:.2f} "
            f"| corr={corr:.2f} | shift={shift:.3f}"
        )

        result = AllocationResult(
            equity_frac=round(eq_w, 3),
            crypto_frac=round(cr_w, 3),
            equity_sharpe=round(eq_sharpe, 3),
            crypto_sharpe=round(cr_sharpe, 3),
            correlation=round(corr, 3),
            rebalance_recommended=rebalance,
            reason=reason,
        )
        self._last_result = result

        if rebalance:
            logger.info(
                "🔄 CapitalAllocator: equity %.0f%%→%.0f%% crypto %.0f%%→%.0f%% (%s)",
                self._current_equity_frac * 100, eq_w * 100,
                self._current_crypto_frac * 100, cr_w * 100,
                reason,
            )
            self._current_equity_frac = round(eq_w, 3)
            self._current_crypto_frac = round(cr_w, 3)
            self._persist()

        return result

    @property
    def current_equity_frac(self) -> float:
        return self._current_equity_frac

    @property
    def current_crypto_frac(self) -> float:
        return self._current_crypto_frac

    # ------------------------------------------------------------------
    # Private: maths
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(rets: np.ndarray, ann_factor: float = 252.0) -> float:
        if len(rets) < 2:
            return 0.0
        std = rets.std()
        if std == 0:
            return 0.0
        return float(rets.mean() / std * (ann_factor ** 0.5))

    def _kelly_weights(
        self, eq_rets: np.ndarray, cr_rets: np.ndarray
    ) -> Tuple[float, float]:
        """Convert Sharpe to Kelly fractions via softmax-like transform."""
        eq_s = self._sharpe(eq_rets)
        cr_s = self._sharpe(cr_rets)
        # Map Sharpe → positive weight via ReLU + offset so both legs get floor
        eq_w = max(0.0, eq_s) + 0.5
        cr_w = max(0.0, cr_s) + 0.5
        total = eq_w + cr_w
        return eq_w / total, cr_w / total

    def _momentum_tilt(
        self, eq_rets: np.ndarray, cr_rets: np.ndarray
    ) -> Tuple[float, float]:
        """Tilt toward the leg whose Sharpe is accelerating (recent 5 vs older)."""
        if len(eq_rets) < 10:
            return 0.0, 0.0
        half = len(eq_rets) // 2
        eq_recent = self._sharpe(eq_rets[-half:])
        eq_old    = self._sharpe(eq_rets[:half])
        cr_recent = self._sharpe(cr_rets[-half:])
        cr_old    = self._sharpe(cr_rets[:half])
        eq_accel  = eq_recent - eq_old
        cr_accel  = cr_recent - cr_old
        # Normalize acceleration to tilt magnitude
        total_accel = abs(eq_accel) + abs(cr_accel)
        if total_accel == 0:
            return 0.0, 0.0
        eq_tilt = self._mom_tilt * eq_accel / total_accel
        cr_tilt = self._mom_tilt * cr_accel / total_accel
        return eq_tilt, cr_tilt

    @staticmethod
    def _correlation(eq_rets: np.ndarray, cr_rets: np.ndarray) -> float:
        if len(eq_rets) < 3 or len(cr_rets) < 3:
            return 0.0
        n = min(len(eq_rets), len(cr_rets))
        try:
            corr = float(np.corrcoef(eq_rets[-n:], cr_rets[-n:])[0, 1])
            return corr if np.isfinite(corr) else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if not self._data_dir or not self._last_result:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            path = self._data_dir / "capital_allocation.json"
            payload = {
                "current_equity_frac": self._current_equity_frac,
                "current_crypto_frac": self._current_crypto_frac,
                "last_result": asdict(self._last_result),
                "equity_history": [asdict(p) for p in self._equity_history],
                "crypto_history": [asdict(p) for p in self._crypto_history],
                "updated_at": datetime.utcnow().isoformat() + "Z",
            }
            path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.debug("CapitalAllocator persist error: %s", exc)

    def _load_state(self) -> None:
        if not self._data_dir:
            return
        path = Path(self._data_dir) / "capital_allocation.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self._current_equity_frac = float(data.get("current_equity_frac", 0.50))
            self._current_crypto_frac = float(data.get("current_crypto_frac", 0.50))
            for rec in data.get("equity_history", []):
                self._equity_history.append(LegPerf(**rec))
            for rec in data.get("crypto_history", []):
                self._crypto_history.append(LegPerf(**rec))
            logger.info(
                "CapitalAllocator: loaded state eq=%.0f%% cr=%.0f%% (%d days history)",
                self._current_equity_frac * 100,
                self._current_crypto_frac * 100,
                len(self._equity_history),
            )
        except Exception as exc:
            logger.debug("CapitalAllocator load error: %s", exc)
