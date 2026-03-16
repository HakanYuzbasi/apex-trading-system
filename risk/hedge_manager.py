"""
risk/hedge_manager.py - Systematic Portfolio Hedge Manager

Generates hedge signal adjustments to protect the portfolio during:
1. Correlation Crisis  — avg pairwise correlation > threshold (positions act as one)
2. VIX Elevated/Fear  — realized volatility rising, asymmetric downside risk
3. Accelerating Drawdown — today's loss already exceeds a warning threshold
4. ML/Technical Disagreement — models pointing opposite directions (uncertainty)

The manager does NOT place orders directly. It returns a `HedgeAdjustment`
that the execution loop applies to the raw signal BEFORE the HOLD/BUY/SELL
decision. This keeps the hedge logic composable and testable.

Signal adjustment logic (applied multiplicatively):
  adjusted_signal = raw_signal * hedge_dampener

  hedge_dampener < 1.0 → dampen BUY conviction (exit sooner, size smaller)
  hedge_dampener > 1.0 → amplify SELL conviction (exit faster when losing)

When all triggers are active simultaneously:
  dampener = corr_factor * vix_factor * dd_factor  (product of individual factors)
  Minimum dampener = 0.10 (never fully zero out a signal, preserves directionality)

Additional "force-exit" flag:
  If daily drawdown > FORCE_EXIT_DD_PCT AND correlation > CRISIS_THRESHOLD,
  the manager sets force_exit=True so that any losing LONG position is closed
  regardless of signal strength.

Usage:
    hedge_mgr = HedgeManager()
    adj = hedge_mgr.get_adjustment(
        symbol="BTC/USD",
        position_side="LONG",
        base_signal=0.12,
        portfolio_avg_correlation=0.91,
        vix=24.2,
        daily_pnl_pct=-0.86,
        ml_signal=-0.24,
        tech_signal=0.33,
    )
    signal = base_signal * adj.dampener
    if adj.force_exit and pnl < 0:
        should_exit = True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

# Correlation thresholds
CORR_WARNING_THRESHOLD: float = 0.70   # Start dampening
CORR_CRISIS_THRESHOLD: float = 0.85    # Strong dampening + force-exit on losses

# VIX thresholds
VIX_ELEVATED: float = 20.0
VIX_FEAR: float = 30.0
VIX_PANIC: float = 40.0

# Daily drawdown thresholds
DD_WARNING_PCT: float = 0.005   # -0.5% today → mild dampening
DD_CONCERN_PCT: float = 0.010   # -1.0% today → moderate dampening
DD_ALARM_PCT: float = 0.020     # -2.0% today → force-exit enabled

# ML vs technical disagreement threshold
# When |ml_signal| > threshold and sign differs from tech_signal, reduce conviction
ML_TECH_DISAGREEMENT_THRESHOLD: float = 0.10

# Floor: never dampen below this even in worst conditions
MIN_DAMPENER: float = 0.10

# Cooldown: log hedge state changes at most every N minutes
_LOG_COOLDOWN_MINUTES: int = 10


# ─── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class HedgeAdjustment:
    """Result of hedge evaluation for a single symbol."""
    symbol: str
    dampener: float           # Multiply raw signal by this (0.10–1.0)
    force_exit: bool          # True → exit any losing long position immediately
    reasons: list             # Human-readable list of active triggers
    # Individual factor breakdown (for logging/debugging)
    corr_factor: float = 1.0
    vix_factor: float = 1.0
    dd_factor: float = 1.0
    disagreement_factor: float = 1.0


# ─── HedgeManager ─────────────────────────────────────────────────────────────

class HedgeManager:
    """
    Computes multiplicative signal dampeners and force-exit flags.

    Thread-safe for async use: all state is read-only after __init__;
    the only mutable state is the log-cooldown timestamp dict.
    """

    def __init__(
        self,
        corr_warning: float = CORR_WARNING_THRESHOLD,
        corr_crisis: float = CORR_CRISIS_THRESHOLD,
        vix_elevated: float = VIX_ELEVATED,
        vix_fear: float = VIX_FEAR,
        vix_panic: float = VIX_PANIC,
        dd_warning: float = DD_WARNING_PCT,
        dd_concern: float = DD_CONCERN_PCT,
        dd_alarm: float = DD_ALARM_PCT,
        min_dampener: float = MIN_DAMPENER,
    ) -> None:
        self.corr_warning = corr_warning
        self.corr_crisis = corr_crisis
        self.vix_elevated = vix_elevated
        self.vix_fear = vix_fear
        self.vix_panic = vix_panic
        self.dd_warning = dd_warning
        self.dd_concern = dd_concern
        self.dd_alarm = dd_alarm
        self.min_dampener = min_dampener
        self._last_log: Dict[str, datetime] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def get_adjustment(
        self,
        symbol: str,
        position_side: str,           # "LONG" | "SHORT" | "FLAT"
        base_signal: float,           # blended signal before hedge
        portfolio_avg_correlation: float = 0.0,
        vix: float = 15.0,
        daily_pnl_pct: float = 0.0,   # negative means losing (e.g. -0.01 = -1%)
        ml_signal: Optional[float] = None,
        tech_signal: Optional[float] = None,
    ) -> HedgeAdjustment:
        """
        Compute the hedge dampener and force-exit flag for one symbol.

        Args:
            symbol: Apex symbol string (e.g. "BTC/USD", "CRYPTO:ETH/USD").
            position_side: Current position direction.
            base_signal: The already-blended signal (after ML+tech+fear&greed).
            portfolio_avg_correlation: Avg pairwise correlation of the portfolio.
            vix: Current VIX level.
            daily_pnl_pct: Today's P&L as a fraction (negative = loss).
            ml_signal: Raw ML component signal (for disagreement check).
            tech_signal: Raw technical component signal (for disagreement check).

        Returns:
            HedgeAdjustment with dampener ∈ [min_dampener, 1.0] and force_exit flag.
        """
        reasons: list = []

        # ── 1. Correlation factor ──────────────────────────────────────────
        corr_factor = self._corr_factor(portfolio_avg_correlation, reasons)

        # ── 2. VIX factor ─────────────────────────────────────────────────
        vix_factor = self._vix_factor(vix, reasons)

        # ── 3. Drawdown factor ────────────────────────────────────────────
        dd_factor = self._dd_factor(daily_pnl_pct, reasons)

        # ── 4. ML/Tech disagreement factor ────────────────────────────────
        dis_factor = self._disagreement_factor(ml_signal, tech_signal, reasons)

        # ── 5. Combine ────────────────────────────────────────────────────
        raw_dampener = corr_factor * vix_factor * dd_factor * dis_factor
        dampener = max(self.min_dampener, min(1.0, raw_dampener))

        # ── 6. Force-exit flag: losing LONG + crisis correlation + alarm DD ─
        force_exit = (
            position_side == "LONG"
            and portfolio_avg_correlation >= self.corr_crisis
            and abs(daily_pnl_pct) >= self.dd_alarm
        )
        if force_exit:
            reasons.append(
                f"FORCE-EXIT: corr={portfolio_avg_correlation:.2f} + dd={daily_pnl_pct:.1%}"
            )

        adj = HedgeAdjustment(
            symbol=symbol,
            dampener=dampener,
            force_exit=force_exit,
            reasons=reasons,
            corr_factor=corr_factor,
            vix_factor=vix_factor,
            dd_factor=dd_factor,
            disagreement_factor=dis_factor,
        )

        self._maybe_log(symbol, adj)
        return adj

    # ── Factor helpers ──────────────────────────────────────────────────────

    def _corr_factor(self, corr: float, reasons: list) -> float:
        """Dampen signals when portfolio correlation is high."""
        if corr <= self.corr_warning:
            return 1.0
        if corr >= self.corr_crisis:
            # CRISIS: correlation so high that diversification is near-zero.
            # Treat the portfolio as a single position — 50% conviction cut.
            reasons.append(f"corr_crisis({corr:.2f})")
            return 0.50
        # Linear interpolation between warning and crisis
        severity = (corr - self.corr_warning) / (self.corr_crisis - self.corr_warning)
        factor = 1.0 - 0.50 * severity   # 1.0 → 0.50
        reasons.append(f"corr_warning({corr:.2f}→×{factor:.2f})")
        return factor

    def _vix_factor(self, vix: float, reasons: list) -> float:
        """Reduce conviction as VIX rises through regime bands."""
        if vix < self.vix_elevated:
            return 1.0
        if vix >= self.vix_panic:
            reasons.append(f"vix_panic({vix:.1f})")
            return 0.40    # 60% conviction cut in panic
        if vix >= self.vix_fear:
            # Linear: 0.65 at fear threshold → 0.40 at panic threshold
            t = (vix - self.vix_fear) / (self.vix_panic - self.vix_fear)
            factor = 0.65 - 0.25 * t
            reasons.append(f"vix_fear({vix:.1f}→×{factor:.2f})")
            return factor
        # Elevated band: 0.85 at 20 → 0.65 at 30
        t = (vix - self.vix_elevated) / (self.vix_fear - self.vix_elevated)
        factor = 0.85 - 0.20 * t
        reasons.append(f"vix_elevated({vix:.1f}→×{factor:.2f})")
        return factor

    def _dd_factor(self, daily_pnl_pct: float, reasons: list) -> float:
        """Tighten conviction when the day's running loss is mounting."""
        loss = -daily_pnl_pct  # positive when losing
        if loss < self.dd_warning:
            return 1.0
        if loss >= self.dd_alarm:
            reasons.append(f"dd_alarm({daily_pnl_pct:.1%})")
            return 0.60
        if loss >= self.dd_concern:
            t = (loss - self.dd_concern) / (self.dd_alarm - self.dd_concern)
            factor = 0.80 - 0.20 * t   # 0.80 → 0.60
            reasons.append(f"dd_concern({daily_pnl_pct:.1%}→×{factor:.2f})")
            return factor
        # Warning band
        t = (loss - self.dd_warning) / (self.dd_concern - self.dd_warning)
        factor = 1.0 - 0.20 * t        # 1.0 → 0.80
        reasons.append(f"dd_warning({daily_pnl_pct:.1%}→×{factor:.2f})")
        return factor

    def _disagreement_factor(
        self,
        ml_signal: Optional[float],
        tech_signal: Optional[float],
        reasons: list,
    ) -> float:
        """Penalise when ML and technical signals disagree strongly."""
        if ml_signal is None or tech_signal is None:
            return 1.0
        if abs(ml_signal) < ML_TECH_DISAGREEMENT_THRESHOLD:
            return 1.0
        if (ml_signal > 0) != (tech_signal > 0):
            # Opposite signs AND ML signal is non-trivial → low confidence
            mag = min(1.0, abs(ml_signal))
            factor = 1.0 - 0.30 * mag   # up to 30% conviction cut
            reasons.append(f"ml_tech_disagree(ml={ml_signal:+.2f} tech={tech_signal:+.2f}→×{factor:.2f})")
            return factor
        return 1.0

    # ── Logging ─────────────────────────────────────────────────────────────

    def _maybe_log(self, symbol: str, adj: HedgeAdjustment) -> None:
        """Log hedge state at most every _LOG_COOLDOWN_MINUTES per symbol."""
        if not adj.reasons:
            return
        now = datetime.now()
        last = self._last_log.get(symbol)
        if last and (now - last).total_seconds() < _LOG_COOLDOWN_MINUTES * 60:
            return
        self._last_log[symbol] = now
        logger.warning(
            "🛡️ HedgeManager [%s]: dampener=×%.2f force_exit=%s — %s",
            symbol, adj.dampener, adj.force_exit, " | ".join(adj.reasons),
        )

    # ── Utility ─────────────────────────────────────────────────────────────

    def summary(
        self,
        portfolio_avg_correlation: float,
        vix: float,
        daily_pnl_pct: float,
    ) -> str:
        """One-line human-readable hedge status for logging."""
        parts = []
        if portfolio_avg_correlation >= self.corr_crisis:
            parts.append(f"corr=CRISIS({portfolio_avg_correlation:.2f})")
        elif portfolio_avg_correlation >= self.corr_warning:
            parts.append(f"corr=WARN({portfolio_avg_correlation:.2f})")
        if vix >= self.vix_fear:
            parts.append(f"vix=FEAR({vix:.1f})")
        elif vix >= self.vix_elevated:
            parts.append(f"vix=ELEVATED({vix:.1f})")
        if -daily_pnl_pct >= self.dd_alarm:
            parts.append(f"dd=ALARM({daily_pnl_pct:.1%})")
        elif -daily_pnl_pct >= self.dd_warning:
            parts.append(f"dd=WARN({daily_pnl_pct:.1%})")
        return " | ".join(parts) if parts else "no_hedge_triggers"
