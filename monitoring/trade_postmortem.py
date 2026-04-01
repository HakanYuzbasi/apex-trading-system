"""
monitoring/trade_postmortem.py — Trade Post-Mortem Intelligence

Generates a structured post-mortem diagnosis at every trade close, classifying
what went right or wrong and storing a ring-buffer of recent cases.

Diagnoses 5 categories:
  1. Signal quality   — was signal strong/weak/misaligned at entry?
  2. Timing          — did entry/exit happen at the right time within holding horizon?
  3. Regime alignment — was the regime compatible with the trade direction?
  4. Execution drag   — was slippage meaningful relative to P&L?
  5. Verdict         — overall: winner / loser / breakeven

Output:
  - PostMortem dataclass with per-category verdict + overall assessment
  - get_recent(n)  → last N post-mortems
  - get_summary()  → aggregated win/loss/breakeven breakdown and dominant causes

Integration (execution_loop.py at trade close, before cleanup):
    pm = self._trade_postmortem.analyze(trade_close_context)
    if pm.verdict == "loser":
        logger.info("PostMortem %s: %s", symbol, pm.primary_failure)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional


# ── Thresholds ─────────────────────────────────────────────────────────────────
_STRONG_SIGNAL = 0.20     # signal above this is considered strong
_WEAK_SIGNAL   = 0.12     # signal below this is considered weak
_GOOD_CONF     = 0.65     # confidence above this is good
_SLIPPAGE_BAD  = 15.0     # bps above which slippage is "meaningful"
_PNL_WINNER    = 0.005    # > +0.5% = winner
_PNL_LOSER     = -0.005   # < -0.5% = loser
_GOOD_REGIMES  = frozenset({"bull", "strong_bull", "neutral"})
_BAD_REGIMES   = frozenset({"bear", "strong_bear", "crisis"})
_MAX_HISTORY   = 200


@dataclass
class PostMortem:
    symbol: str
    pnl_pct: float
    hold_hours: float
    exit_reason: str

    # Per-category assessment (each: "good" | "neutral" | "bad")
    signal_quality: str
    timing: str
    regime_alignment: str
    execution_drag: str

    # Derived
    verdict: str              # "winner" | "loser" | "breakeven"
    primary_failure: str      # e.g. "weak_signal", "bad_regime", "slippage", "early_exit"
    confidence_at_entry: float
    signal_at_entry: float
    slippage_bps: float
    regime: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "symbol":             self.symbol,
            "pnl_pct":            round(self.pnl_pct, 6),
            "hold_hours":         round(self.hold_hours, 2),
            "exit_reason":        self.exit_reason,
            "signal_quality":     self.signal_quality,
            "timing":             self.timing,
            "regime_alignment":   self.regime_alignment,
            "execution_drag":     self.execution_drag,
            "verdict":            self.verdict,
            "primary_failure":    self.primary_failure,
            "confidence_at_entry": round(self.confidence_at_entry, 4),
            "signal_at_entry":    round(self.signal_at_entry, 4),
            "slippage_bps":       round(self.slippage_bps, 2),
            "regime":             self.regime,
            "timestamp":          self.timestamp,
        }


class TradePostMortem:
    """
    Lightweight post-trade diagnosis engine.

    Thread-safe for reading (get_recent / get_summary).
    analyze() called synchronously at trade close — non-blocking (pure computation).
    """

    def __init__(self, max_history: int = _MAX_HISTORY) -> None:
        self._history: Deque[PostMortem] = deque(maxlen=max_history)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        pnl_pct: float,
        hold_hours: float,
        exit_reason: str,
        signal_at_entry: float,
        confidence_at_entry: float,
        regime: str,
        slippage_bps: float = 0.0,
        optimal_hold_hours: float = 4.0,
    ) -> PostMortem:
        """Compute and store a post-mortem. Returns the PostMortem record."""
        pm = PostMortem(
            symbol=symbol,
            pnl_pct=pnl_pct,
            hold_hours=hold_hours,
            exit_reason=exit_reason,
            signal_quality=self._assess_signal(signal_at_entry, confidence_at_entry),
            timing=self._assess_timing(hold_hours, optimal_hold_hours, pnl_pct, exit_reason),
            regime_alignment=self._assess_regime(regime, pnl_pct),
            execution_drag=self._assess_execution(slippage_bps, pnl_pct),
            verdict=self._verdict(pnl_pct),
            primary_failure=self._primary_failure(
                pnl_pct, signal_at_entry, confidence_at_entry,
                regime, slippage_bps, hold_hours, optimal_hold_hours, exit_reason,
            ),
            confidence_at_entry=confidence_at_entry,
            signal_at_entry=signal_at_entry,
            slippage_bps=slippage_bps,
            regime=regime,
        )
        self._history.append(pm)
        return pm

    def get_recent(self, n: int = 20) -> List[dict]:
        """Return last N post-mortems as dicts, most recent first."""
        items = list(self._history)
        return [pm.to_dict() for pm in reversed(items[-n:])]

    def get_summary(self) -> dict:
        """Aggregate statistics over all stored post-mortems."""
        items = list(self._history)
        if not items:
            return {"total": 0, "verdict_counts": {}, "failure_counts": {}, "avg_pnl_pct": 0.0}

        verdict_counts: Dict[str, int] = {}
        failure_counts: Dict[str, int] = {}
        pnl_sum = 0.0
        for pm in items:
            verdict_counts[pm.verdict] = verdict_counts.get(pm.verdict, 0) + 1
            failure_counts[pm.primary_failure] = failure_counts.get(pm.primary_failure, 0) + 1
            pnl_sum += pm.pnl_pct

        return {
            "total": len(items),
            "verdict_counts": verdict_counts,
            "failure_counts": dict(sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)),
            "avg_pnl_pct": round(pnl_sum / len(items), 6),
            "win_rate": round(verdict_counts.get("winner", 0) / len(items), 4),
        }

    # ── Assessment helpers ────────────────────────────────────────────────────

    @staticmethod
    def _assess_signal(signal: float, confidence: float) -> str:
        if abs(signal) >= _STRONG_SIGNAL and confidence >= _GOOD_CONF:
            return "good"
        if abs(signal) <= _WEAK_SIGNAL or confidence < 0.45:
            return "bad"
        return "neutral"

    @staticmethod
    def _assess_timing(
        hold_hours: float, optimal: float, pnl_pct: float, exit_reason: str
    ) -> str:
        # Stop-loss or max-hold exits that went badly = poor timing
        if exit_reason in ("stop_loss", "hard_stop") and pnl_pct < _PNL_LOSER:
            return "bad"
        # Very short holds (< 30min) on a loser = premature stop or noise
        if hold_hours < 0.5 and pnl_pct < 0:
            return "bad"
        # Optimal hold alignment
        ratio = hold_hours / max(optimal, 0.1)
        if 0.5 <= ratio <= 2.5:
            return "good"
        return "neutral"

    @staticmethod
    def _assess_regime(regime: str, pnl_pct: float) -> str:
        r = regime.lower()
        if r in _GOOD_REGIMES:
            return "good"
        if r in _BAD_REGIMES and pnl_pct < 0:
            return "bad"   # traded in bad regime and lost
        return "neutral"

    @staticmethod
    def _assess_execution(slippage_bps: float, pnl_pct: float) -> str:
        if slippage_bps > _SLIPPAGE_BAD:
            # Slippage > 15bps is bad; even worse if trade was a loser
            return "bad" if pnl_pct < 0 else "neutral"
        return "good"

    @staticmethod
    def _verdict(pnl_pct: float) -> str:
        if pnl_pct > _PNL_WINNER:
            return "winner"
        if pnl_pct < _PNL_LOSER:
            return "loser"
        return "breakeven"

    @staticmethod
    def _primary_failure(
        pnl_pct: float,
        signal: float,
        confidence: float,
        regime: str,
        slippage_bps: float,
        hold_hours: float,
        optimal: float,
        exit_reason: str,
    ) -> str:
        """Identify the single most likely root cause for bad trades (or "none" for winners)."""
        if pnl_pct > _PNL_WINNER:
            return "none"

        # Rank candidates by severity
        candidates: List[tuple] = []

        # 1. Slippage eating into P&L
        if slippage_bps > _SLIPPAGE_BAD:
            candidates.append((slippage_bps / 100.0, "slippage"))

        # 2. Weak signal quality
        if abs(signal) < _WEAK_SIGNAL:
            candidates.append((0.8, "weak_signal"))
        elif confidence < 0.45:
            candidates.append((0.7, "low_confidence"))

        # 3. Bad regime
        if regime.lower() in _BAD_REGIMES:
            candidates.append((0.6, "bad_regime"))

        # 4. Timing issues
        if hold_hours < 0.5 and pnl_pct < 0:
            candidates.append((0.5, "premature_exit"))
        elif hold_hours / max(optimal, 0.1) > 3.0:
            candidates.append((0.4, "held_too_long"))

        # 5. Exit reason
        if exit_reason in ("stop_loss", "hard_stop"):
            candidates.append((0.3, "stop_hit"))
        elif exit_reason == "max_hold":
            candidates.append((0.2, "max_hold_expired"))

        if not candidates:
            return "unknown"
        # Return the failure with the highest weight
        return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
