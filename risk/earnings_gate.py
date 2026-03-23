"""
risk/earnings_gate.py — Pre-Earnings Sizing Gate

Systematically manages position sizing and entry decisions around earnings:

  Pre-earnings (approaching):
    - 72h–48h before: size × 0.75  (caution zone)
    - 48h–24h before: size × 0.50  (warning zone)
    - <24h before:    size × 0.25  (danger zone; MacroEventShield blocks entries)

  Post-earnings (PEAD window):
    - 0–3 days after: no additional dampening (PEAD plays allowed)
    - 3–30 days after: normal sizing

Only applies to equity symbols. Crypto, FX, and indices are pass-through.

Wire-in:
    1. gate.update_earnings(symbol, next_earnings_dt)  — call periodically
    2. mult = gate.get_sizing_mult(symbol)              — call in sizing stack
    3. report = gate.get_report()                       — for dashboard
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Hours-before thresholds → multiplier
_PRE_EARNINGS_TIERS = [
    (24,  0.25),   # <24h  → danger zone
    (48,  0.50),   # 24-48h → warning
    (72,  0.75),   # 48-72h → caution
]
_NORMAL_MULT = 1.0


class EarningsEventGate:
    """
    Pre-earnings position sizing dampener.

    Thread-safe for read access; update_earnings should be called
    from a single writer thread (the execution loop main cycle).
    """

    def __init__(self) -> None:
        # symbol → next earnings datetime (UTC-aware)
        self._next_earnings: Dict[str, datetime] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update_earnings(self, symbol: str, next_earnings_dt: Optional[datetime]) -> None:
        """
        Record the next earnings date for a symbol.

        Args:
            symbol:            Trading symbol (normalized, no prefix).
            next_earnings_dt:  UTC-aware datetime of next earnings release,
                               or None to clear.
        """
        if next_earnings_dt is None:
            self._next_earnings.pop(symbol, None)
            return
        # Ensure UTC-aware
        if next_earnings_dt.tzinfo is None:
            next_earnings_dt = next_earnings_dt.replace(tzinfo=timezone.utc)
        self._next_earnings[symbol] = next_earnings_dt

    def get_sizing_mult(self, symbol: str) -> float:
        """
        Return a sizing multiplier [0.25, 1.0] based on proximity to earnings.

        Returns 1.0 for crypto/FX (no earnings), and 1.0 when no earnings data
        is available for an equity.
        """
        # Crypto and FX pass-through
        if self._is_non_equity(symbol):
            return _NORMAL_MULT

        earnings_dt = self._next_earnings.get(symbol) or self._next_earnings.get(
            self._strip_prefix(symbol)
        )
        if earnings_dt is None:
            return _NORMAL_MULT

        now = datetime.now(timezone.utc)
        hours_until = (earnings_dt - now).total_seconds() / 3600.0

        if hours_until < 0:
            # Earnings already passed — no dampening (PEAD window handled by EarningsCatalyst)
            return _NORMAL_MULT

        for threshold_h, mult in _PRE_EARNINGS_TIERS:
            if hours_until <= threshold_h:
                logger.debug(
                    "EarningsGate: %s is %.1fh from earnings → size×%.2f",
                    symbol, hours_until, mult,
                )
                return mult

        return _NORMAL_MULT

    def hours_until_earnings(self, symbol: str) -> Optional[float]:
        """Return hours until next earnings, or None if unknown / already passed."""
        earnings_dt = self._next_earnings.get(symbol) or self._next_earnings.get(
            self._strip_prefix(symbol)
        )
        if earnings_dt is None:
            return None
        now = datetime.now(timezone.utc)
        h = (earnings_dt - now).total_seconds() / 3600.0
        return h if h > 0 else None

    def get_report(self) -> dict:
        """Diagnostic dict for dashboard / walk-forward report."""
        now = datetime.now(timezone.utc)
        rows = {}
        for sym, dt in self._next_earnings.items():
            hours = (dt - now).total_seconds() / 3600.0
            mult = self.get_sizing_mult(sym)
            rows[sym] = {
                "next_earnings": dt.isoformat(),
                "hours_until": round(hours, 1),
                "sizing_mult": mult,
                "zone": self._zone(hours),
            }
        return {"symbols": rows, "count": len(rows)}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_non_equity(symbol: str) -> bool:
        s = symbol.upper()
        return (
            s.startswith("CRYPTO:")
            or s.startswith("FX:")
            or "/" in s  # crypto pair like BTC/USD
            or s in ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "HYG")
        )

    @staticmethod
    def _strip_prefix(symbol: str) -> str:
        for prefix in ("CRYPTO:", "FX:"):
            if symbol.upper().startswith(prefix):
                return symbol[len(prefix):]
        return symbol

    @staticmethod
    def _zone(hours_until: float) -> str:
        if hours_until <= 0:
            return "post"
        if hours_until <= 24:
            return "danger"
        if hours_until <= 48:
            return "warning"
        if hours_until <= 72:
            return "caution"
        return "clear"
