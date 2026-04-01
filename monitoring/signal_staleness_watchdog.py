"""
monitoring/signal_staleness_watchdog.py
────────────────────────────────────────
Signal Staleness Watchdog.

Tracks when each symbol's signal was last refreshed.  Raises a stale-signal
alert if a symbol has not been updated within the configured threshold.

Usage
─────
    watchdog = SignalStalenessWatchdog(stale_threshold_seconds=1800)
    watchdog.record_signal(symbol, signal_value)          # call each cycle

    alerts = watchdog.get_stale_alerts()                  # → List[StalenessAlert]
    report = watchdog.get_report()                        # → dict for API

The watchdog is PASSIVE — it does not raise exceptions or block processing.
Callers can check `get_stale_alerts()` and log/alert accordingly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STALE_THRESHOLD = 1800   # 30 minutes
_DEFAULT_CRITICAL_THRESHOLD = 3600  # 60 minutes


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class StalenessAlert:
    symbol: str
    last_seen_ts: float
    age_seconds: float
    severity: str   # "warning" | "critical"
    last_signal: Optional[float] = None

    @property
    def age_minutes(self) -> float:
        return self.age_seconds / 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol":         self.symbol,
            "age_minutes":    round(self.age_minutes, 1),
            "age_seconds":    round(self.age_seconds, 0),
            "severity":       self.severity,
            "last_signal":    self.last_signal,
            "last_seen_at":   datetime.fromtimestamp(
                                  self.last_seen_ts, tz=timezone.utc
                              ).isoformat(),
        }


# ── Main class ─────────────────────────────────────────────────────────────────

class SignalStalenessWatchdog:
    """
    Signal Staleness Watchdog.

    Parameters
    ──────────
    stale_threshold_seconds    : age that triggers a "warning" alert
    critical_threshold_seconds : age that triggers a "critical" alert
    tracked_symbols            : optional set of symbols to monitor;
                                 if None, all recorded symbols are tracked
    """

    def __init__(
        self,
        stale_threshold_seconds: float    = _DEFAULT_STALE_THRESHOLD,
        critical_threshold_seconds: float = _DEFAULT_CRITICAL_THRESHOLD,
        tracked_symbols: Optional[List[str]] = None,
    ) -> None:
        self.stale_threshold    = stale_threshold_seconds
        self.critical_threshold = critical_threshold_seconds
        self._tracked           = set(tracked_symbols) if tracked_symbols else None

        # symbol → (last_update_ts, last_signal_value)
        self._last_seen: Dict[str, float]           = {}
        self._last_signal: Dict[str, Optional[float]] = {}

    # ── Recording ──────────────────────────────────────────────────────────────

    def record_signal(self, symbol: str, signal_value: Optional[float] = None) -> None:
        """Record that a signal was received for `symbol` right now."""
        if self._tracked is not None and symbol not in self._tracked:
            return
        self._last_seen[symbol]   = time.time()
        self._last_signal[symbol] = signal_value

    def record_many(self, signals: Dict[str, Optional[float]]) -> None:
        """Bulk-record a {symbol: signal} dict."""
        ts = time.time()
        for sym, val in signals.items():
            if self._tracked is not None and sym not in self._tracked:
                continue
            self._last_seen[sym]   = ts
            self._last_signal[sym] = val

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_age(self, symbol: str) -> Optional[float]:
        """Return seconds since the last signal for `symbol`, or None if never seen."""
        ts = self._last_seen.get(symbol)
        if ts is None:
            return None
        return time.time() - ts

    def is_stale(self, symbol: str) -> bool:
        """True if the symbol has exceeded stale_threshold or has never been seen."""
        age = self.get_age(symbol)
        return age is None or age > self.stale_threshold

    def is_critical(self, symbol: str) -> bool:
        """True if the symbol has exceeded critical_threshold."""
        age = self.get_age(symbol)
        return age is None or age > self.critical_threshold

    def get_stale_alerts(self) -> List[StalenessAlert]:
        """Return alerts for all stale symbols, sorted by age descending."""
        now = time.time()
        alerts: List[StalenessAlert] = []
        symbols = (
            list(self._last_seen.keys())
            if self._tracked is None
            else list(self._tracked)
        )
        for sym in symbols:
            ts = self._last_seen.get(sym)
            if ts is None:
                # Tracked but never seen
                alerts.append(StalenessAlert(
                    symbol=sym,
                    last_seen_ts=0.0,
                    age_seconds=float("inf"),
                    severity="critical",
                    last_signal=None,
                ))
                continue
            age = now - ts
            if age > self.critical_threshold:
                alerts.append(StalenessAlert(
                    symbol=sym, last_seen_ts=ts, age_seconds=age,
                    severity="critical", last_signal=self._last_signal.get(sym),
                ))
            elif age > self.stale_threshold:
                alerts.append(StalenessAlert(
                    symbol=sym, last_seen_ts=ts, age_seconds=age,
                    severity="warning", last_signal=self._last_signal.get(sym),
                ))

        alerts.sort(key=lambda a: a.age_seconds, reverse=True)
        return alerts

    def get_report(self) -> Dict[str, Any]:
        """JSON-serialisable status report."""
        now = time.time()
        alerts = self.get_stale_alerts()
        total = len(self._last_seen)
        stale_count    = sum(1 for a in alerts if a.severity == "warning")
        critical_count = sum(1 for a in alerts if a.severity == "critical")

        freshest = min(
            (now - ts for ts in self._last_seen.values()), default=None
        )
        stalest  = max(
            (now - ts for ts in self._last_seen.values()), default=None
        )

        return {
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "tracked_symbols": total,
            "stale_count":     stale_count,
            "critical_count":  critical_count,
            "stale_threshold_minutes":    round(self.stale_threshold / 60, 1),
            "critical_threshold_minutes": round(self.critical_threshold / 60, 1),
            "freshest_age_seconds": round(freshest, 1) if freshest is not None else None,
            "stalest_age_seconds":  round(stalest, 1)  if stalest  is not None else None,
            "alerts": [a.to_dict() for a in alerts],
        }

    def reset(self, symbol: str) -> None:
        """Mark a symbol as freshly updated (without a signal value)."""
        self._last_seen[symbol]   = time.time()
        self._last_signal[symbol] = self._last_signal.get(symbol)

    def clear(self, symbol: str) -> None:
        """Remove a symbol from tracking entirely."""
        self._last_seen.pop(symbol, None)
        self._last_signal.pop(symbol, None)
