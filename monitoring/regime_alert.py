"""Live Regime Transition Alert.

Watches the RegimeTransitionForecaster's output for severity escalations
and emits structured alert events when thresholds are crossed.

Alert severity levels (in escalating order):
    clear    → no concern
    caution  → moderate transition probability (>= CAUTION_PROB)
    warning  → high transition probability (>= WARN_PROB)
    critical → probability >= CRITICAL_PROB and mult <= CRITICAL_MULT

Cooldown logic prevents alert spam: once an alert fires at a given level,
the same level will not fire again until `cooldown_seconds` has elapsed.

Usage::

    alerter = RegimeTransitionAlerter()
    alert = alerter.check(forecaster.get_forecast())
    if alert:
        log_alert(alert)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_CAUTION_PROB: float = 0.40
_WARN_PROB: float = 0.60
_CRITICAL_PROB: float = 0.78
_CRITICAL_MULT: float = 0.55   # mult this low or lower → critical
_DEFAULT_COOLDOWN: float = 900.0  # 15 minutes

_SEVERITY_ORDER = {"clear": 0, "caution": 1, "warning": 2, "critical": 3}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RegimeAlert:
    severity: str            # "caution" | "warning" | "critical"
    previous_severity: str   # what severity was before this event
    probability: float
    size_multiplier: float
    message: str
    timestamp: float = field(default_factory=time.time)
    features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_iso"] = _ts_iso(self.timestamp)
        return d

    @property
    def is_escalation(self) -> bool:
        return _SEVERITY_ORDER.get(self.severity, 0) > _SEVERITY_ORDER.get(self.previous_severity, 0)

    @property
    def is_de_escalation(self) -> bool:
        return _SEVERITY_ORDER.get(self.severity, 0) < _SEVERITY_ORDER.get(self.previous_severity, 0)


# ── Main class ────────────────────────────────────────────────────────────────

class RegimeTransitionAlerter:
    """Wraps the RegimeTransitionForecaster and fires structured alerts.

    Parameters
    ----------
    cooldown_seconds : float
        Minimum gap between repeat alerts at the *same* severity level.
        Escalations (severity increases) always fire immediately.
    data_dir : Path | None
        If given, alert history is persisted to ``regime_alerts.jsonl``.
    max_history : int
        Maximum number of alerts kept in memory.
    """

    def __init__(
        self,
        cooldown_seconds: float = _DEFAULT_COOLDOWN,
        data_dir: Optional[Path] = None,
        max_history: int = 100,
    ) -> None:
        self._cooldown = cooldown_seconds
        self._data_dir = data_dir
        self._max_history = max_history

        self._current_severity: str = "clear"
        self._last_alert_ts: dict[str, float] = {}   # severity → last fired ts
        self._alert_history: List[RegimeAlert] = []

        if data_dir is not None:
            self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, forecast: object) -> Optional[RegimeAlert]:
        """Evaluate a RegimeTransitionForecast and fire an alert if warranted.

        Args:
            forecast: A ``RegimeTransitionForecast`` instance (duck-typed —
                      needs ``.prob``, ``.mult``, ``.signal``, and optionally
                      ``.features``).

        Returns:
            A ``RegimeAlert`` if an alert fires, else ``None``.
        """
        prob = float(getattr(forecast, "prob", 0.0))
        mult = float(getattr(forecast, "mult", 1.0))
        features = dict(getattr(forecast, "features", {}) or {})

        new_severity = self._classify(prob, mult)
        alert = self._maybe_fire(new_severity, prob, mult, features)
        self._current_severity = new_severity
        return alert

    def check_values(
        self,
        probability: float,
        size_multiplier: float,
        features: Optional[dict] = None,
    ) -> Optional[RegimeAlert]:
        """Check using raw values instead of a forecast object."""

        class _FakeForecast:
            def __init__(self, p, m, f):
                self.prob = p
                self.mult = m
                self.features = f or {}

        return self.check(_FakeForecast(probability, size_multiplier, features or {}))

    @property
    def current_severity(self) -> str:
        return self._current_severity

    @property
    def alert_history(self) -> List[RegimeAlert]:
        return list(self._alert_history)

    def get_recent_alerts(self, n: int = 10) -> List[dict]:
        """Return the last n alerts as dicts, newest first."""
        return [a.to_dict() for a in reversed(self._alert_history[-n:])]

    def get_report(self) -> dict:
        return {
            "current_severity": self._current_severity,
            "total_alerts": len(self._alert_history),
            "recent_alerts": self.get_recent_alerts(5),
            "cooldown_seconds": self._cooldown,
            "thresholds": {
                "caution_prob": _CAUTION_PROB,
                "warn_prob": _WARN_PROB,
                "critical_prob": _CRITICAL_PROB,
                "critical_mult": _CRITICAL_MULT,
            },
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _classify(self, prob: float, mult: float) -> str:
        if prob >= _CRITICAL_PROB or mult <= _CRITICAL_MULT:
            return "critical"
        if prob >= _WARN_PROB:
            return "warning"
        if prob >= _CAUTION_PROB:
            return "caution"
        return "clear"

    def _maybe_fire(
        self, new_severity: str, prob: float, mult: float, features: dict
    ) -> Optional[RegimeAlert]:
        prev = self._current_severity
        now = time.time()

        if new_severity == "clear" and prev == "clear":
            return None

        is_escalation = _SEVERITY_ORDER.get(new_severity, 0) > _SEVERITY_ORDER.get(prev, 0)
        is_deescalation = _SEVERITY_ORDER.get(new_severity, 0) < _SEVERITY_ORDER.get(prev, 0)

        # Always fire on severity change
        if is_escalation or is_deescalation:
            return self._fire(new_severity, prev, prob, mult, features, now)

        # Same severity — respect cooldown
        last = self._last_alert_ts.get(new_severity, 0.0)
        if new_severity != "clear" and (now - last) >= self._cooldown:
            return self._fire(new_severity, prev, prob, mult, features, now)

        return None

    def _fire(
        self,
        severity: str,
        prev: str,
        prob: float,
        mult: float,
        features: dict,
        now: float,
    ) -> RegimeAlert:
        msg = self._build_message(severity, prob, mult)
        alert = RegimeAlert(
            severity=severity,
            previous_severity=prev,
            probability=round(prob, 4),
            size_multiplier=round(mult, 4),
            message=msg,
            timestamp=now,
            features=features,
        )
        self._current_severity = severity  # update before persisting
        self._alert_history.append(alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history :]
        self._last_alert_ts[severity] = now

        if self._data_dir is not None:
            self._append_alert(alert)

        return alert

    @staticmethod
    def _build_message(severity: str, prob: float, mult: float) -> str:
        pct = f"{prob * 100:.0f}%"
        if severity == "critical":
            return f"CRITICAL regime shift detected — prob={pct}, size_mult={mult:.2f}"
        if severity == "warning":
            return f"WARNING regime transition likely — prob={pct}, size_mult={mult:.2f}"
        if severity == "caution":
            return f"Caution: elevated transition probability={pct}"
        return f"Regime alert cleared (prob={pct})"

    # ── Persistence ───────────────────────────────────────────────────────────

    def _alert_path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / "regime_alerts.jsonl"

    def _state_path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / "regime_alerter_state.json"

    def _append_alert(self, alert: RegimeAlert) -> None:
        try:
            self._alert_path().parent.mkdir(parents=True, exist_ok=True)
            with open(self._alert_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
            # Save lightweight state
            state = {
                "current_severity": self._current_severity,
                "last_alert_ts": self._last_alert_ts,
            }
            tmp = self._state_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(self._state_path())
        except Exception:
            pass

    def _load(self) -> None:
        try:
            sp = self._state_path()
            if sp.exists():
                raw = json.loads(sp.read_text(encoding="utf-8"))
                self._current_severity = raw.get("current_severity", "clear")
                self._last_alert_ts = {k: float(v) for k, v in raw.get("last_alert_ts", {}).items()}
            # Load recent alert history (last max_history lines)
            ap = self._alert_path()
            if ap.exists():
                lines = ap.read_text(encoding="utf-8").strip().splitlines()
                for line in lines[-self._max_history:]:
                    try:
                        d = json.loads(line)
                        self._alert_history.append(RegimeAlert(
                            severity=d["severity"],
                            previous_severity=d["previous_severity"],
                            probability=d["probability"],
                            size_multiplier=d["size_multiplier"],
                            message=d["message"],
                            timestamp=d["timestamp"],
                            features=d.get("features", {}),
                        ))
                    except Exception:
                        pass
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_iso(ts: float) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
