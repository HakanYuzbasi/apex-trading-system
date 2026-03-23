"""Tests for RegimeTransitionAlerter."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from monitoring.regime_alert import (
    RegimeTransitionAlerter,
    RegimeAlert,
    _CAUTION_PROB,
    _WARN_PROB,
    _CRITICAL_PROB,
    _CRITICAL_MULT,
)


def _alerter(**kw) -> RegimeTransitionAlerter:
    defaults = dict(cooldown_seconds=0.0)   # no cooldown in tests
    defaults.update(kw)
    return RegimeTransitionAlerter(**defaults)


class _Forecast:
    """Minimal duck-typed forecast object."""
    def __init__(self, prob: float, mult: float = 1.0, signal: str = "clear"):
        self.prob = prob
        self.mult = mult
        self.signal = signal
        self.transition_prob = prob
        self.size_multiplier = mult
        self.features = {"vix_z": 1.0}


# ── Default / empty state ────────────────────────────────────────────────────

class TestDefaultState:

    def test_current_severity_is_clear(self):
        a = _alerter()
        assert a.current_severity == "clear"

    def test_alert_history_empty(self):
        a = _alerter()
        assert a.alert_history == []

    def test_report_is_dict(self):
        a = _alerter()
        assert isinstance(a.get_report(), dict)

    def test_report_keys(self):
        a = _alerter()
        r = a.get_report()
        for k in ("current_severity", "total_alerts", "recent_alerts", "cooldown_seconds", "thresholds"):
            assert k in r

    def test_clear_stays_silent(self):
        a = _alerter()
        result = a.check(_Forecast(0.10))
        assert result is None


# ── Classification ───────────────────────────────────────────────────────────

class TestClassification:

    def test_below_caution_is_clear(self):
        a = _alerter()
        a._current_severity = "clear"
        assert a._classify(0.10, 1.0) == "clear"

    def test_caution_threshold(self):
        a = _alerter()
        assert a._classify(_CAUTION_PROB, 1.0) == "caution"

    def test_warn_threshold(self):
        a = _alerter()
        assert a._classify(_WARN_PROB, 1.0) == "warning"

    def test_critical_prob_threshold(self):
        a = _alerter()
        assert a._classify(_CRITICAL_PROB, 1.0) == "critical"

    def test_critical_mult_threshold(self):
        a = _alerter()
        assert a._classify(0.20, _CRITICAL_MULT) == "critical"

    def test_low_mult_overrides_prob(self):
        a = _alerter()
        # Even with low prob, very low mult → critical
        assert a._classify(0.10, 0.40) == "critical"


# ── Alert firing ─────────────────────────────────────────────────────────────

class TestAlertFiring:

    def test_escalation_fires_alert(self):
        a = _alerter()
        alert = a.check(_Forecast(prob=_CAUTION_PROB + 0.01))
        assert alert is not None
        assert isinstance(alert, RegimeAlert)

    def test_returns_none_on_clear_to_clear(self):
        a = _alerter()
        assert a.check(_Forecast(0.05)) is None

    def test_alert_has_correct_severity(self):
        a = _alerter()
        alert = a.check(_Forecast(_WARN_PROB + 0.01))
        assert alert is not None
        assert alert.severity == "warning"

    def test_escalation_is_escalation_property(self):
        a = _alerter()
        alert = a.check(_Forecast(_WARN_PROB + 0.01))
        assert alert is not None
        assert alert.is_escalation is True

    def test_deescalation_fires_alert(self):
        a = _alerter()
        # First: escalate to warning
        a.check(_Forecast(_WARN_PROB + 0.01))
        # Then: de-escalate to caution
        alert = a.check(_Forecast(_CAUTION_PROB + 0.01, mult=0.90))
        assert alert is not None
        assert alert.is_de_escalation is True

    def test_alert_history_grows(self):
        a = _alerter()
        a.check(_Forecast(_CAUTION_PROB + 0.01))
        a.check(_Forecast(_WARN_PROB + 0.01))
        assert len(a.alert_history) == 2

    def test_alert_severity_tracked(self):
        a = _alerter()
        a.check(_Forecast(_WARN_PROB + 0.01))
        assert a.current_severity == "warning"

    def test_clear_after_alert_fires_de_escalation(self):
        a = _alerter()
        a.check(_Forecast(_WARN_PROB + 0.01))
        alert = a.check(_Forecast(0.05))
        assert alert is not None
        assert alert.severity == "clear"

    def test_alert_contains_probability(self):
        a = _alerter()
        p = _WARN_PROB + 0.05
        alert = a.check(_Forecast(p))
        assert alert is not None
        assert alert.probability == pytest.approx(p, abs=0.001)

    def test_alert_contains_features(self):
        a = _alerter()
        alert = a.check(_Forecast(_WARN_PROB + 0.01))
        assert isinstance(alert.features, dict)


# ── Cooldown ─────────────────────────────────────────────────────────────────

class TestCooldown:

    def test_same_severity_respects_cooldown(self):
        a = _alerter(cooldown_seconds=60.0)
        # First escalation
        a.check(_Forecast(_CAUTION_PROB + 0.01))
        a._current_severity = "caution"
        # Second check at same severity — should be suppressed
        alert = a.check(_Forecast(_CAUTION_PROB + 0.01))
        assert alert is None

    def test_escalation_always_fires_despite_cooldown(self):
        a = _alerter(cooldown_seconds=999.0)
        # Escalate to caution
        a.check(_Forecast(_CAUTION_PROB + 0.01))
        # Further escalate to warning — must fire even within cooldown
        alert = a.check(_Forecast(_WARN_PROB + 0.01))
        assert alert is not None
        assert alert.severity == "warning"

    def test_cooldown_zero_always_fires(self):
        a = _alerter(cooldown_seconds=0.0)
        a.check(_Forecast(_CAUTION_PROB + 0.01))
        alert = a.check(_Forecast(_CAUTION_PROB + 0.01))
        assert alert is not None


# ── check_values helper ───────────────────────────────────────────────────────

class TestCheckValues:

    def test_check_values_equivalent_to_check(self):
        a1 = _alerter()
        a2 = _alerter()
        p = _WARN_PROB + 0.01
        r1 = a1.check(_Forecast(p, 0.80))
        r2 = a2.check_values(p, 0.80)
        assert (r1 is None) == (r2 is None)
        if r1 is not None and r2 is not None:
            assert r1.severity == r2.severity

    def test_check_values_with_features(self):
        a = _alerter()
        alert = a.check_values(_WARN_PROB + 0.01, 0.90, features={"vix_z": 2.5})
        assert alert is not None
        assert alert.features.get("vix_z") == pytest.approx(2.5)


# ── to_dict / message ────────────────────────────────────────────────────────

class TestAlertFormatting:

    def test_to_dict_has_severity(self):
        a = _alerter()
        alert = a.check(_Forecast(_WARN_PROB + 0.01))
        assert alert is not None
        d = alert.to_dict()
        assert "severity" in d
        assert "message" in d
        assert "timestamp_iso" in d

    def test_message_contains_prob(self):
        a = _alerter()
        alert = a.check(_Forecast(_WARN_PROB + 0.01))
        assert alert is not None
        assert "%" in alert.message

    def test_critical_message_says_critical(self):
        a = _alerter()
        alert = a.check(_Forecast(_CRITICAL_PROB + 0.01))
        assert alert is not None
        assert "CRITICAL" in alert.message.upper() or "critical" in alert.message.lower()


# ── get_recent_alerts ─────────────────────────────────────────────────────────

class TestRecentAlerts:

    def test_returns_list(self):
        a = _alerter()
        assert isinstance(a.get_recent_alerts(), list)

    def test_recent_alerts_as_dicts(self):
        a = _alerter()
        a.check(_Forecast(_CAUTION_PROB + 0.01))
        recents = a.get_recent_alerts(1)
        assert len(recents) == 1
        assert isinstance(recents[0], dict)


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:

    def test_alert_written_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = RegimeTransitionAlerter(cooldown_seconds=0.0, data_dir=Path(tmp))
            a.check(_Forecast(_WARN_PROB + 0.01))
            p = Path(tmp) / "regime_alerts.jsonl"
            assert p.exists()
            assert p.stat().st_size > 0

    def test_state_written_after_alert(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = RegimeTransitionAlerter(cooldown_seconds=0.0, data_dir=Path(tmp))
            a.check(_Forecast(_CAUTION_PROB + 0.01))
            assert (Path(tmp) / "regime_alerter_state.json").exists()

    def test_severity_restored_on_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            a1 = RegimeTransitionAlerter(cooldown_seconds=0.0, data_dir=Path(tmp))
            a1.check(_Forecast(_WARN_PROB + 0.01))
            saved = a1.current_severity

            a2 = RegimeTransitionAlerter(cooldown_seconds=0.0, data_dir=Path(tmp))
            assert a2.current_severity == saved

    def test_alert_history_restored_on_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            a1 = RegimeTransitionAlerter(cooldown_seconds=0.0, data_dir=Path(tmp))
            a1.check(_Forecast(_WARN_PROB + 0.01))
            n = len(a1.alert_history)

            a2 = RegimeTransitionAlerter(cooldown_seconds=0.0, data_dir=Path(tmp))
            assert len(a2.alert_history) == n

    def test_no_dir_no_crash(self):
        a = RegimeTransitionAlerter(data_dir=None, cooldown_seconds=0.0)
        a.check(_Forecast(_WARN_PROB + 0.01))
        assert len(a.alert_history) == 1
