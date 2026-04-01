"""tests/test_signal_staleness_watchdog.py — Signal Staleness Watchdog tests."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from monitoring.signal_staleness_watchdog import (
    SignalStalenessWatchdog,
    StalenessAlert,
)


# ── StalenessAlert ────────────────────────────────────────────────────────────

class TestStalenessAlert:
    def test_age_minutes(self):
        a = StalenessAlert("AAPL", 0.0, 3600.0, "critical")
        assert abs(a.age_minutes - 60.0) < 1e-9

    def test_to_dict_keys(self):
        a = StalenessAlert("AAPL", time.time() - 100, 100.0, "warning", 0.15)
        d = a.to_dict()
        assert d["symbol"] == "AAPL"
        assert "age_minutes" in d
        assert d["severity"] == "warning"
        assert d["last_signal"] == 0.15
        assert "last_seen_at" in d


# ── record_signal ─────────────────────────────────────────────────────────────

class TestRecordSignal:
    def test_records_symbol(self):
        wd = SignalStalenessWatchdog()
        wd.record_signal("AAPL", 0.15)
        assert "AAPL" in wd._last_seen
        assert wd._last_signal["AAPL"] == 0.15

    def test_updates_timestamp(self):
        wd = SignalStalenessWatchdog()
        wd.record_signal("AAPL", 0.10)
        t1 = wd._last_seen["AAPL"]
        time.sleep(0.01)
        wd.record_signal("AAPL", 0.12)
        t2 = wd._last_seen["AAPL"]
        assert t2 > t1

    def test_tracked_symbols_filter(self):
        wd = SignalStalenessWatchdog(tracked_symbols=["AAPL"])
        wd.record_signal("TSLA", 0.10)
        assert "TSLA" not in wd._last_seen

    def test_tracked_allows_listed(self):
        wd = SignalStalenessWatchdog(tracked_symbols=["AAPL", "TSLA"])
        wd.record_signal("AAPL", 0.10)
        assert "AAPL" in wd._last_seen

    def test_record_many(self):
        wd = SignalStalenessWatchdog()
        wd.record_many({"AAPL": 0.10, "TSLA": 0.15, "MSFT": 0.12})
        assert len(wd._last_seen) == 3


# ── get_age ───────────────────────────────────────────────────────────────────

class TestGetAge:
    def test_none_when_never_seen(self):
        wd = SignalStalenessWatchdog()
        assert wd.get_age("AAPL") is None

    def test_small_age_after_record(self):
        wd = SignalStalenessWatchdog()
        wd.record_signal("AAPL")
        age = wd.get_age("AAPL")
        assert age is not None
        assert age < 1.0


# ── is_stale / is_critical ────────────────────────────────────────────────────

class TestIsStale:
    def test_fresh_signal_not_stale(self):
        wd = SignalStalenessWatchdog(stale_threshold_seconds=60)
        wd.record_signal("AAPL")
        assert not wd.is_stale("AAPL")

    def test_old_signal_is_stale(self):
        wd = SignalStalenessWatchdog(stale_threshold_seconds=30)
        wd._last_seen["AAPL"] = time.time() - 60
        assert wd.is_stale("AAPL")

    def test_never_seen_is_stale(self):
        wd = SignalStalenessWatchdog()
        assert wd.is_stale("UNKNOWN")

    def test_critical_threshold(self):
        wd = SignalStalenessWatchdog(
            stale_threshold_seconds=30,
            critical_threshold_seconds=60,
        )
        wd._last_seen["AAPL"] = time.time() - 90
        assert wd.is_critical("AAPL")

    def test_below_critical_threshold(self):
        wd = SignalStalenessWatchdog(
            stale_threshold_seconds=30,
            critical_threshold_seconds=60,
        )
        wd._last_seen["AAPL"] = time.time() - 45
        assert not wd.is_critical("AAPL")


# ── get_stale_alerts ──────────────────────────────────────────────────────────

class TestGetStaleAlerts:
    def test_no_alerts_when_fresh(self):
        wd = SignalStalenessWatchdog(stale_threshold_seconds=60)
        wd.record_signal("AAPL")
        wd.record_signal("TSLA")
        assert wd.get_stale_alerts() == []

    def test_warning_alert(self):
        wd = SignalStalenessWatchdog(
            stale_threshold_seconds=30,
            critical_threshold_seconds=120,
        )
        wd._last_seen["AAPL"] = time.time() - 60
        alerts = wd.get_stale_alerts()
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].symbol == "AAPL"

    def test_critical_alert(self):
        wd = SignalStalenessWatchdog(
            stale_threshold_seconds=30,
            critical_threshold_seconds=60,
        )
        wd._last_seen["AAPL"] = time.time() - 90
        alerts = wd.get_stale_alerts()
        assert any(a.severity == "critical" for a in alerts)

    def test_sorted_by_age_descending(self):
        wd = SignalStalenessWatchdog(stale_threshold_seconds=10)
        wd._last_seen["A"] = time.time() - 100
        wd._last_seen["B"] = time.time() - 50
        alerts = wd.get_stale_alerts()
        assert alerts[0].symbol == "A"
        assert alerts[1].symbol == "B"

    def test_tracked_never_seen_is_critical(self):
        wd = SignalStalenessWatchdog(tracked_symbols=["AAPL", "TSLA"])
        # TSLA never recorded
        wd.record_signal("AAPL")
        alerts = wd.get_stale_alerts()
        tsla_alert = next((a for a in alerts if a.symbol == "TSLA"), None)
        assert tsla_alert is not None
        assert tsla_alert.severity == "critical"


# ── get_report ────────────────────────────────────────────────────────────────

class TestGetReport:
    def test_empty_report(self):
        wd = SignalStalenessWatchdog()
        r = wd.get_report()
        assert r["tracked_symbols"] == 0
        assert r["stale_count"] == 0
        assert r["critical_count"] == 0
        assert r["alerts"] == []

    def test_report_keys(self):
        wd = SignalStalenessWatchdog()
        wd.record_signal("AAPL")
        r = wd.get_report()
        assert "generated_at" in r
        assert "stale_threshold_minutes" in r
        assert "critical_threshold_minutes" in r
        assert "freshest_age_seconds" in r
        assert "stalest_age_seconds" in r

    def test_report_counts_stale(self):
        wd = SignalStalenessWatchdog(stale_threshold_seconds=10, critical_threshold_seconds=100)
        wd._last_seen["A"] = time.time() - 50
        wd._last_seen["B"] = time.time() - 1
        r = wd.get_report()
        assert r["stale_count"] == 1
        assert r["tracked_symbols"] == 2


# ── reset / clear ─────────────────────────────────────────────────────────────

class TestResetClear:
    def test_reset_refreshes_ts(self):
        wd = SignalStalenessWatchdog(stale_threshold_seconds=10)
        wd._last_seen["AAPL"] = time.time() - 100
        assert wd.is_stale("AAPL")
        wd.reset("AAPL")
        assert not wd.is_stale("AAPL")

    def test_clear_removes_symbol(self):
        wd = SignalStalenessWatchdog()
        wd.record_signal("AAPL")
        wd.clear("AAPL")
        assert "AAPL" not in wd._last_seen
        assert wd.get_age("AAPL") is None
