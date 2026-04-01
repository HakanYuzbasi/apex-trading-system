"""tests/test_missed_opportunity_wiring.py — MissedOpportunityTracker unit tests."""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from monitoring.missed_opportunity_tracker import (
    MissedOpportunity,
    MissedEarnerReport,
    MissedOpportunityTracker,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tracker(tmp_path: Path) -> MissedOpportunityTracker:
    return MissedOpportunityTracker(data_dir=tmp_path, session_type="test")


def _record(tracker: MissedOpportunityTracker, **kwargs) -> None:
    defaults = dict(
        symbol="AAPL",
        signal_strength=0.18,
        confidence=0.55,
        direction="long",
        regime="neutral",
        filter_reason="signal_threshold",
        entry_price=150.0,
        asset_class="equity",
    )
    defaults.update(kwargs)
    tracker.record_missed(**defaults)


# ── record_missed ──────────────────────────────────────────────────────────────

class TestRecordMissed:
    def test_adds_to_pending(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t)
        assert len(t._pending) == 1

    def test_symbol_preserved(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t, symbol="TSLA")
        assert t._pending[0].symbol == "TSLA"

    def test_filter_reason_preserved(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t, filter_reason="confidence_threshold")
        assert t._pending[0].filter_reason == "confidence_threshold"

    def test_asset_class_preserved(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t, asset_class="crypto")
        assert t._pending[0].asset_class == "crypto"

    def test_multiple_records(self, tmp_path):
        t = _tracker(tmp_path)
        for i in range(5):
            _record(t, symbol=f"SYM{i}")
        assert len(t._pending) == 5

    def test_pending_capped_at_max(self, tmp_path):
        t = _tracker(tmp_path)
        t.MAX_PENDING = 5
        for _ in range(10):
            _record(t)
        assert len(t._pending) <= 5

    def test_entry_price_stored(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t, entry_price=200.0)
        assert abs(t._pending[0].entry_price - 200.0) < 1e-9


# ── update_retrospective_prices ────────────────────────────────────────────────

class TestRetrospectivePricing:
    def test_no_prices_no_change(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t, symbol="AAPL")
        t.update_retrospective_prices({})
        assert len(t._pending) == 1  # still pending

    def test_price_before_5d_no_completion(self, tmp_path):
        """Symbol in prices but < 5 days elapsed — stays pending."""
        t = _tracker(tmp_path)
        _record(t, symbol="AAPL", entry_price=100.0)
        # Set signal_date to now (< 5 days elapsed)
        t._pending[0].signal_date = __import__("datetime").datetime.utcnow().isoformat()
        t.update_retrospective_prices({"AAPL": 110.0})
        # Should stay in pending (not yet 5 days)
        assert len(t._pending) == 1

    def test_20d_completion_moves_to_completed(self, tmp_path):
        """After 20+ days, opportunity moves to completed."""
        from datetime import datetime, timedelta
        t = _tracker(tmp_path)
        _record(t, symbol="AAPL", entry_price=100.0)
        t._pending[0].signal_date = (datetime.utcnow() - timedelta(days=25)).isoformat()
        t.update_retrospective_prices({"AAPL": 115.0})
        assert len(t._completed) == 1
        assert len(t._pending) == 0

    def test_completed_pnl_long_direction(self, tmp_path):
        from datetime import datetime, timedelta
        t = _tracker(tmp_path)
        _record(t, symbol="AAPL", entry_price=100.0, direction="long")
        t._pending[0].signal_date = (datetime.utcnow() - timedelta(days=25)).isoformat()
        t.update_retrospective_prices({"AAPL": 110.0})
        assert abs(t._completed[0].missed_pnl_20d_pct - 0.10) < 1e-9

    def test_completed_pnl_short_direction(self, tmp_path):
        from datetime import datetime, timedelta
        t = _tracker(tmp_path)
        _record(t, symbol="AAPL", entry_price=100.0, direction="short")
        t._pending[0].signal_date = (datetime.utcnow() - timedelta(days=25)).isoformat()
        t.update_retrospective_prices({"AAPL": 80.0})
        # short: entry/current - 1 = 100/80 - 1 = 0.25
        assert abs(t._completed[0].missed_pnl_20d_pct - 0.25) < 1e-9


# ── generate_report ────────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_empty_report_total_zero(self, tmp_path):
        t = _tracker(tmp_path)
        r = t.generate_report()
        assert r.total_missed == 0

    def test_report_counts_completed(self, tmp_path):
        from datetime import datetime, timedelta
        t = _tracker(tmp_path)
        for sym in ["AAPL", "MSFT", "GOOG"]:
            _record(t, symbol=sym, entry_price=100.0, filter_reason="signal_threshold")
            t._pending[-1].signal_date = (datetime.utcnow() - timedelta(days=25)).isoformat()
        t.update_retrospective_prices({"AAPL": 110.0, "MSFT": 108.0, "GOOG": 105.0})
        r = t.generate_report()
        assert r.total_missed == 3

    def test_by_filter_reason_counted(self, tmp_path):
        from datetime import datetime, timedelta
        t = _tracker(tmp_path)
        for _ in range(3):
            _record(t, entry_price=100.0, filter_reason="signal_threshold")
            t._pending[-1].signal_date = (datetime.utcnow() - timedelta(days=25)).isoformat()
        for _ in range(2):
            _record(t, entry_price=100.0, filter_reason="confidence_threshold")
            t._pending[-1].signal_date = (datetime.utcnow() - timedelta(days=25)).isoformat()
        t.update_retrospective_prices({"AAPL": 105.0})
        r = t.generate_report()
        assert r.by_filter_reason.get("signal_threshold", 0) == 3
        assert r.by_filter_reason.get("confidence_threshold", 0) == 2

    def test_top_missed_symbols_sorted(self, tmp_path):
        from datetime import datetime, timedelta
        t = _tracker(tmp_path)
        # TSLA has bigger missed pnl
        _record(t, symbol="TSLA", entry_price=100.0)
        _record(t, symbol="AAPL", entry_price=100.0)
        for o in t._pending:
            o.signal_date = (datetime.utcnow() - timedelta(days=25)).isoformat()
        t.update_retrospective_prices({"TSLA": 150.0, "AAPL": 105.0})
        r = t.generate_report()
        assert len(r.top_missed_symbols) > 0
        assert r.top_missed_symbols[0]["symbol"] == "TSLA"

    def test_report_generated_at_set(self, tmp_path):
        t = _tracker(tmp_path)
        r = t.generate_report()
        assert r.generated_at != ""


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_reload_recovers_pending(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t, symbol="AAPL")
        t._save()
        t2 = _tracker(tmp_path)
        assert len(t2._pending) == 1
        assert t2._pending[0].symbol == "AAPL"

    def test_reload_recovers_completed(self, tmp_path):
        from datetime import datetime, timedelta
        t = _tracker(tmp_path)
        _record(t, symbol="AAPL", entry_price=100.0)
        t._pending[0].signal_date = (datetime.utcnow() - timedelta(days=25)).isoformat()
        t.update_retrospective_prices({"AAPL": 110.0})
        t2 = _tracker(tmp_path)
        assert len(t2._completed) == 1

    def test_file_created_on_save(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t)
        t._save()
        assert t._output_file.exists()


# ── MissedOpportunity dataclass ───────────────────────────────────────────────

class TestDataclass:
    def test_asdict_has_all_fields(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t)
        d = asdict(t._pending[0])
        for key in (
            "symbol", "signal_date", "signal_strength", "confidence",
            "direction", "regime", "filter_reason", "entry_price",
            "asset_class", "session_type",
        ):
            assert key in d, f"missing key: {key}"

    def test_missed_pnl_fields_initially_none(self, tmp_path):
        t = _tracker(tmp_path)
        _record(t)
        o = t._pending[0]
        assert o.missed_pnl_5d_pct is None
        assert o.missed_pnl_10d_pct is None
        assert o.missed_pnl_20d_pct is None
