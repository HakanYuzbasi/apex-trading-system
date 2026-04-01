"""
Tests for GET /ops/order-rejections endpoint logic.

We test the JSONL reading, filtering, sorting, and aggregation that the
endpoint performs — without spinning up the FastAPI server.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers that replicate the endpoint logic in a test-friendly form
# ---------------------------------------------------------------------------

def _write_records(gateway_dir: Path, date_str: str, records: list[dict]) -> None:
    """Write audit records to the pretrade_gateway JSONL for a given date."""
    gateway_dir.mkdir(parents=True, exist_ok=True)
    fpath = gateway_dir / f"pretrade_gateway_{date_str}.jsonl"
    with open(fpath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_record(
    *,
    event_id: str = "ptg-abc",
    symbol: str = "AAPL",
    asset_class: str = "equity",
    side: str = "BUY",
    quantity: int = 100,
    price: float = 180.0,
    allowed: bool = False,
    reason_code: str = "max_order_notional",
    message: str = "Exceeds notional limit",
    metadata: dict | None = None,
    actor: str = "strategy_loop",
    ts_offset_minutes: int = 0,
) -> dict:
    ts = (datetime.utcnow() - timedelta(minutes=ts_offset_minutes)).isoformat()
    return {
        "event_id": event_id,
        "timestamp": ts,
        "symbol": symbol,
        "asset_class": asset_class,
        "side": side,
        "quantity": quantity,
        "price": price,
        "allowed": allowed,
        "reason_code": reason_code,
        "message": message,
        "metadata": metadata or {},
        "actor": actor,
        "prev_hash": None,
        "hash": "dummyhash",
    }


def _load_rejections(
    gateway_dir: Path,
    *,
    limit: int = 50,
    reason_code: str = "",
) -> dict:
    """Pure-Python reimplementation of the endpoint's read logic for testing."""
    from collections import Counter

    today = datetime.utcnow()
    dates = [today.strftime("%Y%m%d"), (today - timedelta(days=1)).strftime("%Y%m%d")]

    records = []
    total_scanned = 0
    for date_str in dates:
        fpath = gateway_dir / f"pretrade_gateway_{date_str}.jsonl"
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_scanned += 1
                if rec.get("allowed", True):
                    continue
                if reason_code and rec.get("reason_code") != reason_code:
                    continue
                records.append({
                    "event_id": rec.get("event_id", ""),
                    "timestamp": rec.get("timestamp", ""),
                    "symbol": rec.get("symbol", ""),
                    "asset_class": rec.get("asset_class", ""),
                    "side": rec.get("side", ""),
                    "quantity": rec.get("quantity", 0),
                    "price": rec.get("price", 0.0),
                    "reason_code": rec.get("reason_code", ""),
                    "message": rec.get("message", ""),
                    "metadata": rec.get("metadata", {}),
                    "actor": rec.get("actor", ""),
                })

    records.sort(key=lambda r: r["timestamp"], reverse=True)
    records = records[:limit]
    reason_counts = Counter(r["reason_code"] for r in records)
    return {
        "available": True,
        "total_scanned": total_scanned,
        "total_rejected": len(records),
        "reason_breakdown": dict(reason_counts),
        "rejections": records,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY = datetime.utcnow().strftime("%Y%m%d")
YESTERDAY = (datetime.utcnow() - timedelta(days=1)).strftime("%Y%m%d")


@pytest.fixture
def gw_dir(tmp_path):
    return tmp_path / "audit" / "pretrade_gateway"


# ---------------------------------------------------------------------------
# Basic operation
# ---------------------------------------------------------------------------

class TestEmptyAudit:
    def test_empty_dir_returns_empty_list(self, gw_dir):
        result = _load_rejections(gw_dir)
        assert result["rejections"] == []
        assert result["total_rejected"] == 0

    def test_no_files_total_scanned_zero(self, gw_dir):
        result = _load_rejections(gw_dir)
        assert result["total_scanned"] == 0

    def test_available_true(self, gw_dir):
        result = _load_rejections(gw_dir)
        assert result["available"] is True


class TestBasicRejections:
    def test_rejection_appears_in_result(self, gw_dir):
        _write_records(gw_dir, TODAY, [_make_record(allowed=False)])
        result = _load_rejections(gw_dir)
        assert result["total_rejected"] == 1

    def test_allowed_orders_excluded(self, gw_dir):
        _write_records(gw_dir, TODAY, [
            _make_record(allowed=True, reason_code="allowed"),
            _make_record(allowed=False, reason_code="max_order_notional"),
        ])
        result = _load_rejections(gw_dir)
        assert result["total_rejected"] == 1
        assert result["total_scanned"] == 2

    def test_rejection_record_has_required_fields(self, gw_dir):
        _write_records(gw_dir, TODAY, [_make_record(allowed=False, symbol="TSLA", reason_code="price_band")])
        rec = _load_rejections(gw_dir)["rejections"][0]
        for field in ["event_id", "timestamp", "symbol", "side", "quantity", "price", "reason_code", "message"]:
            assert field in rec

    def test_symbol_preserved(self, gw_dir):
        _write_records(gw_dir, TODAY, [_make_record(allowed=False, symbol="NVDA")])
        rec = _load_rejections(gw_dir)["rejections"][0]
        assert rec["symbol"] == "NVDA"

    def test_reason_code_preserved(self, gw_dir):
        _write_records(gw_dir, TODAY, [_make_record(allowed=False, reason_code="gross_exposure")])
        rec = _load_rejections(gw_dir)["rejections"][0]
        assert rec["reason_code"] == "gross_exposure"


class TestSortingAndLimit:
    def test_most_recent_first(self, gw_dir):
        records = [
            _make_record(event_id="old", allowed=False, ts_offset_minutes=60),
            _make_record(event_id="new", allowed=False, ts_offset_minutes=0),
        ]
        _write_records(gw_dir, TODAY, records)
        result = _load_rejections(gw_dir)
        assert result["rejections"][0]["event_id"] == "new"
        assert result["rejections"][1]["event_id"] == "old"

    def test_limit_respected(self, gw_dir):
        records = [_make_record(event_id=f"e{i}", allowed=False) for i in range(10)]
        _write_records(gw_dir, TODAY, records)
        result = _load_rejections(gw_dir, limit=3)
        assert len(result["rejections"]) == 3

    def test_limit_does_not_exceed_available(self, gw_dir):
        _write_records(gw_dir, TODAY, [_make_record(allowed=False)])
        result = _load_rejections(gw_dir, limit=50)
        assert len(result["rejections"]) == 1


class TestReasonFilter:
    def test_filter_returns_only_matching(self, gw_dir):
        _write_records(gw_dir, TODAY, [
            _make_record(allowed=False, reason_code="price_band"),
            _make_record(allowed=False, reason_code="max_order_notional"),
            _make_record(allowed=False, reason_code="price_band"),
        ])
        result = _load_rejections(gw_dir, reason_code="price_band")
        assert result["total_rejected"] == 2
        assert all(r["reason_code"] == "price_band" for r in result["rejections"])

    def test_empty_filter_returns_all(self, gw_dir):
        _write_records(gw_dir, TODAY, [
            _make_record(allowed=False, reason_code="price_band"),
            _make_record(allowed=False, reason_code="adv_participation"),
        ])
        result = _load_rejections(gw_dir, reason_code="")
        assert result["total_rejected"] == 2

    def test_unknown_filter_returns_empty(self, gw_dir):
        _write_records(gw_dir, TODAY, [_make_record(allowed=False, reason_code="price_band")])
        result = _load_rejections(gw_dir, reason_code="nonexistent_code")
        assert result["total_rejected"] == 0


class TestReasonBreakdown:
    def test_breakdown_counts_correct(self, gw_dir):
        _write_records(gw_dir, TODAY, [
            _make_record(allowed=False, reason_code="max_order_notional"),
            _make_record(allowed=False, reason_code="max_order_notional"),
            _make_record(allowed=False, reason_code="price_band"),
        ])
        bd = _load_rejections(gw_dir)["reason_breakdown"]
        assert bd["max_order_notional"] == 2
        assert bd["price_band"] == 1

    def test_breakdown_empty_when_no_rejections(self, gw_dir):
        _write_records(gw_dir, TODAY, [_make_record(allowed=True, reason_code="allowed")])
        bd = _load_rejections(gw_dir)["reason_breakdown"]
        assert bd == {}


class TestYesterdayFile:
    def test_reads_yesterday_file(self, gw_dir):
        _write_records(gw_dir, YESTERDAY, [_make_record(allowed=False, symbol="MSFT")])
        result = _load_rejections(gw_dir)
        assert result["total_rejected"] == 1
        assert result["rejections"][0]["symbol"] == "MSFT"

    def test_combines_today_and_yesterday(self, gw_dir):
        _write_records(gw_dir, TODAY, [_make_record(allowed=False, symbol="AAPL")])
        _write_records(gw_dir, YESTERDAY, [_make_record(allowed=False, symbol="TSLA")])
        result = _load_rejections(gw_dir)
        symbols = {r["symbol"] for r in result["rejections"]}
        assert "AAPL" in symbols
        assert "TSLA" in symbols


class TestMalformedLines:
    def test_skips_malformed_json_lines(self, gw_dir):
        gw_dir.mkdir(parents=True, exist_ok=True)
        fpath = gw_dir / f"pretrade_gateway_{TODAY}.jsonl"
        with open(fpath, "w") as f:
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(_make_record(allowed=False)) + "\n")
        result = _load_rejections(gw_dir)
        assert result["total_rejected"] == 1

    def test_skips_blank_lines(self, gw_dir):
        gw_dir.mkdir(parents=True, exist_ok=True)
        fpath = gw_dir / f"pretrade_gateway_{TODAY}.jsonl"
        with open(fpath, "w") as f:
            f.write("\n\n")
            f.write(json.dumps(_make_record(allowed=False)) + "\n")
        result = _load_rejections(gw_dir)
        assert result["total_rejected"] == 1
