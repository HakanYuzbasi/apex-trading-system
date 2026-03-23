"""Tests for EarningsEventGate."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from risk.earnings_gate import EarningsEventGate


def _gate() -> EarningsEventGate:
    return EarningsEventGate()


def _utc(hours_from_now: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours_from_now)


class TestDefaultBehaviour:

    def test_unknown_symbol_returns_one(self):
        gate = _gate()
        assert gate.get_sizing_mult("AAPL") == pytest.approx(1.0)

    def test_crypto_always_one(self):
        gate = _gate()
        for sym in ("CRYPTO:BTC/USD", "BTC/USD", "ETH/USD"):
            assert gate.get_sizing_mult(sym) == pytest.approx(1.0)

    def test_fx_always_one(self):
        gate = _gate()
        assert gate.get_sizing_mult("FX:EUR/USD") == pytest.approx(1.0)

    def test_etf_always_one(self):
        gate = _gate()
        for etf in ("SPY", "QQQ", "GLD"):
            assert gate.get_sizing_mult(etf) == pytest.approx(1.0)


class TestSizingMultipliers:

    def test_danger_zone_lt24h(self):
        gate = _gate()
        gate.update_earnings("AAPL", _utc(12))
        assert gate.get_sizing_mult("AAPL") == pytest.approx(0.25)

    def test_warning_zone_24_48h(self):
        gate = _gate()
        gate.update_earnings("MSFT", _utc(36))
        assert gate.get_sizing_mult("MSFT") == pytest.approx(0.50)

    def test_caution_zone_48_72h(self):
        gate = _gate()
        gate.update_earnings("TSLA", _utc(60))
        assert gate.get_sizing_mult("TSLA") == pytest.approx(0.75)

    def test_beyond_72h_returns_one(self):
        gate = _gate()
        gate.update_earnings("NVDA", _utc(96))
        assert gate.get_sizing_mult("NVDA") == pytest.approx(1.0)

    def test_post_earnings_returns_one(self):
        gate = _gate()
        gate.update_earnings("GOOGL", _utc(-5))   # 5h ago
        assert gate.get_sizing_mult("GOOGL") == pytest.approx(1.0)

    def test_exactly_at_24h_boundary(self):
        gate = _gate()
        gate.update_earnings("AMD", _utc(24.0))
        # exactly 24h hits ≤24 → danger zone (0.25)
        assert gate.get_sizing_mult("AMD") == pytest.approx(0.25)

    def test_exactly_at_48h_boundary(self):
        gate = _gate()
        gate.update_earnings("INTC", _utc(48.0))
        assert gate.get_sizing_mult("INTC") == pytest.approx(0.50)

    def test_exactly_at_72h_boundary(self):
        gate = _gate()
        gate.update_earnings("QCOM", _utc(72.0))
        assert gate.get_sizing_mult("QCOM") == pytest.approx(0.75)


class TestUpdateEarnings:

    def test_clear_earnings_with_none(self):
        gate = _gate()
        gate.update_earnings("AAPL", _utc(12))
        assert gate.get_sizing_mult("AAPL") < 1.0
        gate.update_earnings("AAPL", None)
        assert gate.get_sizing_mult("AAPL") == pytest.approx(1.0)

    def test_overwrite_updates_date(self):
        gate = _gate()
        gate.update_earnings("AAPL", _utc(12))   # danger
        gate.update_earnings("AAPL", _utc(96))   # far away → clear
        assert gate.get_sizing_mult("AAPL") == pytest.approx(1.0)

    def test_naive_datetime_accepted(self):
        gate = _gate()
        # Naive datetime (no tzinfo) → should be accepted without crash
        naive_dt = datetime.utcnow() + timedelta(hours=10)
        gate.update_earnings("META", naive_dt)
        assert gate.get_sizing_mult("META") == pytest.approx(0.25)


class TestHoursUntilEarnings:

    def test_known_future_returns_positive(self):
        gate = _gate()
        gate.update_earnings("AAPL", _utc(30))
        h = gate.hours_until_earnings("AAPL")
        assert h is not None
        assert 29 < h < 31

    def test_unknown_returns_none(self):
        gate = _gate()
        assert gate.hours_until_earnings("UNKNOWN") is None

    def test_past_earnings_returns_none(self):
        gate = _gate()
        gate.update_earnings("PAST", _utc(-2))
        assert gate.hours_until_earnings("PAST") is None


class TestReport:

    def test_report_structure(self):
        gate = _gate()
        gate.update_earnings("AAPL", _utc(30))
        gate.update_earnings("MSFT", _utc(60))
        report = gate.get_report()
        assert "symbols" in report
        assert "count" in report
        assert report["count"] == 2

    def test_report_has_zone(self):
        gate = _gate()
        gate.update_earnings("AAPL", _utc(12))
        report = gate.get_report()
        assert report["symbols"]["AAPL"]["zone"] == "danger"

    def test_report_has_sizing_mult(self):
        gate = _gate()
        gate.update_earnings("TSLA", _utc(36))
        report = gate.get_report()
        assert report["symbols"]["TSLA"]["sizing_mult"] == pytest.approx(0.50)

    def test_empty_gate_empty_report(self):
        gate = _gate()
        report = gate.get_report()
        assert report["count"] == 0
        assert report["symbols"] == {}


class TestZoneHelper:

    def test_zone_post(self):
        assert EarningsEventGate._zone(-5) == "post"

    def test_zone_danger(self):
        assert EarningsEventGate._zone(10) == "danger"

    def test_zone_warning(self):
        assert EarningsEventGate._zone(36) == "warning"

    def test_zone_caution(self):
        assert EarningsEventGate._zone(60) == "caution"

    def test_zone_clear(self):
        assert EarningsEventGate._zone(100) == "clear"
