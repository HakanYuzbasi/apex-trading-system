"""
Tests for ComplianceManager pre-trade checks, audit trail, and statistics.
"""
import json
import tempfile
from pathlib import Path

import pytest

from monitoring.compliance_manager import ComplianceManager


def _make_config(**overrides):
    base = {
        "max_position_pct": 0.02,
        "max_exposure_pct": 0.95,
        "max_shares_per_symbol": 200,
        "allow_short_selling": False,
        "min_stock_price": 5.0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def mgr(tmp_path):
    return ComplianceManager(audit_dir=str(tmp_path / "audit"))


class TestPreTradeCheckApproved:
    def test_valid_buy_approved(self, mgr):
        result = mgr.pre_trade_check(
            symbol="AAPL", side="BUY", quantity=100, price=180.0,
            portfolio_value=1_000_000, current_positions={},
            config=_make_config(),
        )
        assert result["approved"] is True
        assert len(result["violations"]) == 0

    def test_valid_sell_existing_position_approved(self, mgr):
        result = mgr.pre_trade_check(
            symbol="AAPL", side="SELL", quantity=50, price=180.0,
            portfolio_value=1_000_000, current_positions={"AAPL": 100},
            config=_make_config(),
        )
        assert result["approved"] is True

    def test_check_id_present(self, mgr):
        result = mgr.pre_trade_check(
            symbol="X", side="BUY", quantity=10, price=100.0,
            portfolio_value=500_000, current_positions={},
            config=_make_config(),
        )
        assert "check_id" in result
        assert result["check_id"].startswith("CHK-")


class TestPreTradeCheckViolations:
    def test_position_too_large_rejected(self, mgr):
        # notional = 1000 * 180 = 180k > 2% of 100k = 2k
        result = mgr.pre_trade_check(
            symbol="AAPL", side="BUY", quantity=1000, price=180.0,
            portfolio_value=100_000, current_positions={},
            config=_make_config(max_position_pct=0.02),
        )
        assert result["approved"] is False
        assert any("exceeds limit" in v for v in result["violations"])

    def test_short_selling_blocked(self, mgr):
        result = mgr.pre_trade_check(
            symbol="AAPL", side="SELL", quantity=100, price=180.0,
            portfolio_value=1_000_000, current_positions={},
            config=_make_config(allow_short_selling=False),
        )
        assert result["approved"] is False
        assert any("Short selling" in v for v in result["violations"])

    def test_short_selling_allowed_when_flag_set(self, mgr):
        result = mgr.pre_trade_check(
            symbol="AAPL", side="SELL", quantity=100, price=180.0,
            portfolio_value=1_000_000, current_positions={},
            config=_make_config(allow_short_selling=True),
        )
        # No short-selling violation; may still have other checks
        assert not any("Short selling" in v for v in result["violations"])

    def test_max_shares_per_symbol_violated(self, mgr):
        result = mgr.pre_trade_check(
            symbol="AAPL", side="BUY", quantity=150, price=10.0,
            portfolio_value=1_000_000, current_positions={"AAPL": 100},
            config=_make_config(max_shares_per_symbol=200),
        )
        assert result["approved"] is False
        assert any("shares" in v.lower() for v in result["violations"])


class TestPreTradeCheckWarnings:
    def test_low_price_generates_warning(self, mgr):
        result = mgr.pre_trade_check(
            symbol="PENNY", side="BUY", quantity=100, price=3.0,
            portfolio_value=1_000_000, current_positions={},
            config=_make_config(min_stock_price=5.0),
        )
        assert len(result["warnings"]) > 0
        assert any("below minimum" in w for w in result["warnings"])

    def test_above_min_price_no_warning(self, mgr):
        result = mgr.pre_trade_check(
            symbol="AAPL", side="BUY", quantity=10, price=180.0,
            portfolio_value=1_000_000, current_positions={},
            config=_make_config(min_stock_price=5.0),
        )
        assert len(result["warnings"]) == 0


class TestAuditTrail:
    def test_log_trade_returns_id(self, mgr):
        trade = {"symbol": "AAPL", "side": "BUY", "quantity": 100, "price": 180.0}
        log_id = mgr.log_trade(trade)
        assert log_id.startswith("TRD-")

    def test_audit_file_created(self, mgr, tmp_path):
        mgr.log_trade({"symbol": "X"})
        audit_files = list(Path(mgr.audit_dir).glob("audit_*.jsonl"))
        assert len(audit_files) == 1

    def test_audit_record_contains_hash(self, mgr, tmp_path):
        mgr.log_trade({"symbol": "X", "qty": 10})
        audit_files = list(Path(mgr.audit_dir).glob("audit_*.jsonl"))
        line = audit_files[0].read_text().strip().splitlines()[0]
        record = json.loads(line)
        assert "hash" in record
        assert len(record["hash"]) == 64  # SHA-256 hex

    def test_audit_trail_verification_passes(self, mgr):
        mgr.log_trade({"symbol": "AAPL"})
        result = mgr.verify_audit_trail()
        assert result["verified"] is True

    def test_no_audit_log_verifies_as_true(self, tmp_path):
        fresh = ComplianceManager(audit_dir=str(tmp_path / "empty_audit"))
        result = fresh.verify_audit_trail(date="19000101")
        assert result["verified"] is True


class TestStatistics:
    def test_empty_returns_empty_dict(self, mgr):
        stats = mgr.get_statistics()
        assert stats == {}

    def test_stats_after_checks(self, mgr):
        for _ in range(3):
            mgr.pre_trade_check(
                symbol="AAPL", side="BUY", quantity=10, price=180.0,
                portfolio_value=1_000_000, current_positions={},
                config=_make_config(),
            )
        stats = mgr.get_statistics()
        assert stats["total_checks"] == 3
        assert stats["approved"] == 3
        assert stats["approval_rate"] == pytest.approx(1.0)

    def test_rejection_recorded_in_stats(self, mgr):
        mgr.pre_trade_check(
            symbol="AAPL", side="SELL", quantity=100, price=5.0,
            portfolio_value=100_000, current_positions={},
            config=_make_config(allow_short_selling=False),
        )
        stats = mgr.get_statistics()
        assert stats["rejected"] >= 1
        assert stats["total_violations"] >= 1


class TestHashIntegrity:
    def test_same_record_same_hash(self, mgr):
        record = {"log_id": "x", "timestamp": "t", "check_id": None, "trade": {"a": 1}}
        h1 = mgr._calculate_hash(record)
        h2 = mgr._calculate_hash(record)
        assert h1 == h2

    def test_different_record_different_hash(self, mgr):
        r1 = {"data": "a"}
        r2 = {"data": "b"}
        assert mgr._calculate_hash(r1) != mgr._calculate_hash(r2)
