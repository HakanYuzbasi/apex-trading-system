"""Tests for EOD Performance Digest generator."""
from __future__ import annotations

import json
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from monitoring.eod_digest import EODDigestGenerator, EODReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_audit(audit_dir: Path, day: date, rows: list[dict]) -> None:
    """Write a fake trade audit JSONL file."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"trade_audit_{day.strftime('%Y%m%d')}.jsonl"
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_exit(symbol="AAPL", pnl_usd=50.0, pnl_pct=0.5, regime="bull",
               broker="ibkr", reason="Excellence: profit target") -> dict:
    return {
        "ts": "2026-03-22T20:00:00Z",
        "event": "EXIT",
        "symbol": symbol,
        "side": "SELL",
        "qty": 10,
        "fill_price": 200.0,
        "slippage_bps": 5.0,
        "signal": 0.22,
        "confidence": 0.65,
        "entry_signal": 0.20,
        "regime": regime,
        "pnl_pct": pnl_pct,
        "pnl_usd": pnl_usd,
        "exit_reason": reason,
        "holding_days": 0.2,
        "broker": broker,
        "pretrade": "PASS",
    }


def _make_entry(symbol="AAPL", broker="ibkr", regime="bull") -> dict:
    return {
        "ts": "2026-03-22T16:00:00Z",
        "event": "ENTRY",
        "symbol": symbol,
        "side": "BUY",
        "qty": 10,
        "fill_price": 199.0,
        "slippage_bps": 4.0,
        "signal": 0.20,
        "confidence": 0.60,
        "regime": regime,
        "broker": broker,
        "pretrade": "PASS",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEODDigestGenerator:

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.data_dir = Path(self.tmp)
        self.today = date(2026, 3, 22)
        self.audit_dir = self.data_dir / "users" / "admin" / "audit"
        self.gen = EODDigestGenerator(data_dir=self.data_dir)

    def test_empty_audit_produces_zero_trades(self):
        _write_audit(self.audit_dir, self.today, [])
        report = self.gen.generate(report_date=self.today)
        assert report.total_trades == 0
        assert report.total_realized_pnl == 0.0
        assert report.overall_win_rate is None

    def test_single_winning_trade(self):
        _write_audit(self.audit_dir, self.today, [
            _make_exit(symbol="AAPL", pnl_usd=100.0, pnl_pct=0.5),
            _make_entry(symbol="AAPL"),
        ])
        report = self.gen.generate(report_date=self.today)
        assert report.total_trades == 1
        assert report.total_entries == 1
        assert report.total_realized_pnl == 100.0
        assert report.overall_win_rate == 1.0

    def test_mixed_trades_win_rate(self):
        rows = [
            _make_exit("AAPL", pnl_usd=200.0, pnl_pct=1.0, regime="bull"),
            _make_exit("MSFT", pnl_usd=-50.0, pnl_pct=-0.5, regime="bull", reason="Stop loss"),
            _make_exit("GOOG", pnl_usd=75.0, pnl_pct=0.8, regime="neutral"),
        ]
        _write_audit(self.audit_dir, self.today, rows)
        report = self.gen.generate(report_date=self.today)
        assert report.total_trades == 3
        assert pytest.approx(report.overall_win_rate, abs=0.01) == 2 / 3
        assert pytest.approx(report.total_realized_pnl, abs=0.01) == 225.0

    def test_by_broker_breakdown(self):
        rows = [
            _make_exit("AAPL", pnl_usd=100.0, broker="ibkr"),
            _make_exit("BTC/USD", pnl_usd=-30.0, broker="alpaca"),
        ]
        _write_audit(self.audit_dir, self.today, rows)
        report = self.gen.generate(report_date=self.today)
        assert "ibkr" in report.by_broker
        assert "alpaca" in report.by_broker
        assert report.by_broker["ibkr"]["realized_pnl"] == 100.0
        assert report.by_broker["alpaca"]["realized_pnl"] == -30.0
        assert report.by_broker["ibkr"]["trades"] == 1
        assert report.by_broker["ibkr"]["win_rate"] == 1.0
        assert report.by_broker["alpaca"]["win_rate"] == 0.0

    def test_by_regime_breakdown(self):
        rows = [
            _make_exit("AAPL", pnl_usd=100.0, regime="bull"),
            _make_exit("MSFT", pnl_usd=50.0, regime="bull"),
            _make_exit("GLD", pnl_usd=-20.0, regime="bear"),
        ]
        _write_audit(self.audit_dir, self.today, rows)
        report = self.gen.generate(report_date=self.today)
        assert "bull" in report.by_regime
        assert report.by_regime["bull"]["trades"] == 2
        assert report.by_regime["bull"]["win_rate"] == 1.0
        assert "bear" in report.by_regime

    def test_top_and_bottom_trades(self):
        rows = [
            _make_exit("A", pnl_usd=500.0, pnl_pct=5.0),
            _make_exit("B", pnl_usd=100.0, pnl_pct=1.0),
            _make_exit("C", pnl_usd=-200.0, pnl_pct=-2.0, reason="Stop loss"),
        ]
        _write_audit(self.audit_dir, self.today, rows)
        report = self.gen.generate(report_date=self.today)
        assert report.top_trades[0]["symbol"] == "A"
        assert report.top_trades[0]["pnl_usd"] == 500.0
        assert report.bottom_trades[-1]["symbol"] == "C"

    def test_exit_reason_bucketing(self):
        rows = [
            _make_exit("A", reason="Stop loss: -3%"),
            _make_exit("B", reason="Excellence: signal turned bearish"),
            _make_exit("C", reason="Hedge: force-exit corr=0.91"),
        ]
        _write_audit(self.audit_dir, self.today, rows)
        report = self.gen.generate(report_date=self.today)
        assert report.exit_reason_summary.get("stop_loss", 0) == 1
        assert report.exit_reason_summary.get("excellence", 0) == 1
        assert report.exit_reason_summary.get("hedge_force", 0) == 1

    def test_recommendations_low_win_rate(self):
        rows = [_make_exit(f"S{i}", pnl_usd=-10.0, regime="bull") for i in range(5)]
        _write_audit(self.audit_dir, self.today, rows)
        report = self.gen.generate(report_date=self.today)
        assert len(report.recommendations) > 0
        rec_text = " ".join(report.recommendations).lower()
        assert "win rate" in rec_text or "bull" in rec_text

    def test_recommendations_high_slippage(self):
        rows = [
            {**_make_exit("AAPL", pnl_usd=50.0), "slippage_bps": 30.0},
            {**_make_exit("MSFT", pnl_usd=50.0), "slippage_bps": 25.0},
        ]
        _write_audit(self.audit_dir, self.today, rows)
        report = self.gen.generate(report_date=self.today)
        # Should have a slippage rec since avg is 27.5 bps > 20
        rec_text = " ".join(report.recommendations).lower()
        assert "slippage" in rec_text

    def test_governor_tier_captured(self):
        _write_audit(self.audit_dir, self.today, [_make_exit()])
        gov_snap = MagicMock()
        gov_snap.tier = "yellow"
        gov_snap.sharpe = 0.75
        report = self.gen.generate(report_date=self.today, governor_snapshot=gov_snap)
        assert report.governor_tier == "yellow"
        assert report.governor_sharpe == pytest.approx(0.75, abs=0.001)

    def test_governor_red_tier_recommendation(self):
        _write_audit(self.audit_dir, self.today, [_make_exit()])
        gov_snap = MagicMock()
        gov_snap.tier = "red"
        gov_snap.sharpe = -0.5
        report = self.gen.generate(report_date=self.today, governor_snapshot=gov_snap)
        assert any("red" in r.lower() or "emergency" in r.lower() for r in report.recommendations)

    def test_unrealized_pnl_from_positions(self):
        _write_audit(self.audit_dir, self.today, [])
        positions = {"AAPL": {"qty": 10, "avg_cost": 190.0}}
        price_cache = {"AAPL": 200.0}
        report = self.gen.generate(
            report_date=self.today,
            positions=positions,
            price_cache=price_cache,
        )
        # 10 shares × ($200 - $190) = $100 unrealized
        assert report.total_unrealized_pnl == pytest.approx(100.0, abs=0.01)
        assert report.by_broker["ibkr"]["unrealized_pnl"] == pytest.approx(100.0, abs=0.01)

    def test_save_writes_valid_json(self):
        _write_audit(self.audit_dir, self.today, [_make_exit()])
        report = self.gen.generate(report_date=self.today)
        path = self.gen.save(report)
        assert path.exists()
        with open(path) as fh:
            data = json.load(fh)
        assert data["report_date"] == "2026-03-22"
        assert "total_trades" in data
        assert "recommendations" in data

    def test_load_latest_returns_saved_reports(self):
        _write_audit(self.audit_dir, self.today, [_make_exit()])
        report = self.gen.generate(report_date=self.today)
        self.gen.save(report)
        loaded = self.gen.load_latest(days_back=2)
        assert len(loaded) == 1
        assert loaded[0]["report_date"] == "2026-03-22"

    def test_report_date_field(self):
        _write_audit(self.audit_dir, self.today, [])
        report = self.gen.generate(report_date=self.today)
        assert report.report_date == "2026-03-22"
        assert "Z" in report.generated_at

    def test_avg_slippage_bps(self):
        rows = [
            {**_make_exit("AAPL"), "slippage_bps": 10.0},
            {**_make_exit("MSFT"), "slippage_bps": 6.0},
        ]
        _write_audit(self.audit_dir, self.today, rows)
        report = self.gen.generate(report_date=self.today)
        assert report.avg_slippage_bps == pytest.approx(8.0, abs=0.1)

    def test_to_dict_serializable(self):
        _write_audit(self.audit_dir, self.today, [_make_exit()])
        report = self.gen.generate(report_date=self.today)
        d = report.to_dict()
        # Must round-trip through JSON without error
        json_str = json.dumps(d, default=str)
        recovered = json.loads(json_str)
        assert recovered["total_trades"] == 1


class TestEODDigestMissingAudit:

    def test_no_audit_file_returns_empty_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = EODDigestGenerator(data_dir=tmp)
            report = gen.generate(report_date=date(2026, 3, 22))
            assert report.total_trades == 0
            assert report.total_realized_pnl == 0.0

    def test_corrupt_audit_line_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            audit_dir = data_dir / "users" / "admin" / "audit"
            audit_dir.mkdir(parents=True)
            path = audit_dir / "trade_audit_20260322.jsonl"
            with open(path, "w") as fh:
                fh.write("{bad json\n")
                fh.write(json.dumps(_make_exit("AAPL", pnl_usd=50.0)) + "\n")
            gen = EODDigestGenerator(data_dir=data_dir)
            report = gen.generate(report_date=date(2026, 3, 22))
            assert report.total_trades == 1
