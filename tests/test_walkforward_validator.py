"""Tests for WalkForward Validator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

from monitoring.walkforward_validator import (
    build_walkforward_report,
    _sharpe,
    _max_drawdown,
    _period_key,
    _dominant_component,
)


# ── Helper fixtures ────────────────────────────────────────────────────────────

def _write_audit(tmp_dir: Path, date: str, rows: list[dict]) -> None:
    """Write trade audit JSONL for a given date."""
    f = tmp_dir / f"trade_audit_{date.replace('-', '')}.jsonl"
    with f.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_daily(tmp_dir: Path, date: str, pnl_pct: float = 0.01) -> None:
    """Write a minimal daily_pnl JSON file."""
    f = tmp_dir / f"daily_pnl_{date.replace('-', '')}.json"
    f.write_text(json.dumps({
        "date": date,
        "daily_pnl_pct": pnl_pct,
        "day_start_capital": 100000,
        "end_capital": 100000 * (1 + pnl_pct),
        "regime": "bull",
    }))


def _entry(ts: str, symbol: str = "AAPL", regime: str = "bull") -> dict:
    return {
        "ts": ts, "event": "ENTRY", "symbol": symbol, "side": "BUY",
        "qty": 10, "fill_price": 100.0, "expected_price": 100.0,
        "slippage_bps": 5.0, "signal": 0.20, "confidence": 0.60,
        "regime": regime, "broker": "ibkr", "pretrade": "PASS",
    }


def _exit(ts: str, symbol: str = "AAPL", pnl_pct: float = 0.01,
          regime: str = "bull", components: dict | None = None) -> dict:
    row = {
        "ts": ts, "event": "EXIT", "symbol": symbol, "side": "SELL",
        "qty": 10, "fill_price": 101.0, "expected_price": 101.0,
        "slippage_bps": 3.0, "signal": 0.15, "confidence": 0.55,
        "entry_signal": 0.20, "regime": regime,
        "pnl_pct": pnl_pct, "pnl_usd": pnl_pct * 1000,
        "exit_reason": "test", "holding_days": 1,
        "broker": "ibkr", "pretrade": "PASS",
    }
    if components:
        row["components"] = components
    return row


# ── Utility tests ──────────────────────────────────────────────────────────────

class TestSharpe:

    def test_positive_returns_positive_sharpe(self):
        pnls = [0.01] * 20 + [0.02] * 10
        s = _sharpe(pnls)
        assert s > 0

    def test_negative_returns_negative_sharpe(self):
        pnls = [-0.01] * 20
        s = _sharpe(pnls)
        assert s < 0

    def test_zero_std_returns_zero(self):
        pnls = [0.0] * 10
        assert _sharpe(pnls) == 0.0

    def test_too_few_points_returns_zero(self):
        assert _sharpe([]) == 0.0
        assert _sharpe([0.01]) == 0.0

    def test_annualised_factor(self):
        # daily mean 0.01, std close to 0.005
        import math
        pnls = [0.01, 0.015, 0.005] * 20
        s = _sharpe(pnls)
        # Should be annualised (~√252)
        assert abs(s) > 1.0  # raw would be ~2, annualised ~30


class TestMaxDrawdown:

    def test_monotonic_up_zero_drawdown(self):
        assert _max_drawdown([0.0, 0.01, 0.02, 0.03]) == 0.0

    def test_full_loss_returns_large_drawdown(self):
        dd = _max_drawdown([0.0, 0.10, 0.05, 0.0])
        assert dd > 0

    def test_empty_returns_zero(self):
        assert _max_drawdown([]) == 0.0


class TestPeriodKey:

    def test_extracts_month(self):
        assert _period_key("2026-03-18T10:00:00Z") == "2026-03"
        assert _period_key("2025-12-01T00:00:00Z") == "2025-12"

    def test_handles_short_string(self):
        # Should not raise
        result = _period_key("2026-03")
        assert result == "2026-03"


class TestDominantComponent:

    def test_uses_components_dict_when_present(self):
        row = {"components": {"ml": 0.7, "tech": 0.3}}
        assert _dominant_component(row) == "ml"

    def test_high_signal_defaults_to_ml(self):
        row = {"signal": 0.25, "entry_signal": 0.25}
        assert _dominant_component(row) == "ml"

    def test_low_signal_defaults_to_tech(self):
        row = {"signal": 0.10, "entry_signal": 0.10}
        assert _dominant_component(row) == "tech"


# ── Integration tests ──────────────────────────────────────────────────────────

class TestEmptyAuditDir:

    def test_empty_dir_returns_valid_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_walkforward_report(audit_dir=Path(tmp))
        assert "periods" in report
        assert "overall" in report
        assert "regime_distribution" in report
        assert "component_alpha_trend" in report
        assert "generated_at" in report

    def test_empty_overall_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_walkforward_report(audit_dir=Path(tmp))
        assert report["overall"]["total_trades"] == 0
        assert report["overall"]["win_rate"] == 0.0


class TestWithAuditData:

    def _setup(self, tmp_dir: Path) -> None:
        """Write a few trade audit + daily P&L files."""
        rows_mar = [
            _entry("2026-03-10T09:30:00Z", "AAPL"),
            _exit("2026-03-10T15:00:00Z", "AAPL", pnl_pct=0.02, regime="bull"),
            _entry("2026-03-11T09:30:00Z", "MSFT"),
            _exit("2026-03-11T15:00:00Z", "MSFT", pnl_pct=-0.01, regime="bear"),
            _entry("2026-03-12T09:30:00Z", "TSLA"),
            _exit("2026-03-12T15:00:00Z", "TSLA", pnl_pct=0.03, regime="bull"),
        ]
        _write_audit(tmp_dir, "2026-03-10", rows_mar[:2])
        _write_audit(tmp_dir, "2026-03-11", rows_mar[2:4])
        _write_audit(tmp_dir, "2026-03-12", rows_mar[4:])
        for d in ["2026-03-10", "2026-03-11", "2026-03-12"]:
            _write_daily(tmp_dir, d, 0.005)

    def test_periods_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(Path(tmp))
            report = build_walkforward_report(audit_dir=Path(tmp))
        assert len(report["periods"]) >= 1
        p = report["periods"][0]
        assert p["period"] == "2026-03"
        assert p["trades"] == 3

    def test_wins_counted_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(Path(tmp))
            report = build_walkforward_report(audit_dir=Path(tmp))
        p = report["periods"][0]
        # 2 wins (AAPL +2%, TSLA +3%), 1 loss (MSFT -1%)
        assert p["wins"] == 2
        assert p["win_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_regime_counts_in_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(Path(tmp))
            report = build_walkforward_report(audit_dir=Path(tmp))
        p = report["periods"][0]
        assert p["regime_counts"]["bull"] == 2
        assert p["regime_counts"]["bear"] == 1

    def test_regime_distribution_overall(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(Path(tmp))
            report = build_walkforward_report(audit_dir=Path(tmp))
        dist = report["regime_distribution"]
        assert dist["bull"] == 2
        assert dist["bear"] == 1

    def test_overall_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(Path(tmp))
            report = build_walkforward_report(audit_dir=Path(tmp))
        ov = report["overall"]
        assert ov["total_trades"] == 3
        assert ov["total_wins"] == 2

    def test_sharpe_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(Path(tmp))
            report = build_walkforward_report(audit_dir=Path(tmp))
        p = report["periods"][0]
        # With 3 daily rows of 0.005 each → std=0 → sharpe=0
        assert isinstance(p["sharpe"], float)

    def test_avg_slippage_in_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(Path(tmp))
            report = build_walkforward_report(audit_dir=Path(tmp))
        p = report["periods"][0]
        # EXIT slippage = 3.0 bps each
        assert p["avg_slippage_bps"] == pytest.approx(3.0, abs=0.1)


class TestComponentAlpha:

    def test_component_alpha_with_explicit_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                _entry("2026-03-10T09:00:00Z"),
                _exit("2026-03-10T15:00:00Z", pnl_pct=0.02,
                       components={"ml": 0.8, "tech": 0.2, "sentiment": 0.0,
                                   "momentum": 0.0, "pairs": 0.0}),
            ]
            _write_audit(tmp_path, "2026-03-10", rows)
            _write_daily(tmp_path, "2026-03-10", 0.02)
            report = build_walkforward_report(audit_dir=tmp_path)
        p = report["periods"][0]
        # ml alpha = 0.02 * 0.8 = 0.016; tech = 0.02 * 0.2 = 0.004
        assert p["component_alpha"]["ml"] == pytest.approx(0.016, abs=1e-6)
        assert p["component_alpha"]["tech"] == pytest.approx(0.004, abs=1e-6)

    def test_component_alpha_trend_has_all_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [_entry("2026-03-10T09:00:00Z"), _exit("2026-03-10T15:00:00Z")]
            _write_audit(tmp_path, "2026-03-10", rows)
            _write_daily(tmp_path, "2026-03-10")
            report = build_walkforward_report(audit_dir=tmp_path)
        trend = report["component_alpha_trend"]
        for comp in ("ml", "tech", "sentiment", "momentum", "pairs"):
            assert comp in trend


class TestRegimeTrend:

    def test_regime_trend_has_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                _entry("2026-03-10T09:00:00Z"),
                _exit("2026-03-10T15:00:00Z", pnl_pct=0.01, regime="bull"),
                _entry("2026-03-11T09:00:00Z"),
                _exit("2026-03-11T15:00:00Z", pnl_pct=-0.01, regime="bear"),
            ]
            _write_audit(tmp_path, "2026-03-10", rows)
            _write_daily(tmp_path, "2026-03-10")
            report = build_walkforward_report(audit_dir=tmp_path)
        regimes = {r["regime"] for r in report["regime_trend"]}
        assert "bull" in regimes
        assert "bear" in regimes

    def test_regime_win_rate_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                _entry("2026-03-10T09:00:00Z"),
                _exit("2026-03-10T10:00:00Z", pnl_pct=0.02, regime="bull"),
                _entry("2026-03-10T11:00:00Z"),
                _exit("2026-03-10T12:00:00Z", pnl_pct=0.01, regime="bull"),
            ]
            _write_audit(tmp_path, "2026-03-10", rows)
            _write_daily(tmp_path, "2026-03-10")
            report = build_walkforward_report(audit_dir=tmp_path)
        bull = next(r for r in report["regime_trend"] if r["regime"] == "bull")
        assert bull["win_rate"] == pytest.approx(1.0, abs=0.01)
        assert bull["trades"] == 2


class TestMultipleFiles:

    def test_multiple_files_aggregated_in_one_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Two separate audit files for same month
            _write_audit(tmp_path, "2026-03-10", [
                _entry("2026-03-10T09:00:00Z"),
                _exit("2026-03-10T15:00:00Z", pnl_pct=0.01),
            ])
            _write_audit(tmp_path, "2026-03-15", [
                _entry("2026-03-15T09:00:00Z"),
                _exit("2026-03-15T15:00:00Z", pnl_pct=0.02),
            ])
            _write_daily(tmp_path, "2026-03-10")
            _write_daily(tmp_path, "2026-03-15")
            report = build_walkforward_report(audit_dir=tmp_path)
        # Both exits should be in the same "2026-03" period
        assert len(report["periods"]) == 1
        assert report["periods"][0]["trades"] == 2

    def test_multiple_months_separate_periods(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_audit(tmp_path, "2026-02-28", [
                _entry("2026-02-28T09:00:00Z"),
                _exit("2026-02-28T15:00:00Z", pnl_pct=0.01),
            ])
            _write_audit(tmp_path, "2026-03-10", [
                _entry("2026-03-10T09:00:00Z"),
                _exit("2026-03-10T15:00:00Z", pnl_pct=0.02),
            ])
            _write_daily(tmp_path, "2026-02-28", 0.005)
            _write_daily(tmp_path, "2026-03-10", 0.01)
            report = build_walkforward_report(audit_dir=tmp_path)
        periods = [p["period"] for p in report["periods"]]
        assert "2026-02" in periods
        assert "2026-03" in periods
        assert len(report["periods"]) == 2


class TestGrossPnl:

    def test_gross_pnl_usd_summed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = [
                _entry("2026-03-10T09:00:00Z"),
                _exit("2026-03-10T10:00:00Z", pnl_pct=0.02),  # pnl_usd = 20
                _entry("2026-03-10T11:00:00Z"),
                _exit("2026-03-10T12:00:00Z", pnl_pct=-0.01),  # pnl_usd = -10
            ]
            _write_audit(tmp_path, "2026-03-10", rows)
            _write_daily(tmp_path, "2026-03-10")
            report = build_walkforward_report(audit_dir=tmp_path)
        p = report["periods"][0]
        assert p["gross_pnl_usd"] == pytest.approx(20.0 - 10.0, abs=0.1)
