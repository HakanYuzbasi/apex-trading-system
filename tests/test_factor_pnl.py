"""tests/test_factor_pnl.py — Factor P&L Decomposition tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from monitoring.factor_pnl import (
    FactorBucket,
    FactorPnlAnalyzer,
    FactorPnlReport,
    _dominant_factor,
    _empty_factor_map,
    _factor_weights,
    _load_closed_trades,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_data(tmp_path):
    return tmp_path


@pytest.fixture()
def analyzer(tmp_data):
    return FactorPnlAnalyzer(data_dir=tmp_data)


def _recent_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_exits(data_dir: Path, records: list[dict]) -> None:
    p = data_dir / "users" / "admin" / "audit"
    p.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    (p / f"trade_audit_{today}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )


def _exit_rec(
    pnl_pct: float = 0.01,
    ml: float = 0.0,
    tech: float = 0.0,
    sentiment: float = 0.0,
    momentum: float = 0.0,
    asset: str = "EQUITY",
    regime: str = "bull",
) -> dict:
    return {
        "event_type": "EXIT",
        "pnl_pct": pnl_pct,
        "pnl_dollars": pnl_pct * 10_000,
        "ml_signal": ml,
        "tech_signal": tech,
        "sentiment_signal": sentiment,
        "cs_momentum_signal": momentum,
        "asset_class": asset,
        "regime": regime,
        "timestamp": _recent_ts(),
    }


# ── FactorBucket ──────────────────────────────────────────────────────────────

class TestFactorBucket:
    def test_avg_pnl_pct_empty(self):
        b = FactorBucket("ml")
        assert b.avg_pnl_pct is None

    def test_win_rate_empty(self):
        b = FactorBucket("ml")
        assert b.win_rate is None

    def test_to_dict_keys(self):
        b = FactorBucket("ml", pnl_sum=100.0, pnl_pct_sum=0.01,
                         trade_count=5, win_count=3)
        d = b.to_dict()
        assert d["factor"] == "ml"
        assert "avg_pnl_pct" in d
        assert "win_rate" in d


# ── _factor_weights ────────────────────────────────────────────────────────────

class TestFactorWeights:
    def test_all_zero_returns_residual(self):
        rec = _exit_rec(ml=0.0, tech=0.0, sentiment=0.0, momentum=0.0)
        w = _factor_weights(rec)
        assert w["residual"] == 1.0
        assert w["ml"] == 0.0

    def test_single_factor_dominates(self):
        rec = _exit_rec(ml=0.20, tech=0.0, sentiment=0.0, momentum=0.0)
        w = _factor_weights(rec)
        assert abs(w["ml"] - 1.0) < 1e-9
        assert w["residual"] == 0.0

    def test_equal_factors_equal_weights(self):
        rec = _exit_rec(ml=0.10, tech=0.10, sentiment=0.10, momentum=0.10)
        w = _factor_weights(rec)
        for k in ("ml", "technical", "sentiment", "momentum"):
            assert abs(w[k] - 0.25) < 1e-9

    def test_weights_sum_to_one(self):
        rec = _exit_rec(ml=0.15, tech=0.08, sentiment=0.03, momentum=0.12)
        w = _factor_weights(rec)
        assert abs(sum(w.values()) - 1.0) < 1e-9


# ── _dominant_factor ──────────────────────────────────────────────────────────

class TestDominantFactor:
    def test_ml_dominant(self):
        rec = _exit_rec(ml=0.20, tech=0.05)
        assert _dominant_factor(rec) == "ml"

    def test_tech_dominant(self):
        rec = _exit_rec(ml=0.05, tech=0.18)
        assert _dominant_factor(rec) == "technical"

    def test_all_zero_residual(self):
        rec = _exit_rec()
        assert _dominant_factor(rec) == "residual"


# ── _load_closed_trades ───────────────────────────────────────────────────────

class TestLoadClosedTrades:
    def test_empty_returns_empty(self, tmp_data):
        assert _load_closed_trades(tmp_data, 7) == []

    def test_loads_exit_rows(self, tmp_data):
        _write_exits(tmp_data, [_exit_rec(), _exit_rec(pnl_pct=-0.01)])
        rows = _load_closed_trades(tmp_data, 7)
        assert len(rows) == 2

    def test_skips_entry_rows(self, tmp_data):
        recs = [
            {"event_type": "ENTRY", "pnl_pct": 0.0, "timestamp": _recent_ts()},
            _exit_rec(),
        ]
        _write_exits(tmp_data, recs)
        rows = _load_closed_trades(tmp_data, 7)
        assert len(rows) == 1


# ── FactorPnlAnalyzer.build_report ────────────────────────────────────────────

class TestBuildReport:
    def test_empty_data(self, analyzer):
        report = analyzer.build_report(7)
        assert isinstance(report, FactorPnlReport)
        assert report.total_trades == 0
        assert report.by_factor == []

    def test_counts_trades(self, analyzer, tmp_data):
        _write_exits(tmp_data, [_exit_rec() for _ in range(5)])
        report = analyzer.build_report(7)
        assert report.total_trades == 5

    def test_total_pnl_accumulated(self, analyzer, tmp_data):
        _write_exits(tmp_data, [_exit_rec(pnl_pct=0.01) for _ in range(4)])
        report = analyzer.build_report(7)
        assert abs(report.total_pnl_pct - 0.04) < 1e-6

    def test_by_factor_has_all_factors(self, analyzer, tmp_data):
        _write_exits(tmp_data, [_exit_rec(ml=0.15, tech=0.05) for _ in range(5)])
        report = analyzer.build_report(7)
        factor_names = {b.factor for b in report.by_factor}
        assert "ml" in factor_names
        assert "residual" in factor_names

    def test_ml_dominated_shows_in_report(self, analyzer, tmp_data):
        _write_exits(tmp_data, [_exit_rec(ml=0.20, pnl_pct=0.02) for _ in range(10)])
        report = analyzer.build_report(7)
        ml_bucket = next(b for b in report.by_factor if b.factor == "ml")
        assert ml_bucket.trade_count > 0
        assert ml_bucket.pnl_pct_sum > 0

    def test_by_asset_grouped(self, analyzer, tmp_data):
        _write_exits(tmp_data, [
            _exit_rec(asset="EQUITY", ml=0.10),
            _exit_rec(asset="CRYPTO", ml=0.10),
        ])
        report = analyzer.build_report(7)
        assert "EQUITY" in report.by_asset
        assert "CRYPTO" in report.by_asset

    def test_by_regime_grouped(self, analyzer, tmp_data):
        _write_exits(tmp_data, [
            _exit_rec(regime="bull",  ml=0.10),
            _exit_rec(regime="bear",  ml=0.10),
            _exit_rec(regime="neutral", tech=0.10),
        ])
        report = analyzer.build_report(7)
        assert "bull" in report.by_regime
        assert "bear" in report.by_regime

    def test_time_windows_populated(self, analyzer, tmp_data):
        _write_exits(tmp_data, [_exit_rec(ml=0.15) for _ in range(5)])
        report = analyzer.build_report(30)
        assert isinstance(report.last_1d, list)
        assert isinstance(report.last_7d, list)
        assert isinstance(report.last_30d, list)

    def test_to_dict_serialisable(self, analyzer, tmp_data):
        _write_exits(tmp_data, [_exit_rec(ml=0.15) for _ in range(3)])
        report = analyzer.build_report(7)
        d = report.to_dict()
        import json as _json
        serialised = _json.dumps(d)
        assert "by_factor" in serialised

    def test_win_rate_in_factor_bucket(self, analyzer, tmp_data):
        _write_exits(tmp_data, [
            _exit_rec(pnl_pct=0.02, ml=0.15),
            _exit_rec(pnl_pct=0.02, ml=0.15),
            _exit_rec(pnl_pct=-0.01, ml=0.15),
        ])
        report = analyzer.build_report(7)
        ml_b = next(b for b in report.by_factor if b.factor == "ml")
        assert ml_b.win_rate is not None
        # 2 wins out of 3 → ~0.67
        assert ml_b.win_rate > 0.5

    def test_generated_at_set(self, analyzer):
        report = analyzer.build_report()
        assert report.generated_at != ""
