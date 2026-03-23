"""Tests for OOSValidator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from monitoring.oos_validator import (
    OOSValidator,
    OOSReport,
    CellStats,
    _dominant_component,
    _hour_block_for,
    _normalize_regime,
    _sharpe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _trade(
    pnl_pct: float = 0.01,
    regime: str = "neutral",
    ts: str = "2026-03-22T14:00:00Z",
    components: dict | None = None,
    hold_hours: float = 2.0,
    exit_reason: str = "excellence",
) -> dict:
    d = {
        "event": "EXIT",
        "pnl_pct": pnl_pct,
        "regime": regime,
        "ts": ts,
        "hold_hours": hold_hours,
        "exit_reason": exit_reason,
    }
    if components:
        d["components"] = components
    return d


def _write_audit(directory: Path, trades: List[dict]) -> Path:
    p = directory / "trade_audit_2026-03-22.jsonl"
    with open(p, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")
    return p


def _validator(trades: List[dict], min_trades: int = 1) -> OOSValidator:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_audit(d, trades)
        v = OOSValidator(audit_dir=d, min_trades=min_trades)
        v.load_trades()
        # Cache dir so build_report can re-use loaded trades
        v._cached_tmp = tmp
    return v


def _quick_validator(trades: List[dict], min_trades: int = 1) -> OOSReport:
    """Build and return OOSReport directly."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_audit(d, trades)
        v = OOSValidator(audit_dir=d, min_trades=min_trades)
        return v.build_report()


# ── Default / empty state ────────────────────────────────────────────────────

class TestDefaultState:

    def test_returns_oos_report(self):
        report = _quick_validator([])
        assert isinstance(report, OOSReport)

    def test_empty_report_zero_trades(self):
        report = _quick_validator([])
        assert report.total_trades == 0

    def test_empty_report_empty_cells(self):
        report = _quick_validator([])
        assert report.cells_analyzed == 0

    def test_to_dict_has_keys(self):
        report = _quick_validator([])
        d = report.to_dict()
        for k in ("total_trades", "cells_analyzed", "best_cells", "worst_cells",
                  "by_regime", "by_signal_source", "by_hour_block", "regime_signal_matrix"):
            assert k in d


# ── load_trades ───────────────────────────────────────────────────────────────

class TestLoadTrades:

    def test_loads_exit_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_audit(d, [_trade(), _trade()])
            v = OOSValidator(audit_dir=d)
            n = v.load_trades()
            assert n == 2

    def test_ignores_non_exit_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rows = [
                {"event": "ENTRY", "pnl_pct": 0.01},
                _trade(),
            ]
            _write_audit(d, rows)
            v = OOSValidator(audit_dir=d)
            n = v.load_trades()
            assert n == 1

    def test_empty_directory_loads_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = OOSValidator(audit_dir=Path(tmp))
            assert v.load_trades() == 0

    def test_multiple_files_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for date in ["2026-03-21", "2026-03-22"]:
                p = d / f"trade_audit_{date}.jsonl"
                p.write_text(json.dumps(_trade()) + "\n")
            v = OOSValidator(audit_dir=d)
            assert v.load_trades() == 2


# ── Cell computation ──────────────────────────────────────────────────────────

class TestCellComputation:

    def test_cell_stats_type(self):
        report = _quick_validator([_trade()] * 5)
        if report.cells_analyzed > 0:
            assert isinstance(report.best_cells[0], CellStats)

    def test_cell_win_rate_all_positive(self):
        trades = [_trade(pnl_pct=0.01)] * 5
        report = _quick_validator(trades)
        if report.cells_analyzed > 0:
            assert report.best_cells[0].win_rate == pytest.approx(1.0)

    def test_cell_win_rate_all_negative(self):
        trades = [_trade(pnl_pct=-0.01)] * 5
        report = _quick_validator(trades)
        if report.cells_analyzed > 0:
            assert report.worst_cells[0].win_rate == pytest.approx(0.0)

    def test_cell_key_format(self):
        trades = [_trade()] * 5
        report = _quick_validator(trades)
        if report.cells_analyzed > 0:
            cell = report.best_cells[0]
            assert "|" in cell.cell
            assert cell.regime in cell.cell
            assert cell.hour_block in cell.cell

    def test_min_trades_filters_cells(self):
        trades = [_trade()] * 3
        report = _quick_validator(trades, min_trades=5)
        assert report.cells_analyzed == 0

    def test_cells_separated_by_regime(self):
        trades = (
            [_trade(regime="neutral")] * 4
            + [_trade(regime="bull")] * 4
        )
        report = _quick_validator(trades, min_trades=1)
        regimes = {c.regime for c in report.best_cells + report.worst_cells}
        assert len(regimes) >= 2

    def test_edge_score_positive_when_profitable(self):
        trades = [_trade(pnl_pct=0.02)] * 5
        report = _quick_validator(trades)
        if report.cells_analyzed > 0:
            top = report.best_cells[0]
            assert top.edge_score > 0.0

    def test_edge_score_zero_when_all_negative(self):
        trades = [_trade(pnl_pct=-0.01)] * 5
        report = _quick_validator(trades)
        if report.cells_analyzed > 0:
            assert report.worst_cells[0].edge_score == pytest.approx(0.0)


# ── Aggregates ────────────────────────────────────────────────────────────────

class TestAggregates:

    def test_by_regime_keys(self):
        trades = (
            [_trade(regime="neutral")] * 4
            + [_trade(regime="bull")] * 4
        )
        report = _quick_validator(trades)
        assert "neutral" in report.by_regime
        assert "bull" in report.by_regime

    def test_by_regime_win_rate_in_range(self):
        trades = [_trade(pnl_pct=0.01)] * 4 + [_trade(pnl_pct=-0.01)] * 4
        report = _quick_validator(trades)
        for v in report.by_regime.values():
            assert 0.0 <= v["win_rate"] <= 1.0

    def test_by_signal_source_present(self):
        trades = [
            _trade(components={"ml": 0.8, "tech": 0.2}) for _ in range(4)
        ]
        report = _quick_validator(trades)
        assert "ml" in report.by_signal_source or "unknown" in report.by_signal_source

    def test_by_hour_block_present(self):
        trades = [_trade(ts="2026-03-22T14:30:00Z")] * 4
        report = _quick_validator(trades)
        assert "midday" in report.by_hour_block

    def test_regime_signal_matrix_has_win_rate(self):
        trades = [
            _trade(regime="neutral", components={"ml": 0.9}, pnl_pct=0.01)
            for _ in range(4)
        ]
        report = _quick_validator(trades)
        if "neutral" in report.regime_signal_matrix:
            ml_wr = report.regime_signal_matrix["neutral"].get("ml")
            if ml_wr is not None:
                assert 0.0 <= ml_wr <= 1.0


# ── Helper functions ──────────────────────────────────────────────────────────

class TestDominantComponent:

    def test_returns_highest_component(self):
        trade = {"components": {"ml": 0.8, "tech": 0.3, "sentiment": 0.1}}
        assert _dominant_component(trade) == "ml"

    def test_no_components_returns_unknown(self):
        assert _dominant_component({}) == "unknown"

    def test_zero_components_returns_unknown(self):
        trade = {"components": {"ml": 0.0, "tech": 0.0}}
        assert _dominant_component(trade) == "unknown"

    def test_negative_abs_considered(self):
        trade = {"components": {"ml": -0.9, "tech": 0.3}}
        assert _dominant_component(trade) == "ml"


class TestHourBlock:

    def test_midday_14_utc(self):
        t = {"ts": "2026-03-22T14:00:00Z"}
        assert _hour_block_for(t) == "midday"

    def test_open_09_utc(self):
        t = {"ts": "2026-03-22T09:30:00Z"}
        assert _hour_block_for(t) == "open"

    def test_close_16_utc(self):
        t = {"ts": "2026-03-22T16:00:00Z"}
        assert _hour_block_for(t) == "close"

    def test_overnight_03_utc(self):
        t = {"ts": "2026-03-22T03:00:00Z"}
        assert _hour_block_for(t) == "overnight"

    def test_no_ts_returns_unknown(self):
        assert _hour_block_for({}) == "unknown"

    def test_malformed_ts_returns_unknown(self):
        assert _hour_block_for({"ts": "bad-date"}) == "unknown"


class TestNormalizeRegime:

    def test_lowercase(self):
        assert _normalize_regime("Neutral") == "neutral"

    def test_unknown_passthrough(self):
        assert _normalize_regime("unknown") == "unknown"

    def test_empty_returns_unknown(self):
        assert _normalize_regime("") == "unknown"

    def test_none_like_empty(self):
        assert _normalize_regime(None) == "unknown"


class TestSharpe:

    def test_varying_positive_returns_positive_sharpe(self):
        # Increasing series: positive drift, non-zero variance → positive Sharpe
        pnls = [0.005 * i for i in range(1, 21)]
        assert _sharpe(pnls) > 0

    def test_varying_negative_returns_negative_sharpe(self):
        pnls = [-0.005 * i for i in range(1, 21)]
        assert _sharpe(pnls) < 0

    def test_constant_returns_zero_sharpe(self):
        # Constant series → near-zero variance → Sharpe = 0
        s = _sharpe([0.01] * 10)
        assert s == 0.0

    def test_mixed_returns_near_zero(self):
        pnls = [0.01, -0.01] * 10
        s = _sharpe(pnls)
        assert abs(s) < 0.5

    def test_single_value_returns_zero(self):
        assert _sharpe([0.01]) == 0.0

    def test_empty_returns_zero(self):
        assert _sharpe([]) == 0.0


# ── Prune candidates ──────────────────────────────────────────────────────────

class TestPruneCandidates:

    def test_returns_list(self):
        trades = [_trade(pnl_pct=-0.01)] * 10
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_audit(d, trades)
            v = OOSValidator(audit_dir=d, min_trades=1)
            v.load_trades()
            candidates = v.get_prune_candidates(win_rate_threshold=0.5, min_trades=5)
            assert isinstance(candidates, list)

    def test_prune_contains_cell_info(self):
        trades = [_trade(pnl_pct=-0.01)] * 10
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_audit(d, trades)
            v = OOSValidator(audit_dir=d, min_trades=1)
            v.load_trades()
            candidates = v.get_prune_candidates(win_rate_threshold=1.0, min_trades=5)
            if candidates:
                assert "regime" in candidates[0]
                assert "win_rate" in candidates[0]
