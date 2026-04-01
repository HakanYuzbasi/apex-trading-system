"""tests/test_universe_selector.py — Dynamic equity universe selector tests."""
from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from monitoring.universe_selector import (
    UniverseSelector,
    SymbolScore,
    _compute_sharpe,
    _strip_prefix,
    _load_exit_records,
    _NEUTRAL_SCORE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_audit_dir(base: Path, user: str = "admin") -> Path:
    d = base / "users" / user / "audit"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_exit(audit_dir: Path, symbol: str, pnl: float, ts: str = None) -> None:
    if ts is None:
        ts = datetime.now(timezone.utc).isoformat()
    record = {"action": "EXIT", "symbol": symbol, "pnl_pct": pnl, "timestamp": ts}
    path = audit_dir / "trade_audit_test.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── _strip_prefix ─────────────────────────────────────────────────────────────

class TestStripPrefix:
    def test_removes_crypto_prefix(self):
        assert _strip_prefix("CRYPTO:BTC/USD") == "BTC/USD"

    def test_removes_fx_prefix(self):
        assert _strip_prefix("FX:EUR/USD") == "EUR/USD"

    def test_no_prefix_unchanged(self):
        assert _strip_prefix("AAPL") == "AAPL"

    def test_unknown_prefix_unchanged(self):
        assert _strip_prefix("NYSE:AAPL") == "NYSE:AAPL"


# ── _compute_sharpe ───────────────────────────────────────────────────────────

class TestComputeSharpe:
    def test_too_few_returns_zero(self):
        assert _compute_sharpe([0.01, 0.02]) == 0.0

    def test_positive_mean_positive_sharpe(self):
        pnls = [0.01] * 20
        assert _compute_sharpe(pnls) > 0

    def test_negative_mean_negative_sharpe(self):
        pnls = [-0.01] * 20
        assert _compute_sharpe(pnls) < 0

    def test_mixed_returns_finite(self):
        pnls = [0.01, -0.005] * 10
        s = _compute_sharpe(pnls)
        assert math.isfinite(s)


# ── SymbolScore ───────────────────────────────────────────────────────────────

class TestSymbolScore:
    def test_to_dict_keys(self):
        s = SymbolScore("AAPL", 0.75, 0.60, 1.2, 15)
        d = s.to_dict()
        for key in ("symbol", "score", "win_rate", "sharpe", "trade_count"):
            assert key in d

    def test_values_rounded(self):
        s = SymbolScore("MSFT", 0.123456, 0.654321, 1.23456, 10)
        d = s.to_dict()
        assert d["score"] == round(0.123456, 4)
        assert d["win_rate"] == round(0.654321, 4)


# ── UniverseSelector scoring ──────────────────────────────────────────────────

class TestCompositeScoring:
    def _sel(self):
        return UniverseSelector(lookback_days=30, min_trades=3)

    def test_all_wins_high_score(self):
        sel = self._sel()
        score = sel._composite_score(win_rate=1.0, sharpe=3.0, n_trades=30)
        assert score > 0.8

    def test_all_losses_low_score(self):
        sel = self._sel()
        score = sel._composite_score(win_rate=0.0, sharpe=-3.0, n_trades=30)
        assert score < 0.3

    def test_neutral_score_midrange(self):
        sel = self._sel()
        score = sel._composite_score(win_rate=0.50, sharpe=0.0, n_trades=20)
        assert 0.3 < score < 0.7

    def test_score_bounded_0_1(self):
        sel = self._sel()
        for wr, sh, n in [(0.0, -10.0, 50), (1.0, 10.0, 50), (0.5, 0.0, 1)]:
            s = sel._composite_score(wr, sh, n)
            assert 0.0 <= s <= 1.0

    def test_few_trades_lowers_activity_component(self):
        sel = self._sel()
        few = sel._composite_score(0.7, 2.0, 3)
        many = sel._composite_score(0.7, 2.0, 30)
        assert many > few


# ── get_score fallback ────────────────────────────────────────────────────────

class TestGetScore:
    def test_unknown_symbol_returns_neutral(self):
        sel = UniverseSelector()
        assert sel.get_score("UNKNOWN_XYZ") == _NEUTRAL_SCORE

    def test_crypto_prefix_stripped_for_lookup(self):
        sel = UniverseSelector()
        sel._scores["BTC/USD"] = SymbolScore("BTC/USD", 0.80, 0.70, 2.0, 20)
        assert sel.get_score("CRYPTO:BTC/USD") == 0.80

    def test_scored_symbol_returns_correct_score(self):
        sel = UniverseSelector()
        sel._scores["AAPL"] = SymbolScore("AAPL", 0.72, 0.65, 1.5, 15)
        assert abs(sel.get_score("AAPL") - 0.72) < 1e-9


# ── get_active_symbols / get_skipped_symbols ──────────────────────────────────

class TestActiveSkipped:
    def _sel_with_scores(self) -> UniverseSelector:
        sel = UniverseSelector()
        sel._scores = {
            "AAPL": SymbolScore("AAPL", 0.80, 0.70, 2.0, 20),
            "MSFT": SymbolScore("MSFT", 0.60, 0.55, 1.2, 15),
            "GME":  SymbolScore("GME",  0.20, 0.20, -0.5, 8),
        }
        return sel

    def test_active_above_threshold(self):
        sel = self._sel_with_scores()
        active = sel.get_active_symbols(["AAPL", "MSFT", "GME"], min_score=0.30)
        assert "AAPL" in active
        assert "MSFT" in active
        assert "GME" not in active

    def test_skipped_below_threshold(self):
        sel = self._sel_with_scores()
        skipped = sel.get_skipped_symbols(["AAPL", "MSFT", "GME"], min_score=0.30)
        assert "GME" in skipped
        assert "AAPL" not in skipped

    def test_active_sorted_by_score_desc(self):
        sel = self._sel_with_scores()
        active = sel.get_active_symbols(["AAPL", "MSFT", "GME"], min_score=0.0)
        assert active[0] == "AAPL"
        assert active[1] == "MSFT"

    def test_unknown_symbol_passes_neutral(self):
        sel = self._sel_with_scores()
        # NVDA not scored → neutral 0.50 → passes min_score=0.30
        active = sel.get_active_symbols(["NVDA"], min_score=0.30)
        assert "NVDA" in active


# ── refresh from JSONL ────────────────────────────────────────────────────────

class TestRefresh:
    def test_empty_dir_scores_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            sel = UniverseSelector(min_trades=3)
            count = sel.refresh(Path(d))
            assert count == 0

    def test_loads_exit_records(self):
        with tempfile.TemporaryDirectory() as d:
            audit_dir = _make_audit_dir(Path(d))
            for i in range(10):
                _write_exit(audit_dir, "AAPL", 0.01 if i < 7 else -0.005)
            sel = UniverseSelector(min_trades=3)
            count = sel.refresh(Path(d))
            assert count == 1
            assert sel.get_score("AAPL") > _NEUTRAL_SCORE  # 70% WR → above neutral

    def test_ignores_entry_records(self):
        with tempfile.TemporaryDirectory() as d:
            audit_dir = _make_audit_dir(Path(d))
            ts = datetime.now(timezone.utc).isoformat()
            with open(audit_dir / "trade_audit_test.jsonl", "w") as f:
                f.write(json.dumps({"action": "ENTRY", "symbol": "AAPL", "timestamp": ts}) + "\n")
            sel = UniverseSelector(min_trades=1)
            sel.refresh(Path(d))
            assert sel.get_score("AAPL") == _NEUTRAL_SCORE

    def test_ignores_old_records(self):
        with tempfile.TemporaryDirectory() as d:
            audit_dir = _make_audit_dir(Path(d))
            old_ts = "2020-01-01T00:00:00+00:00"
            _write_exit(audit_dir, "AAPL", 0.01, ts=old_ts)
            sel = UniverseSelector(lookback_days=30, min_trades=1)
            sel.refresh(Path(d))
            assert sel.get_score("AAPL") == _NEUTRAL_SCORE

    def test_multiple_symbols_scored_separately(self):
        with tempfile.TemporaryDirectory() as d:
            audit_dir = _make_audit_dir(Path(d))
            for _ in range(10):
                _write_exit(audit_dir, "AAPL", 0.015)  # profitable
                _write_exit(audit_dir, "GME",  -0.02)   # losing
            sel = UniverseSelector(min_trades=3)
            sel.refresh(Path(d))
            assert sel.get_score("AAPL") > sel.get_score("GME")

    def test_last_refresh_set_after_refresh(self):
        with tempfile.TemporaryDirectory() as d:
            sel = UniverseSelector()
            assert sel._last_refresh is None
            sel.refresh(Path(d))
            assert sel._last_refresh is not None

    def test_get_report_structure(self):
        with tempfile.TemporaryDirectory() as d:
            audit_dir = _make_audit_dir(Path(d))
            for _ in range(10):
                _write_exit(audit_dir, "AAPL", 0.01)
            sel = UniverseSelector(min_trades=3)
            sel.refresh(Path(d))
            report = sel.get_report()
            assert "scored_count" in report
            assert "symbols" in report
            assert "lookback_days" in report


# ── get_priority_scores ───────────────────────────────────────────────────────

class TestPriorityScores:
    def test_returns_dict_of_floats(self):
        sel = UniverseSelector()
        sel._scores = {
            "AAPL": SymbolScore("AAPL", 0.75, 0.65, 1.5, 15),
        }
        ps = sel.get_priority_scores()
        assert isinstance(ps, dict)
        assert abs(ps["AAPL"] - 0.75) < 1e-9

    def test_empty_when_not_refreshed(self):
        sel = UniverseSelector()
        assert sel.get_priority_scores() == {}
