"""Tests for LiquiditySizer."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from risk.liquidity_sizer import ConcentrationHeat, LiquiditySizer


# ── helpers ───────────────────────────────────────────────────────────────────

def _sizer(**kwargs) -> LiquiditySizer:
    return LiquiditySizer(**kwargs)


def _fill_history(s: LiquiditySizer, symbol: str, spread: float, n: int = 10):
    for _ in range(n):
        s.record_spread(symbol, spread)


# ── record_spread / get_baseline_spread ──────────────────────────────────────

class TestRecordSpread:
    def test_zero_spread_ignored(self):
        s = _sizer()
        s.record_spread("AAPL", 0.0)
        assert s.get_baseline_spread("AAPL") is None

    def test_negative_spread_ignored(self):
        s = _sizer()
        s.record_spread("AAPL", -1.0)
        assert s.get_baseline_spread("AAPL") is None

    def test_baseline_is_median(self):
        s = _sizer()
        for v in [2.0, 4.0, 6.0]:
            s.record_spread("AAPL", v)
        assert s.get_baseline_spread("AAPL") == pytest.approx(4.0)

    def test_rolling_window_capped(self):
        s = _sizer(spread_window=5)
        for v in range(10):
            s.record_spread("X", float(v + 1))
        # Only last 5 values kept: 6,7,8,9,10 → median = 8
        assert s.get_baseline_spread("X") == pytest.approx(8.0)

    def test_unknown_symbol_returns_none(self):
        s = _sizer()
        assert s.get_baseline_spread("UNKNOWN") is None


# ── get_liquidity_multiplier ──────────────────────────────────────────────────

class TestGetLiquidityMultiplier:
    def test_no_history_returns_one(self):
        s = _sizer()
        assert s.get_liquidity_multiplier("AAPL", 5.0) == pytest.approx(1.0)

    def test_insufficient_history_returns_one(self):
        s = _sizer()
        for _ in range(4):  # < 5 required
            s.record_spread("AAPL", 3.0)
        assert s.get_liquidity_multiplier("AAPL", 6.0) == pytest.approx(1.0)

    def test_spread_below_baseline_returns_one(self):
        s = _sizer()
        _fill_history(s, "AAPL", 5.0)
        result = s.get_liquidity_multiplier("AAPL", 4.0)
        assert result == pytest.approx(1.0)

    def test_spread_equal_to_baseline_returns_one(self):
        s = _sizer()
        _fill_history(s, "AAPL", 5.0)
        result = s.get_liquidity_multiplier("AAPL", 5.0)
        assert result == pytest.approx(1.0)

    def test_double_spread_reduces_multiplier(self):
        """spread_ratio=2 → mult = 1/(1 + 1.5*1) = 0.40"""
        s = _sizer(penalty_k=1.5)
        _fill_history(s, "AAPL", 5.0)
        result = s.get_liquidity_multiplier("AAPL", 10.0)
        assert result == pytest.approx(0.40, abs=0.01)

    def test_triple_spread_hits_floor(self):
        """Very wide spread should hit mult_floor=0.30"""
        s = _sizer(penalty_k=1.5, mult_floor=0.30)
        _fill_history(s, "AAPL", 2.0)
        result = s.get_liquidity_multiplier("AAPL", 200.0)   # 100× baseline
        assert result == pytest.approx(0.30)

    def test_multiplier_in_valid_range(self):
        s = _sizer(mult_floor=0.30)
        _fill_history(s, "SPY", 1.0)
        for spread in [0.5, 1.0, 2.0, 5.0, 20.0]:
            m = s.get_liquidity_multiplier("SPY", spread)
            assert 0.30 <= m <= 1.0

    def test_zero_current_spread_returns_one(self):
        s = _sizer()
        _fill_history(s, "AAPL", 5.0)
        assert s.get_liquidity_multiplier("AAPL", 0.0) == pytest.approx(1.0)


# ── get_concentration_heat ───────────────────────────────────────────────────

class TestGetConcentrationHeat:
    def test_empty_positions_returns_zeroes(self):
        s = _sizer()
        h = s.get_concentration_heat({})
        assert h.top5_share == pytest.approx(0.0)
        assert h.concentration_breached is False

    def test_single_position_full_concentration(self):
        s = _sizer()
        h = s.get_concentration_heat({"AAPL": 10000.0})
        assert h.top5_share == pytest.approx(1.0)
        assert h.top1_share == pytest.approx(1.0)

    def test_equal_positions_concentration(self):
        s = _sizer()
        positions = {f"S{i}": 1000.0 for i in range(10)}
        h = s.get_concentration_heat(positions)
        # top5 = 5/10 = 0.50 < default cap 0.65 → not breached
        assert h.top5_share == pytest.approx(0.50)
        assert h.concentration_breached is False

    def test_concentration_breached_when_top5_exceeds_cap(self):
        s = _sizer(concentration_cap=0.65)
        # 5 symbols hold 90% of capital
        positions = {f"BIG{i}": 18000.0 for i in range(5)}
        positions.update({f"SMALL{i}": 1000.0 for i in range(5)})
        h = s.get_concentration_heat(positions)
        assert h.concentration_breached is True

    def test_top5_symbols_sorted_by_weight(self):
        s = _sizer()
        positions = {"A": 5000.0, "B": 3000.0, "C": 2000.0, "D": 1000.0}
        h = s.get_concentration_heat(positions)
        assert h.top5_symbols[0] == "A"

    def test_negative_notional_treated_as_abs(self):
        s = _sizer()
        positions = {"LONG": 5000.0, "SHORT": -3000.0}
        h = s.get_concentration_heat(positions)
        total = 5000 + 3000
        assert h.top1_share == pytest.approx(5000 / total, abs=0.001)

    def test_by_symbol_weights_sum_to_one(self):
        s = _sizer()
        positions = {"A": 4000.0, "B": 3000.0, "C": 3000.0}
        h = s.get_concentration_heat(positions)
        assert sum(h.by_symbol.values()) == pytest.approx(1.0, abs=0.001)

    def test_to_dict_keys(self):
        s = _sizer()
        h = s.get_concentration_heat({"X": 1000.0})
        d = h.to_dict()
        for k in ("top5_share", "top1_share", "top5_symbols", "concentration_breached", "by_symbol"):
            assert k in d


# ── get_stress_exposure ───────────────────────────────────────────────────────

class TestGetStressExposure:
    def test_empty_returns_zeros(self):
        s = _sizer()
        r = s.get_stress_exposure({})
        assert r["total_loss"] == pytest.approx(0.0)

    def test_equity_only_loss(self):
        s = _sizer()
        positions = {"AAPL": 10000.0, "MSFT": 5000.0}
        r = s.get_stress_exposure(positions, scenario_drop_pct=0.05)
        assert r["equity_loss"] == pytest.approx(750.0)
        assert r["crypto_loss"] == pytest.approx(0.0)

    def test_crypto_only_loss(self):
        s = _sizer()
        positions = {"BTC": 20000.0}
        r = s.get_stress_exposure(
            positions,
            scenario_drop_pct=0.10,
            crypto_symbols=["BTC"],
        )
        assert r["crypto_loss"] == pytest.approx(2000.0)
        assert r["equity_loss"] == pytest.approx(0.0)

    def test_mixed_portfolio_total_loss(self):
        s = _sizer()
        positions = {"AAPL": 10000.0, "ETH": 5000.0}
        r = s.get_stress_exposure(
            positions,
            scenario_drop_pct=0.10,
            crypto_symbols=["ETH"],
        )
        assert r["total_loss"] == pytest.approx(1500.0)

    def test_short_positions_excluded(self):
        s = _sizer()
        positions = {"AAPL": 10000.0, "SPY": -5000.0}  # short SPY
        r = s.get_stress_exposure(positions, scenario_drop_pct=0.10)
        # Only long AAPL contributes
        assert r["equity_loss"] == pytest.approx(1000.0)

    def test_scenario_drop_pct_negative_handled(self):
        s = _sizer()
        positions = {"AAPL": 10000.0}
        r = s.get_stress_exposure(positions, scenario_drop_pct=-0.05)
        assert r["equity_loss"] == pytest.approx(500.0)  # abs applied


# ── get_report ─────────────────────────────────────────────────────────────────

class TestGetReport:
    def test_report_keys(self):
        s = _sizer()
        r = s.get_report({"AAPL": 10000.0, "MSFT": 5000.0})
        for k in ("concentration", "stress_5pct", "spread_baseline_coverage", "total_positions"):
            assert k in r

    def test_spread_baseline_coverage_counts_symbols_with_history(self):
        s = _sizer()
        _fill_history(s, "AAPL", 5.0)
        _fill_history(s, "MSFT", 3.0)
        r = s.get_report({"AAPL": 5000.0, "MSFT": 3000.0, "NVDA": 2000.0})
        assert r["spread_baseline_coverage"] == 2


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_spread_history_survives_save_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            s1 = LiquiditySizer(data_dir=d)
            _fill_history(s1, "AAPL", 5.0)
            s1.save()

            s2 = LiquiditySizer(data_dir=d)
            assert s2.get_baseline_spread("AAPL") == pytest.approx(5.0)

    def test_multiplier_works_after_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            s1 = LiquiditySizer(data_dir=d)
            _fill_history(s1, "AAPL", 5.0)
            s1.save()

            s2 = LiquiditySizer(data_dir=d)
            # Baseline restored → wide spread should reduce mult
            m = s2.get_liquidity_multiplier("AAPL", 10.0)
            assert m < 1.0
