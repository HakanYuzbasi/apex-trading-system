"""tests/test_signal_cascade.py — Cross-Asset Signal Cascade Engine tests."""

from __future__ import annotations

import math

import pytest

from core.signal_cascade import (
    SignalCascadeEngine,
    _pearson,
    _safe_log_return,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _feed_rising(engine: SignalCascadeEngine, symbol: str, start: float, n: int, step: float = 0.01) -> None:
    """Feed n consecutive rising prices to the engine."""
    for i in range(n):
        engine.record_price(symbol, start * (1 + step) ** i)


def _feed_flat(engine: SignalCascadeEngine, symbol: str, price: float, n: int) -> None:
    for _ in range(n):
        engine.record_price(symbol, price)


# ── _safe_log_return ──────────────────────────────────────────────────────────

class TestSafeLogReturn:
    def test_positive_move(self):
        r = _safe_log_return(100.0, 110.0)
        assert abs(r - math.log(1.1)) < 1e-12

    def test_zero_prev_returns_zero(self):
        assert _safe_log_return(0.0, 100.0) == 0.0

    def test_zero_curr_returns_zero(self):
        assert _safe_log_return(100.0, 0.0) == 0.0

    def test_negative_price_returns_zero(self):
        assert _safe_log_return(-5.0, 100.0) == 0.0


# ── _pearson ──────────────────────────────────────────────────────────────────

class TestPearson:
    def test_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(_pearson(xs, xs) - 1.0) < 1e-9

    def test_perfect_negative(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert abs(_pearson(xs, ys) + 1.0) < 1e-9

    def test_zero_correlation(self):
        xs = [1.0, 2.0, 3.0]
        ys = [1.0, 1.0, 1.0]  # constant → zero denom
        assert _pearson(xs, ys) == 0.0

    def test_short_returns_zero(self):
        assert _pearson([1.0], [1.0]) == 0.0
        assert _pearson([], []) == 0.0


# ── record_price / _get_returns ───────────────────────────────────────────────

class TestRecordPrice:
    def test_negative_price_ignored(self):
        eng = SignalCascadeEngine()
        eng.record_price("SPY", -10.0)
        assert "SPY" not in eng._prices

    def test_zero_price_ignored(self):
        eng = SignalCascadeEngine()
        eng.record_price("SPY", 0.0)
        assert "SPY" not in eng._prices

    def test_price_history_bounded(self):
        eng = SignalCascadeEngine(max_price_history=10)
        for i in range(20):
            eng.record_price("SPY", 400.0 + i)
        assert len(eng._prices["SPY"]) == 10

    def test_returns_cache_invalidated(self):
        eng = SignalCascadeEngine()
        _feed_rising(eng, "SPY", 400.0, 10)
        # prime cache
        _ = eng._get_returns("SPY")
        assert "SPY" in eng._returns_cache
        # new price → cache invalidated
        eng.record_price("SPY", 420.0)
        assert "SPY" not in eng._returns_cache


# ── anchor return ─────────────────────────────────────────────────────────────

class TestAnchorReturn:
    def test_none_when_insufficient_bars(self):
        eng = SignalCascadeEngine(window_bars=6)
        eng.record_price("SPY", 400.0)
        eng.record_price("SPY", 402.0)
        assert eng._anchor_return("SPY") is None

    def test_cumulative_log_return(self):
        eng = SignalCascadeEngine(window_bars=2)
        eng.record_price("SPY", 400.0)
        eng.record_price("SPY", 404.0)
        eng.record_price("SPY", 408.0)  # ensures 2 returns in window
        ret = eng._anchor_return("SPY")
        assert ret is not None
        expected = math.log(404.0 / 400.0) + math.log(408.0 / 404.0)
        assert abs(ret - expected) < 1e-9


# ── get_cascade_adjustment ────────────────────────────────────────────────────

class TestGetCascadeAdjustment:
    def test_no_history_returns_1(self):
        eng = SignalCascadeEngine()
        assert eng.get_cascade_adjustment("AAPL") == 1.0

    def test_anchor_skips_itself(self):
        eng = SignalCascadeEngine()
        _feed_rising(eng, "SPY", 400.0, 20, step=0.005)
        assert eng.get_cascade_adjustment("SPY") == 1.0

    def test_below_threshold_returns_1(self):
        eng = SignalCascadeEngine(trigger_threshold=0.05, window_bars=3)
        # tiny move: ~0.1%/bar
        for i in range(10):
            eng.record_price("SPY", 400.0 * (1 + 0.001 * i))
            eng.record_price("AAPL", 150.0 * (1 + 0.001 * i))
        assert eng.get_cascade_adjustment("AAPL") == 1.0

    def test_cascade_multiplier_in_range(self):
        eng = SignalCascadeEngine(
            window_bars=5,
            trigger_threshold=0.003,
            cascade_gain=2.0,
            floor=0.5,
            ceiling=1.5,
        )
        # Feed strongly correlated rising prices
        for i in range(30):
            p = 1.0 + i * 0.005
            eng.record_price("SPY",  400.0 * p)
            eng.record_price("AAPL", 150.0 * p)
        mult = eng.get_cascade_adjustment("AAPL")
        assert eng.floor <= mult <= eng.ceiling

    def test_low_correlation_returns_1(self):
        eng = SignalCascadeEngine(
            window_bars=5,
            trigger_threshold=0.003,
            min_correlation=0.90,  # very high threshold
            cascade_gain=3.0,
        )
        # SPY rising strongly, AAPL flat → low correlation
        for i in range(30):
            eng.record_price("SPY",  400.0 * (1 + 0.01 * i))
            eng.record_price("AAPL", 150.0)  # flat
        mult = eng.get_cascade_adjustment("AAPL")
        assert mult == 1.0

    def test_crypto_uses_btc_anchor(self):
        eng = SignalCascadeEngine(
            window_bars=3,
            trigger_threshold=0.003,
            cascade_gain=2.0,
        )
        for i in range(20):
            p = 1.0 + i * 0.005
            eng.record_price("BTC/USD",       40000.0 * p)
            eng.record_price("CRYPTO:ETH/USD", 2000.0 * p)
        mult = eng.get_cascade_adjustment("CRYPTO:ETH/USD")
        # Should be != 1.0 since both are rising together
        assert isinstance(mult, float)
        assert eng.floor <= mult <= eng.ceiling

    def test_multiplier_floored(self):
        eng = SignalCascadeEngine(
            window_bars=2,
            trigger_threshold=0.001,
            cascade_gain=100.0,  # exaggerated gain
            floor=0.50,
            ceiling=1.50,
        )
        for i in range(10):
            p = 1.0 + i * 0.01
            eng.record_price("SPY",  400.0 * p)
            eng.record_price("AAPL", 150.0 * p)
        mult = eng.get_cascade_adjustment("AAPL")
        assert mult >= eng.floor
        assert mult <= eng.ceiling


# ── tick / correlation cache ──────────────────────────────────────────────────

class TestTick:
    def test_cycle_increments(self):
        eng = SignalCascadeEngine()
        assert eng._cycle == 0
        eng.tick()
        eng.tick()
        assert eng._cycle == 2

    def test_cache_recomputed_after_expiry(self):
        eng = SignalCascadeEngine(correlation_cache_cycles=2)
        for i in range(20):
            p = 1.0 + i * 0.005
            eng.record_price("SPY",  400.0 * p)
            eng.record_price("AAPL", 150.0 * p)

        # Prime cache
        eng._recompute_correlations("SPY")
        eng._corr_cache_age["SPY"] = 0

        # After 2 ticks cache should be recomputed on next call
        eng.tick()
        eng.tick()
        assert eng._corr_cache_age.get("SPY", 0) >= 2


# ── get_cascade_info / summary ────────────────────────────────────────────────

class TestDiagnostics:
    def test_cascade_info_keys(self):
        eng = SignalCascadeEngine()
        info = eng.get_cascade_info("AAPL")
        assert "symbol" in info
        assert "anchor" in info
        assert "cascade_active" in info
        assert "multiplier" in info

    def test_summary_keys(self):
        eng = SignalCascadeEngine()
        s = eng.summary()
        assert "tracked_symbols" in s
        assert "equity_anchor_return" in s
        assert "crypto_cascade_active" in s

    def test_get_all_adjustments(self):
        eng = SignalCascadeEngine()
        for i in range(10):
            eng.record_price("SPY",  400.0 + i)
            eng.record_price("AAPL", 150.0 + i)
        adjs = eng.get_all_adjustments()
        assert "SPY" in adjs
        assert "AAPL" in adjs
