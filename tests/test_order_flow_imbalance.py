"""
tests/test_order_flow_imbalance.py — Unit tests for data/order_flow_imbalance.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.order_flow_imbalance import (
    OrderFlowSignal,
    compute_ofi,
    get_ofi_signal,
)


# ── compute_ofi helper ────────────────────────────────────────────────────────

def _make_df(closes, volumes=None):
    n = len(closes)
    vols = volumes if volumes is not None else [10000.0] * n
    return pd.DataFrame({"Close": closes, "Volume": vols})


class TestComputeOfi:
    def test_none_df_returns_zero(self):
        assert compute_ofi(None) == 0.0

    def test_empty_df_returns_zero(self):
        assert compute_ofi(pd.DataFrame()) == 0.0

    def test_too_few_rows_returns_zero(self):
        df = _make_df([100.0, 101.0])
        assert compute_ofi(df) == 0.0

    def test_all_up_ticks_returns_one(self):
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        df = _make_df(closes)
        result = compute_ofi(df, window=5)
        assert result == pytest.approx(1.0)

    def test_all_down_ticks_returns_minus_one(self):
        closes = [105.0, 104.0, 103.0, 102.0, 101.0, 100.0]
        df = _make_df(closes)
        result = compute_ofi(df, window=5)
        assert result == pytest.approx(-1.0)

    def test_balanced_ticks_near_zero(self):
        # Alternating up/down of equal volume: 5 transitions from 6 bars
        # Pattern: [100, 101, 100, 101, 100, 101] gives 3 up + 2 down (window=5)
        # Use window=4 with 5 bars → [101, 100, 101, 100, 101]:
        # i=1: 100 < 101 → down; i=2: 101 > 100 → up; i=3: 100 < 101 → down; i=4: 101 > 100 → up
        # 2 up + 2 down → OFI = 0.0
        closes = [101.0, 100.0, 101.0, 100.0, 101.0]
        df = _make_df(closes, volumes=[10000.0] * 5)
        result = compute_ofi(df, window=4)
        assert abs(result) < 0.01  # exactly zero with equal up/down count

    def test_neutral_ticks_return_zero(self):
        closes = [100.0] * 10
        df = _make_df(closes)
        # All neutral: buy=sell, should be 0
        result = compute_ofi(df, window=9)
        assert result == pytest.approx(0.0)

    def test_result_bounded_minus_one_to_one(self):
        for closes in [
            [100.0] * 5,
            [100.0, 102.0, 104.0, 103.0, 101.0, 100.0],
            list(range(100, 120)),
        ]:
            df = _make_df(closes)
            result = compute_ofi(df)
            assert -1.0 <= result <= 1.0

    def test_low_volume_returns_zero(self):
        # Total volume below min
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        df = _make_df(closes, volumes=[1.0] * 5)  # very low volume
        result = compute_ofi(df, window=4)
        assert result == 0.0

    def test_missing_volume_uses_ones(self):
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        # No Volume column — should still compute (but min_vol check may block)
        # With OFI_MIN_VOLUME = 1000 and volume=1 per bar, 5 bars → total=5 < 1000
        # Just check it doesn't raise
        result = compute_ofi(df, window=4)
        assert isinstance(result, float)

    def test_weighted_by_volume(self):
        # 1 big up-tick vs many small down-ticks
        closes = [100.0, 200.0, 199.0, 198.0, 197.0, 196.0]  # 1 up, 4 down
        volumes = [1000.0, 50000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        df = _make_df(closes, volumes)
        result = compute_ofi(df, window=5)
        # Big up volume dominates → positive OFI
        assert result > 0.0


# ── OrderFlowSignal ───────────────────────────────────────────────────────────

class TestOrderFlowSignal:
    def test_get_signal_returns_float(self):
        ofi = OrderFlowSignal()
        closes = list(range(100, 125))
        df = _make_df(closes)
        result = ofi.get_signal("AAPL", df)
        assert isinstance(result, float)

    def test_get_signal_bounded(self):
        ofi = OrderFlowSignal()
        closes = list(range(100, 125))
        df = _make_df(closes)
        result = ofi.get_signal("AAPL", df)
        assert -1.0 <= result <= 1.0

    def test_ema_smoothing_applied(self):
        ofi = OrderFlowSignal()
        # First call with pure up-ticks → raw OFI = +1.0
        closes_up = [100.0 + i for i in range(22)]
        df_up = _make_df(closes_up)
        v1 = ofi.get_signal("AAPL", df_up)
        # Smooth: alpha=0.30, prev=initial → should be less than 1.0 if prev was 0
        assert -1.0 <= v1 <= 1.0

    def test_cache_returns_same_value_quickly(self):
        ofi = OrderFlowSignal()
        closes = list(range(100, 125))
        df = _make_df(closes)
        v1 = ofi.get_signal("AAPL", df)
        v2 = ofi.get_signal("AAPL", df)  # should hit cache
        assert v1 == pytest.approx(v2)

    def test_different_symbols_independent(self):
        ofi = OrderFlowSignal()
        closes_up = [100.0 + i for i in range(22)]
        closes_down = [200.0 - i for i in range(22)]
        df_up = _make_df(closes_up)
        df_down = _make_df(closes_down)
        v_up = ofi.get_signal("AAPL", df_up)
        v_down = ofi.get_signal("MSFT", df_down)
        assert v_up > v_down

    def test_record_tick_buy(self):
        ofi = OrderFlowSignal()
        ofi.record_tick("AAPL", 150.0, 5000.0, "buy")
        assert len(ofi._raw_buf["AAPL"]) == 1

    def test_record_tick_buffer_capped(self):
        ofi = OrderFlowSignal()
        for i in range(250):
            ofi.record_tick("AAPL", 150.0, 100.0, "buy")
        assert len(ofi._raw_buf["AAPL"]) == 200

    def test_get_tick_ofi_insufficient_ticks(self):
        ofi = OrderFlowSignal()
        result = ofi.get_tick_ofi("AAPL")
        assert result == 0.0

    def test_get_tick_ofi_all_buys(self):
        ofi = OrderFlowSignal()
        for i in range(20):
            ofi.record_tick("AAPL", 150.0, 10000.0, "buy")
        result = ofi.get_tick_ofi("AAPL")
        assert result == pytest.approx(1.0)

    def test_get_tick_ofi_all_sells(self):
        ofi = OrderFlowSignal()
        for i in range(20):
            ofi.record_tick("AAPL", 150.0, 10000.0, "sell")
        result = ofi.get_tick_ofi("AAPL")
        assert result == pytest.approx(-1.0)

    def test_get_summary_returns_dict(self):
        ofi = OrderFlowSignal()
        closes = list(range(100, 125))
        df = _make_df(closes)
        ofi.get_signal("AAPL", df)
        summary = ofi.get_summary()
        assert isinstance(summary, dict)
        assert "AAPL" in summary

    def test_disabled_returns_zero(self, monkeypatch):
        import data.order_flow_imbalance as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "OFI_ENABLED" else mod._DEF.get(k))
        ofi = OrderFlowSignal()
        closes = list(range(100, 125))
        df = _make_df(closes)
        result = ofi.get_signal("AAPL", df)
        assert result == 0.0


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_ofi_signal_returns_instance(self):
        ofi = get_ofi_signal()
        assert isinstance(ofi, OrderFlowSignal)

    def test_singleton_same_object(self):
        a = get_ofi_signal()
        b = get_ofi_signal()
        assert a is b
