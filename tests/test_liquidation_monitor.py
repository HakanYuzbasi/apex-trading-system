"""
tests/test_liquidation_monitor.py — Unit tests for monitoring/liquidation_monitor.py
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from monitoring.liquidation_monitor import (
    LiquidationMonitor,
    LiquidationState,
    funding_rate_signal,
    get_liquidation_monitor,
    oi_velocity_signal,
    price_oi_divergence_signal,
)


# ── funding_rate_signal ───────────────────────────────────────────────────────

class TestFundingRateSignal:
    def test_high_positive_funding_bearish(self):
        # 0.15% funding with 0.10% threshold → bearish
        sig = funding_rate_signal(0.0015, 0.001)
        assert sig < 0.0

    def test_high_negative_funding_slightly_bullish(self):
        # Shorts overleveraged → capitulation → slightly positive
        sig = funding_rate_signal(-0.002, 0.001)
        assert sig > 0.0

    def test_zero_funding_returns_zero(self):
        assert funding_rate_signal(0.0, 0.001) == 0.0

    def test_output_bounded(self):
        sig = funding_rate_signal(0.1, 0.001)  # extreme
        assert -1.0 <= sig <= 1.0

    def test_at_threshold_moderate_bearish(self):
        sig = funding_rate_signal(0.001, 0.001)  # exactly at threshold
        assert sig == pytest.approx(-1.0)

    def test_below_threshold_partial_signal(self):
        sig = funding_rate_signal(0.0005, 0.001)  # half threshold
        assert -1.0 < sig < 0.0

    def test_negative_capped_at_positive_half(self):
        sig = funding_rate_signal(-0.01, 0.001)  # very negative
        assert sig <= 0.5


# ── oi_velocity_signal ────────────────────────────────────────────────────────

class TestOIVelocitySignal:
    def test_insufficient_data_returns_zero(self):
        assert oi_velocity_signal([1000.0], 0.05, 0.10) == 0.0
        assert oi_velocity_signal([], 0.05, 0.10) == 0.0

    def test_critical_drop_returns_minus_one(self):
        # OI drops 15% → critical
        oi = [1000.0, 950.0, 900.0, 850.0]
        sig = oi_velocity_signal(oi, 0.05, 0.10)
        assert sig == pytest.approx(-1.0)

    def test_warning_drop_returns_partial(self):
        # OI drops 7% → warning
        oi = [1000.0, 970.0, 950.0, 930.0]
        sig = oi_velocity_signal(oi, 0.05, 0.10)
        assert -1.0 < sig < -0.4

    def test_flat_oi_returns_zero(self):
        oi = [1000.0, 1001.0, 999.0, 1000.0]
        sig = oi_velocity_signal(oi, 0.05, 0.10)
        assert sig == 0.0

    def test_rising_oi_slightly_positive(self):
        oi = [1000.0, 1020.0, 1040.0, 1060.0]  # +6% rise
        sig = oi_velocity_signal(oi, 0.05, 0.10)
        assert sig > 0.0
        assert sig <= 0.3

    def test_zero_start_returns_zero(self):
        oi = [0.0, 100.0, 200.0]
        sig = oi_velocity_signal(oi, 0.05, 0.10)
        assert sig == 0.0

    def test_output_bounded(self):
        oi = [1000.0, 1.0]  # extreme 99% drop
        sig = oi_velocity_signal(oi, 0.05, 0.10)
        assert -1.0 <= sig <= 1.0


# ── price_oi_divergence_signal ────────────────────────────────────────────────

class TestPriceOIDivergenceSignal:
    def test_price_down_oi_down_bearish(self):
        # Long liquidation cascade
        sig = price_oi_divergence_signal(price_change=-0.05, oi_change=-0.08)
        assert sig < 0.0

    def test_price_up_oi_down_positive(self):
        # Short squeeze
        sig = price_oi_divergence_signal(price_change=0.03, oi_change=-0.06)
        assert sig > 0.0

    def test_small_changes_return_zero(self):
        sig = price_oi_divergence_signal(price_change=0.001, oi_change=0.001)
        assert sig == 0.0

    def test_price_up_oi_up_returns_zero(self):
        # Healthy trend: no divergence signal
        sig = price_oi_divergence_signal(price_change=0.03, oi_change=0.05)
        assert sig == 0.0

    def test_output_bounded(self):
        sig = price_oi_divergence_signal(price_change=-1.0, oi_change=-1.0)
        assert -1.0 <= sig <= 1.0


# ── LiquidationMonitor ────────────────────────────────────────────────────────

class TestLiquidationMonitor:
    def test_get_signal_returns_float(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            sig = mon.get_signal("CRYPTO:BTC/USD")
        assert isinstance(sig, float)

    def test_get_signal_bounded(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            sig = mon.get_signal("CRYPTO:BTC/USD")
        assert -1.0 <= sig <= 1.0

    def test_high_funding_gives_negative_signal(self):
        mon = LiquidationMonitor()
        # Extreme positive funding → bearish cascade risk
        with patch.object(mon, "_fetch_funding_rate", return_value=0.002), \
             patch.object(mon, "_fetch_oi_series", return_value=[1000.0, 1000.0]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            mon._cache.clear()
            sig = mon.get_signal("CRYPTO:BTC/USD")
        assert sig < 0.0

    def test_critical_oi_drop_gives_negative_signal(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[1000.0, 850.0, 800.0]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(-0.05, -0.20)):
            mon._cache.clear()
            sig = mon.get_signal("CRYPTO:BTC/USD")
        assert sig < 0.0

    def test_normal_conditions_sizing_mult_one(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[1000.0, 1000.0]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            mon._cache.clear()
            mult = mon.get_sizing_multiplier("CRYPTO:BTC/USD")
        assert mult == pytest.approx(1.0)

    def test_cascade_risk_reduces_sizing(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.003), \
             patch.object(mon, "_fetch_oi_series", return_value=[1000.0, 850.0]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(-0.08, -0.15)):
            mon._cache.clear()
            mult = mon.get_sizing_multiplier("CRYPTO:BTC/USD")
        assert mult < 1.0

    def test_sizing_multiplier_bounded_by_floor(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.01), \
             patch.object(mon, "_fetch_oi_series", return_value=[1000.0, 100.0]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(-0.5, -0.9)):
            mon._cache.clear()
            mult = mon.get_sizing_multiplier("CRYPTO:BTC/USD")
        floor = float(_cfg_val("LIQUIDATION_MONITOR_SIZING_MULT_FLOOR"))
        assert mult >= floor

    def test_get_state_returns_liquidation_state(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            state = mon.get_state("CRYPTO:BTC/USD")
        assert isinstance(state, LiquidationState)

    def test_get_report_has_expected_keys(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            report = mon.get_report("CRYPTO:BTC/USD")
        for key in ["symbol", "funding_signal", "oi_signal", "composite_signal", "risk_level", "sizing_multiplier"]:
            assert key in report

    def test_caching_returns_same_state(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            s1 = mon.get_state("CRYPTO:ETH/USD")
            s2 = mon.get_state("CRYPTO:ETH/USD")
        assert s1 is s2

    def test_disabled_returns_zero_signal(self, monkeypatch):
        import monitoring.liquidation_monitor as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "LIQUIDATION_MONITOR_ENABLED" else mod._DEF.get(k))
        mon = LiquidationMonitor()
        assert mon.get_signal("CRYPTO:BTC/USD") == 0.0

    def test_disabled_returns_one_sizing_mult(self, monkeypatch):
        import monitoring.liquidation_monitor as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "LIQUIDATION_MONITOR_ENABLED" else mod._DEF.get(k))
        mon = LiquidationMonitor()
        assert mon.get_sizing_multiplier("CRYPTO:BTC/USD") == pytest.approx(1.0)

    def test_critical_risk_level(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.005), \
             patch.object(mon, "_fetch_oi_series", return_value=[1000.0, 700.0]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(-0.10, -0.30)):
            mon._cache.clear()
            state = mon.get_state("CRYPTO:BTC/USD")
        assert state.risk_level in ("warning", "critical")

    def test_normal_conditions_risk_level_normal(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[1000.0, 1000.0]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            mon._cache.clear()
            state = mon.get_state("CRYPTO:ETH/USD")
        assert state.risk_level == "normal"

    def test_liquidation_state_to_dict(self):
        s = LiquidationState(symbol="CRYPTO:BTC/USD", composite_signal=-0.5, risk_level="warning")
        d = s.to_dict()
        assert d["symbol"] == "CRYPTO:BTC/USD"
        assert d["risk_level"] == "warning"
        assert "timestamp" in d

    def test_get_all_report_after_query(self):
        mon = LiquidationMonitor()
        with patch.object(mon, "_fetch_funding_rate", return_value=0.0), \
             patch.object(mon, "_fetch_oi_series", return_value=[]), \
             patch.object(mon, "_fetch_price_oi_change", return_value=(0.0, 0.0)):
            mon.get_signal("CRYPTO:BTC/USD")
            report = mon.get_all_report()
        assert "CRYPTO:BTC/USD" in report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg_val(key: str):
    from monitoring.liquidation_monitor import _DEF
    return _DEF[key]


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_instance(self):
        assert isinstance(get_liquidation_monitor(), LiquidationMonitor)

    def test_same_object(self):
        a = get_liquidation_monitor()
        b = get_liquidation_monitor()
        assert a is b
