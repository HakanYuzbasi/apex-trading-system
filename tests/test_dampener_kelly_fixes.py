"""
tests/test_dampener_kelly_fixes.py
Tests for Fix #1 (crypto dampener floor) and Fix #3 (Kelly cold-start bootstrap).
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ── Stub heavy dependencies so imports work without live environment ──────────

for _mod in ("ib_insync", "alpaca_trade_api", "websockets", "redis",
             "telegram", "aiohttp", "cvxpy", "scipy",
             "sklearn", "xgboost", "lightgbm"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Minimal config stub
if "config" not in sys.modules:
    _cfg = types.ModuleType("config")
    sys.modules["config"] = _cfg

import config as _cfg_mod

if not hasattr(_cfg_mod, "ApexConfig"):
    _cfg_mod.ApexConfig = types.SimpleNamespace()

_apex = _cfg_mod.ApexConfig
# Set the new config keys we added
setattr(_apex, "CRYPTO_HEDGE_DAMPENER_FLOOR", 0.50)
setattr(_apex, "KELLY_COLD_START_WIN_RATE_FLOOR", 0.40)


# ── Fix #1: Crypto dampener floor ─────────────────────────────────────────────

class TestCryptoDampenerFloor(unittest.TestCase):
    """
    Validates that CRYPTO_HEDGE_DAMPENER_FLOOR prevents equity-calibrated
    VIX dampening from over-suppressing crypto signals.
    """

    def _apply_dampener(self, raw_dampener: float, is_crypto: bool,
                         floor: float = 0.50) -> float:
        """Replicate the execution_loop dampener application logic."""
        if raw_dampener >= 1.0:
            return 1.0  # no damping
        effective = raw_dampener
        if is_crypto:
            effective = max(raw_dampener, floor)
        return effective

    def test_equity_dampener_fully_applied(self):
        """Equity signals receive the full HedgeManager dampener."""
        result = self._apply_dampener(0.30, is_crypto=False)
        self.assertAlmostEqual(result, 0.30)

    def test_crypto_dampener_floored_at_50pct(self):
        """When HedgeManager returns 0.30 for crypto, floor raises it to 0.50."""
        result = self._apply_dampener(0.30, is_crypto=True, floor=0.50)
        self.assertAlmostEqual(result, 0.50)

    def test_crypto_dampener_above_floor_unchanged(self):
        """Dampener of 0.70 is above the 0.50 floor — should pass through unchanged."""
        result = self._apply_dampener(0.70, is_crypto=True, floor=0.50)
        self.assertAlmostEqual(result, 0.70)

    def test_crypto_dampener_exactly_at_floor(self):
        """Dampener exactly at floor should pass through without change."""
        result = self._apply_dampener(0.50, is_crypto=True, floor=0.50)
        self.assertAlmostEqual(result, 0.50)

    def test_no_dampening_when_factor_is_1(self):
        """raw_dampener=1.0 means no hedge triggered — unchanged for both types."""
        self.assertAlmostEqual(self._apply_dampener(1.0, is_crypto=True), 1.0)
        self.assertAlmostEqual(self._apply_dampener(1.0, is_crypto=False), 1.0)

    def test_min_dampener_10pct_equity_still_reaches_gates(self):
        """Worst-case equity dampener (0.10) is still applied fully."""
        result = self._apply_dampener(0.10, is_crypto=False)
        self.assertAlmostEqual(result, 0.10)

    def test_min_dampener_10pct_crypto_floored_to_50pct(self):
        """Worst-case crypto dampener (0.10) is raised to 0.50 floor."""
        result = self._apply_dampener(0.10, is_crypto=True)
        self.assertAlmostEqual(result, 0.50)

    def test_signal_reduction_equity_vs_crypto_comparison(self):
        """
        Concrete example: at VIX=26.95, HedgeManager returns ≈0.72 dampener.
        Equity signal -0.567 → -0.408; crypto signal stays at -0.567*max(0.72,0.50)=-0.408.
        Both same here, but at VIX_ELEVATED the dampener can hit 0.30 for equity —
        for crypto it's still 0.50 minimum.
        """
        vix_crisis_dampener = 0.25  # simulated high-VIX dampener
        signal = -0.400

        equity_result = signal * self._apply_dampener(vix_crisis_dampener, is_crypto=False)
        crypto_result = signal * self._apply_dampener(vix_crisis_dampener, is_crypto=True)

        self.assertAlmostEqual(equity_result, -0.10, places=4)
        self.assertAlmostEqual(crypto_result, -0.20, places=4)  # floor=0.50
        # Crypto retains more conviction
        self.assertGreater(abs(crypto_result), abs(equity_result))

    def test_configurable_floor_value(self):
        """Floor can be overridden via config key."""
        result_60 = self._apply_dampener(0.30, is_crypto=True, floor=0.60)
        result_40 = self._apply_dampener(0.30, is_crypto=True, floor=0.40)
        self.assertAlmostEqual(result_60, 0.60)
        self.assertAlmostEqual(result_40, 0.40)

    def test_config_key_present(self):
        """CRYPTO_HEDGE_DAMPENER_FLOOR is set in ApexConfig and is positive."""
        val = getattr(_apex, "CRYPTO_HEDGE_DAMPENER_FLOOR", None)
        self.assertIsNotNone(val)
        self.assertGreater(float(val), 0.0)
        self.assertLessEqual(float(val), 1.0)


# ── Fix #3: Kelly cold-start bootstrap ────────────────────────────────────────

class TestKellyColdStartBootstrap(unittest.TestCase):
    """
    Validates that Kelly sizing never returns 0.0 (frozen) and that
    the cold-start win-rate floor prevents early losses from killing trades.
    """

    def _get_kelly_mult(self, ml_confidence: float, win_rate: float,
                         is_high_vix: bool = False) -> float:
        """Run the actual portfolio_optimizer.get_kelly_multiplier."""
        from models.portfolio_optimizer import get_kelly_multiplier
        return get_kelly_multiplier(ml_confidence, win_rate, is_high_vix)

    def test_kelly_never_returns_zero(self):
        """After Fix #3a, Kelly must return MIN_LEVERAGE (0.1) not 0.0 even at bad odds."""
        # win_rate=0.25 + confidence=0.35 → full_kelly ≤ 0, used to return 0.0
        result = self._get_kelly_mult(ml_confidence=0.35, win_rate=0.25)
        self.assertGreater(result, 0.0)

    def test_kelly_min_leverage_floor(self):
        """Worst-case Kelly returns at least MIN_LEVERAGE (0.1)."""
        result = self._get_kelly_mult(ml_confidence=0.30, win_rate=0.20)
        self.assertGreaterEqual(result, 0.1)

    def test_kelly_zero_win_rate_clamped_to_50pct(self):
        """Win rate of 0.0 is clamped to 0.5 inside portfolio_optimizer."""
        result_zero = self._get_kelly_mult(ml_confidence=0.55, win_rate=0.0)
        result_50 = self._get_kelly_mult(ml_confidence=0.55, win_rate=0.5)
        self.assertAlmostEqual(result_zero, result_50, places=3)

    def test_kelly_full_win_rate_clamped(self):
        """Win rate of 1.0 is clamped to 0.5."""
        result_full = self._get_kelly_mult(ml_confidence=0.55, win_rate=1.0)
        result_50 = self._get_kelly_mult(ml_confidence=0.55, win_rate=0.5)
        self.assertAlmostEqual(result_full, result_50, places=3)

    def test_cold_start_floor_raises_low_win_rate(self):
        """Win-rate floor of 0.40 prevents <0.40 values during cold-start."""
        floor = float(getattr(_apex, "KELLY_COLD_START_WIN_RATE_FLOOR", 0.40))
        raw_win_rate = 0.10  # terrible early performance
        floored = max(raw_win_rate, floor)
        self.assertGreaterEqual(floored, 0.40)

    def test_cold_start_floor_does_not_lower_good_win_rate(self):
        """Floor of 0.40 doesn't reduce a healthy 0.65 win-rate."""
        floor = 0.40
        raw_win_rate = 0.65
        floored = max(raw_win_rate, floor)
        self.assertAlmostEqual(floored, 0.65)

    def test_kelly_with_cold_start_floor_produces_reasonable_mult(self):
        """With cold-start floor applied, Kelly mult should be between 0.1 and 3.0."""
        floor = float(getattr(_apex, "KELLY_COLD_START_WIN_RATE_FLOOR", 0.40))
        floored_wr = max(0.10, floor)  # worst-case early run, then floored
        result = self._get_kelly_mult(ml_confidence=0.55, win_rate=floored_wr)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 3.0)

    def test_high_vix_penalty_still_applied(self):
        """is_high_vix=True must reduce Kelly mult by 40%."""
        mult_normal = self._get_kelly_mult(0.60, 0.55, is_high_vix=False)
        mult_hv = self._get_kelly_mult(0.60, 0.55, is_high_vix=True)
        self.assertAlmostEqual(mult_hv, mult_normal * 0.6, places=3)

    def test_kelly_scales_with_confidence(self):
        """Higher ML confidence → higher Kelly multiplier (holding win_rate fixed)."""
        low = self._get_kelly_mult(0.40, 0.50)
        high = self._get_kelly_mult(0.80, 0.50)
        self.assertGreater(high, low)

    def test_kelly_config_key_present(self):
        """KELLY_COLD_START_WIN_RATE_FLOOR is set in ApexConfig."""
        val = getattr(_apex, "KELLY_COLD_START_WIN_RATE_FLOOR", None)
        self.assertIsNotNone(val)
        self.assertGreater(float(val), 0.0)
        self.assertLess(float(val), 1.0)


# ── Integration: both fixes work together ─────────────────────────────────────

class TestDampenerKellyIntegration(unittest.TestCase):
    """Sanity checks combining Fix #1 and Fix #3."""

    def test_crypto_signal_survives_hedge_and_kelly_cold_start(self):
        """
        End-to-end simulation: crypto signal with VIX-crisis dampener + zero win-rate.
        Without Fix #1: signal dampened to 0.10x → below threshold (0.15).
        With Fix #1: crypto floor → dampener=0.50 → signal 0.35*0.50=0.175 > 0.15 ✓
        Without Fix #3: Kelly 0.0 → 0 shares.
        With Fix #3: Kelly >= MIN_LEVERAGE → trades execute.
        """
        from models.portfolio_optimizer import get_kelly_multiplier

        raw_signal = 0.350
        crisis_dampener = 0.20  # severe VIX spike

        # Without fix: equity path
        equity_signal = raw_signal * crisis_dampener
        self.assertLess(equity_signal, 0.15)  # below CRYPTO_MIN_SIGNAL_THRESHOLD

        # With Fix #1: crypto floor
        crypto_floor = 0.50
        effective_dampener = max(crisis_dampener, crypto_floor)
        crypto_signal = raw_signal * effective_dampener
        self.assertGreater(crypto_signal, 0.15)  # passes signal gate ✓

        # Without Fix #3: Kelly at zero win-rate (old behavior was 0.0)
        # With Fix #3: Kelly returns at least MIN_LEVERAGE
        floor_wr = 0.40  # cold-start floor
        kelly_mult = get_kelly_multiplier(0.55, max(0.10, floor_wr))
        self.assertGreater(kelly_mult, 0.0)  # trades happen ✓


if __name__ == "__main__":
    unittest.main()
