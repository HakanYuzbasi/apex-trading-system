"""Tests for CorrelationEarlyWarning module."""
from __future__ import annotations

import pytest

from risk.correlation_early_warning import CorrelationEarlyWarning, CorrelationStats


class TestCorrelationEarlyWarning:

    def _make(self, **kwargs) -> CorrelationEarlyWarning:
        defaults = dict(short_window=6, long_window=20, rising_threshold=0.20,
                        alert_threshold=0.45, shift_threshold=0.35, min_samples=4)
        defaults.update(kwargs)
        return CorrelationEarlyWarning(**defaults)

    def test_returns_one_before_min_samples(self):
        cew = self._make()
        # Only 3 prices — below min_samples=4
        for i in range(3):
            cew.record_prices(spy_price=400.0 + i, btc_price=20000.0 + i * 10)
        assert cew.get_position_multiplier() == 1.0

    def test_returns_one_after_min_samples_with_no_correlation(self):
        cew = self._make()
        # SPY drifts up, BTC drifts down — negative correlation, no shift triggered
        for i in range(15):
            spy = 400.0 + i * 0.5
            btc = 20000.0 - i * 100.0
            cew.record_prices(spy, btc)
        # Should not trigger (negative short_corr < shift_threshold)
        assert cew.get_position_multiplier() == 1.0

    def test_normal_tier_low_velocity(self):
        cew = self._make()
        # Feed identical trends → correlation near 1, but no velocity (short == long)
        import random
        rng = random.Random(42)
        for i in range(25):
            delta = rng.gauss(0, 1)
            cew.record_prices(400.0 + delta, 20000.0 + delta * 10)
        stats = cew.get_stats()
        assert stats is not None
        # velocity = short_corr - long_corr; with same data, should be near 0
        assert abs(stats.velocity) < 0.50  # broad sanity check

    def test_get_diagnostics_before_any_prices(self):
        cew = self._make()
        diag = cew.get_diagnostics()
        assert diag["tier"] == "normal"
        assert diag["position_multiplier"] == 1.0
        assert diag["short_corr"] is None

    def test_get_diagnostics_after_prices(self):
        cew = self._make()
        for i in range(10):
            cew.record_prices(400.0 + i, 20000.0 + i * 50)
        diag = cew.get_diagnostics()
        assert "tier" in diag
        assert "velocity" in diag
        assert "samples_short" in diag

    def test_record_invalid_prices_ignored(self):
        cew = self._make()
        cew.record_prices(0, 20000)      # spy=0 ignored
        cew.record_prices(400, 0)        # btc=0 ignored
        cew.record_prices(-1, 20000)     # spy<0 ignored
        assert len(cew._spy_prices) == 0

    def test_classify_alert_tier(self):
        """High velocity + corr above shift threshold → alert, mult=0.50."""
        cew = self._make(shift_threshold=0.35, alert_threshold=0.45)
        tier, mult = cew._classify(short_corr=0.80, velocity=0.50)
        assert tier == "alert"
        assert mult == pytest.approx(0.50)

    def test_classify_warning_tier(self):
        cew = self._make(rising_threshold=0.20, alert_threshold=0.45)
        tier, mult = cew._classify(short_corr=0.80, velocity=0.35)
        assert tier == "warning"
        assert mult == pytest.approx(0.65)

    def test_classify_caution_tier(self):
        cew = self._make(rising_threshold=0.20, alert_threshold=0.45)
        tier, mult = cew._classify(short_corr=0.80, velocity=0.22)
        assert tier == "caution"
        assert mult == pytest.approx(0.80)

    def test_classify_normal_below_shift_threshold(self):
        """Even high velocity is ignored if short_corr < shift_threshold."""
        cew = self._make(shift_threshold=0.50)
        tier, mult = cew._classify(short_corr=0.30, velocity=0.90)
        assert tier == "normal"
        assert mult == 1.0

    def test_rolling_corr_perfect_correlation(self):
        import numpy as np
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = CorrelationEarlyWarning._rolling_corr(x, x)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_rolling_corr_constant_series(self):
        import numpy as np
        # Constant series has 0 std → returns 0.0 gracefully
        x = np.ones(10)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        corr = CorrelationEarlyWarning._rolling_corr(x, y)
        assert corr == 0.0

    def test_maxlen_deque_bounded(self):
        """Buffer should never grow beyond long_window+1."""
        cew = self._make(long_window=10)
        for i in range(30):
            cew.record_prices(400.0 + i, 20000.0 + i)
        assert len(cew._spy_prices) <= 11
        assert len(cew._btc_prices) <= 11
