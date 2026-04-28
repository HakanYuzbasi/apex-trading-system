"""tests/test_kelly_real_winrate.py — Kelly sizing real win-rate feedback tests.

Verifies that:
  1. get_kelly_multiplier responds correctly to real win-rate values
  2. SignalDriftMonitor.get_state().rolling_win_rate is the correct data source
  3. Fallback to 0.5 fires when SDM has < 10 trades
  4. High win-rate → larger Kelly multiplier; low win-rate → smaller / zero
  5. Invalid win-rate values (0.0, 1.0) are clamped to 0.5 inside the function
"""
from __future__ import annotations

import pytest

from models.portfolio_optimizer import get_kelly_multiplier
from monitoring.signal_drift_monitor import SignalDriftMonitor


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sdm(**kwargs) -> SignalDriftMonitor:
    defaults = dict(
        short_window=10,
        long_window=30,
        drift_threshold=0.15,
        min_trades=5,
        recover_tolerance=0.04,
    )
    defaults.update(kwargs)
    return SignalDriftMonitor(**defaults)


def _record_n(sdm: SignalDriftMonitor, n: int, correct: bool) -> None:
    for _ in range(n):
        sdm.record_outcome("SYM", correct)


def _kelly_win_rate_from_sdm(sdm: SignalDriftMonitor, min_trades: int = 10) -> float:
    """Mirrors the production logic added to execution_loop.py."""
    if sdm is not None and sdm._total >= min_trades:
        return float(sdm.get_state().rolling_win_rate)
    return 0.5  # fallback


# ── get_kelly_multiplier basic behaviour ──────────────────────────────────────

class TestKellyMultiplierBasic:
    def test_returns_float(self):
        m = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.55)
        assert isinstance(m, float)

    def test_neutral_win_rate_and_confidence(self):
        m = get_kelly_multiplier(ml_confidence=0.50, historical_win_rate=0.50)
        # p=0.5, q=0.5, b=1.5 → kelly = 0.5 - 0.5/1.5 ≈ 0.167
        # fractional_kelly * (1/0.10) should be in (0, 2.5]
        assert 0.0 < m <= 2.5

    def test_high_win_rate_bigger_than_neutral(self):
        low = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.40)
        high = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.75)
        assert high > low

    def test_low_win_rate_may_return_zero(self):
        m = get_kelly_multiplier(ml_confidence=0.40, historical_win_rate=0.25)
        assert m >= 0.0

    def test_win_rate_zero_clamped_to_half(self):
        m_zero = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.0)
        m_half = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.5)
        assert abs(m_zero - m_half) < 1e-9  # 0.0 → clamped to 0.5

    def test_win_rate_one_clamped_to_half(self):
        m_one = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=1.0)
        m_half = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.5)
        assert abs(m_one - m_half) < 1e-9  # 1.0 → clamped to 0.5

    def test_high_vix_reduces_multiplier(self):
        normal = get_kelly_multiplier(ml_confidence=0.70, historical_win_rate=0.60)
        high_vix = get_kelly_multiplier(ml_confidence=0.70, historical_win_rate=0.60, is_high_vix=True)
        assert high_vix < normal

    def test_high_vix_factor_approx_60pct(self):
        normal = get_kelly_multiplier(ml_confidence=0.70, historical_win_rate=0.60)
        high_vix = get_kelly_multiplier(ml_confidence=0.70, historical_win_rate=0.60, is_high_vix=True)
        if normal > 0:
            assert abs(high_vix / normal - 0.6) < 1e-9

    def test_clipped_to_max(self):
        m = get_kelly_multiplier(ml_confidence=0.99, historical_win_rate=0.85)
        assert m <= 3.0  # MAX_LEVERAGE = 3.0

    def test_not_negative(self):
        m = get_kelly_multiplier(ml_confidence=0.10, historical_win_rate=0.10)
        assert m >= 0.0


# ── SDM win-rate as Kelly input ───────────────────────────────────────────────

class TestSDMWinRateSource:
    def test_fallback_when_too_few_trades(self):
        sdm = _sdm()
        _record_n(sdm, 5, True)  # only 5 trades < 10
        wr = _kelly_win_rate_from_sdm(sdm, min_trades=10)
        assert wr == 0.5

    def test_uses_sdm_when_enough_trades(self):
        sdm = _sdm(short_window=10, long_window=30)
        _record_n(sdm, 10, True)  # 100% correct
        wr = _kelly_win_rate_from_sdm(sdm, min_trades=10)
        assert abs(wr - 1.0) < 1e-9

    def test_uses_sdm_partial_win_rate(self):
        sdm = _sdm(short_window=10, long_window=30)
        _record_n(sdm, 6, True)
        _record_n(sdm, 4, False)
        wr = _kelly_win_rate_from_sdm(sdm, min_trades=10)
        assert abs(wr - 0.6) < 1e-9

    def test_sdm_none_returns_fallback(self):
        wr = _kelly_win_rate_from_sdm(None, min_trades=10)  # type: ignore[arg-type]
        assert wr == 0.5

    def test_exactly_10_trades_qualifies(self):
        sdm = _sdm(short_window=10, long_window=30)
        _record_n(sdm, 10, False)  # exactly 10, all wrong
        wr = _kelly_win_rate_from_sdm(sdm, min_trades=10)
        assert wr == 0.0  # real SDM value, not fallback

    def test_nine_trades_still_falls_back(self):
        sdm = _sdm(short_window=10, long_window=30)
        _record_n(sdm, 9, True)
        wr = _kelly_win_rate_from_sdm(sdm, min_trades=10)
        assert wr == 0.5


# ── Kelly output with SDM-sourced win-rate ────────────────────────────────────

class TestKellyWithSDMWinRate:
    def test_high_real_winrate_increases_kelly(self):
        sdm = _sdm(short_window=10, long_window=30)
        _record_n(sdm, 8, True)
        _record_n(sdm, 2, False)  # 80% rolling win-rate (not clamped — 0.0/1.0 are the edge cases)
        wr = _kelly_win_rate_from_sdm(sdm)
        assert abs(wr - 0.8) < 1e-9
        m_real = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=wr)
        m_neutral = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.5)
        assert m_real > m_neutral

    def test_low_real_winrate_decreases_kelly(self):
        sdm = _sdm(short_window=10, long_window=30)
        _record_n(sdm, 3, True)
        _record_n(sdm, 7, False)  # 30% correct in short window
        wr = _kelly_win_rate_from_sdm(sdm)
        m_real = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=wr)
        m_neutral = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.5)
        assert m_real < m_neutral

    def test_very_low_winrate_returns_zero_kelly(self):
        # p = (conf + wr) / 2 <= 0.4 needed for Kelly <= 0
        # With conf=0.65: wr <= 0.15 triggers zero Kelly
        sdm = _sdm(short_window=10, long_window=30)
        _record_n(sdm, 1, True)
        _record_n(sdm, 9, False)  # 10% rolling win-rate (not clamped)
        wr = _kelly_win_rate_from_sdm(sdm)
        assert abs(wr - 0.1) < 1e-9
        m = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=wr)
        # full_kelly = 0.375 - (0.625/1.5) = 0.375 - 0.417 = -0.042 <= 0 → returns MIN_LEVERAGE=0.1
        assert m == pytest.approx(0.1)

    def test_zero_winrate_clamped_to_half_in_kelly(self):
        """SDM rolling_win_rate=0.0 gets clamped to 0.5 inside get_kelly_multiplier."""
        sdm = _sdm(short_window=10, long_window=30)
        _record_n(sdm, 10, False)  # 0% rolling win-rate
        wr = _kelly_win_rate_from_sdm(sdm)
        assert wr == 0.0
        # Inside Kelly: 0.0 → clamped to 0.5, so result = same as neutral
        m_zero = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=wr)
        m_neutral = get_kelly_multiplier(ml_confidence=0.65, historical_win_rate=0.5)
        assert abs(m_zero - m_neutral) < 1e-9

    def test_fallback_neutral_produces_positive_kelly(self):
        sdm = _sdm()
        _record_n(sdm, 5, True)  # < 10 → fallback 0.5
        wr = _kelly_win_rate_from_sdm(sdm)
        assert wr == 0.5
        m = get_kelly_multiplier(ml_confidence=0.60, historical_win_rate=wr)
        assert m > 0.0

    def test_sdm_drift_state_updates_kelly_input(self):
        sdm = _sdm(short_window=5, long_window=20, drift_threshold=0.15, min_trades=5)
        # Build a strong baseline first
        _record_n(sdm, 15, True)
        wr_before = _kelly_win_rate_from_sdm(sdm)
        # Now introduce all-wrong trades (triggers drift)
        _record_n(sdm, 5, False)
        wr_after = _kelly_win_rate_from_sdm(sdm)
        # After the bad streak, rolling win-rate should be lower
        assert wr_after < wr_before
