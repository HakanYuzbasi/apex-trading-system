"""
tests/test_short_selling_gate.py — Unit tests for risk/short_selling_gate.py
"""
from __future__ import annotations

import pytest
from risk.short_selling_gate import check_short_gate, get_short_exposure_summary


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _call(
    symbol="AAPL",
    signal=-0.20,
    confidence=0.65,
    regime="bear",
    asset_class="EQUITY",
    current_price=150.0,
    positions=None,
    price_cache=None,
    current_vix=18.0,
):
    return check_short_gate(
        symbol=symbol,
        signal=signal,
        confidence=confidence,
        regime=regime,
        asset_class=asset_class,
        current_price=current_price,
        current_positions=positions or {},
        price_cache=price_cache or {},
        current_vix=current_vix,
    )


# ── Positive signal passes through ───────────────────────────────────────────

class TestPositiveSignal:
    def test_positive_signal_not_a_short(self):
        allowed, reason, _ = _call(signal=0.20)
        assert allowed is True
        assert reason == "not_a_short"

    def test_zero_signal_not_a_short(self):
        allowed, reason, _ = _call(signal=0.0)
        assert allowed is True


# ── Disabled gate ─────────────────────────────────────────────────────────────

class TestDisabledGate:
    def test_disabled_blocks_all_shorts(self, monkeypatch):
        import risk.short_selling_gate as mod
        monkeypatch.setattr(mod, "_cfg", lambda k: False if k == "SHORT_SELLING_ENABLED" else mod._DEF.get(k))
        allowed, reason, _ = _call()
        assert allowed is False
        assert "disabled" in reason


# ── Regime checks ─────────────────────────────────────────────────────────────

class TestRegimeChecks:
    def test_bear_regime_allowed(self):
        allowed, _, _ = _call(regime="bear")
        assert allowed is True

    def test_strong_bear_allowed(self):
        allowed, _, _ = _call(regime="strong_bear")
        assert allowed is True

    def test_neutral_allowed(self):
        allowed, _, _ = _call(regime="neutral")
        assert allowed is True

    def test_volatile_allowed(self):
        allowed, _, _ = _call(regime="volatile")
        assert allowed is True

    def test_bull_blocked(self):
        allowed, reason, _ = _call(regime="bull")
        assert allowed is False
        assert "regime_blocked" in reason

    def test_strong_bull_blocked(self):
        allowed, reason, _ = _call(regime="strong_bull")
        assert allowed is False


# ── Crypto pass-through ───────────────────────────────────────────────────────

class TestCryptoPassthrough:
    def test_crypto_short_allowed_regardless(self):
        allowed, reason, _ = _call(asset_class="CRYPTO", regime="neutral")
        assert allowed is True
        assert reason == "crypto_short_allowed"


# ── VIX guard ─────────────────────────────────────────────────────────────────

class TestVixGuard:
    def test_normal_vix_allowed(self):
        allowed, _, _ = _call(current_vix=18.0)
        assert allowed is True

    def test_high_vix_blocked(self):
        allowed, reason, _ = _call(current_vix=40.0)
        assert allowed is False
        assert "vix_too_high" in reason

    def test_none_vix_allowed(self):
        allowed, _, _ = _call(current_vix=None)
        assert allowed is True


# ── Price guard ───────────────────────────────────────────────────────────────

class TestPriceGuard:
    def test_normal_price_allowed(self):
        allowed, _, _ = _call(current_price=50.0)
        assert allowed is True

    def test_penny_stock_blocked(self):
        allowed, reason, _ = _call(current_price=5.0)
        assert allowed is False
        assert "price_too_low" in reason

    def test_zero_price_allowed(self):
        # Zero price = unknown price, skip check
        allowed, _, _ = _call(current_price=0.0)
        assert allowed is True


# ── Signal and confidence floors ──────────────────────────────────────────────

class TestFloors:
    def test_weak_signal_blocked(self):
        allowed, reason, _ = _call(signal=-0.05)
        assert allowed is False
        assert "signal_too_weak" in reason

    def test_strong_signal_allowed(self):
        allowed, _, _ = _call(signal=-0.20)
        assert allowed is True

    def test_low_confidence_blocked(self):
        allowed, reason, _ = _call(confidence=0.40)
        assert allowed is False
        assert "confidence_too_low" in reason

    def test_sufficient_confidence_allowed(self):
        allowed, _, _ = _call(confidence=0.65)
        assert allowed is True


# ── Position limits ───────────────────────────────────────────────────────────

class TestPositionLimits:
    def test_max_positions_block(self):
        positions = {"SYM1": -100, "SYM2": -50, "SYM3": -75}  # 3 shorts = max
        allowed, reason, _ = _call(positions=positions, price_cache={"SYM1": 50, "SYM2": 100, "SYM3": 80})
        assert allowed is False
        assert "max_positions" in reason

    def test_below_max_positions_allowed(self):
        positions = {"SYM1": -100, "SYM2": -50}  # 2 shorts < max (3)
        allowed, _, _ = _call(positions=positions, price_cache={"SYM1": 50, "SYM2": 100})
        assert allowed is True

    def test_total_notional_limit(self):
        # $70k total short notional > $60k limit
        positions = {"SYM1": -200, "SYM2": -150}
        price_cache = {"SYM1": 200.0, "SYM2": 200.0}  # 200×200 + 150×200 = $70k
        allowed, reason, _ = _call(positions=positions, price_cache=price_cache)
        assert allowed is False
        assert "total_notional" in reason

    def test_long_positions_not_counted_as_shorts(self):
        positions = {"SYM1": 100, "SYM2": 200}  # LONG positions
        allowed, _, _ = _call(positions=positions, price_cache={"SYM1": 50, "SYM2": 50})
        assert allowed is True


# ── Confidence adjustment ─────────────────────────────────────────────────────

class TestConfidenceAdjustment:
    def test_confidence_slightly_reduced_for_shorts(self):
        _, _, conf_adj = _call(confidence=0.70)
        assert conf_adj < 0.70
        assert conf_adj > 0.60  # not too much reduction


# ── Short exposure summary ────────────────────────────────────────────────────

class TestShortExposureSummary:
    def test_empty_positions(self):
        summary = get_short_exposure_summary({}, {})
        assert summary["short_count"] == 0
        assert summary["total_short_notional"] == 0.0

    def test_counts_short_positions_only(self):
        positions = {"AAPL": -100, "MSFT": 50, "TSLA": -30}  # 2 shorts, 1 long
        price_cache = {"AAPL": 150.0, "MSFT": 300.0, "TSLA": 200.0}
        summary = get_short_exposure_summary(positions, price_cache)
        assert summary["short_count"] == 2

    def test_notional_computed_correctly(self):
        positions = {"AAPL": -100}
        price_cache = {"AAPL": 150.0}
        summary = get_short_exposure_summary(positions, price_cache)
        assert summary["total_short_notional"] == pytest.approx(15000.0)

    def test_summary_has_expected_keys(self):
        summary = get_short_exposure_summary({}, {})
        assert "short_count" in summary
        assert "total_short_notional" in summary
        assert "max_allowed" in summary
        assert "max_total_notional" in summary
        assert "positions" in summary
