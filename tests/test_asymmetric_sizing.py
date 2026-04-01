"""
tests/test_asymmetric_sizing.py — Unit tests for risk/asymmetric_sizing.py
"""
import pytest
from risk.asymmetric_sizing import (
    compute_asymmetric_levels,
    update_breakeven_lock,
    check_signal_hold,
)


# ── compute_asymmetric_levels ────────────────────────────────────────────────

class TestComputeAsymmetricLevels:

    def _base(self, **overrides):
        kwargs = dict(entry_price=100.0, atr=1.5, signal=0.20, confidence=0.70,
                      regime="bull", is_crypto=False, is_long=True)
        kwargs.update(overrides)
        return compute_asymmetric_levels(**kwargs)

    def test_returns_dict_with_required_keys(self):
        r = self._base()
        for key in ("stop_loss", "take_profit", "trailing_stop_pct",
                    "trailing_activation_pct", "breakeven_trigger_pct",
                    "breakeven_price", "breakeven_locked",
                    "signal_hold_threshold", "atr", "rr_ratio", "asym_managed"):
            assert key in r, f"Missing key: {key}"

    def test_asym_managed_flag(self):
        assert self._base()["asym_managed"] is True

    def test_breakeven_locked_starts_false(self):
        assert self._base()["breakeven_locked"] is False

    def test_long_stop_below_entry(self):
        r = self._base(entry_price=100.0, is_long=True)
        assert r["stop_loss"] < 100.0

    def test_short_stop_above_entry(self):
        r = self._base(entry_price=100.0, is_long=False)
        assert r["stop_loss"] > 100.0

    def test_long_tp_above_entry(self):
        r = self._base(entry_price=100.0, is_long=True)
        assert r["take_profit"] > 100.0

    def test_short_tp_below_entry(self):
        r = self._base(entry_price=100.0, is_long=False)
        assert r["take_profit"] < 100.0

    def test_rr_ratio_at_least_two(self):
        r = self._base(confidence=0.0)
        assert r["rr_ratio"] >= 2.0

    def test_higher_confidence_gives_better_rr(self):
        r_low = self._base(confidence=0.0)
        r_high = self._base(confidence=1.0)
        assert r_high["rr_ratio"] > r_low["rr_ratio"]

    def test_crypto_gets_wider_stop(self):
        r_eq = self._base(is_crypto=False, entry_price=100.0, atr=2.0)
        r_cr = self._base(is_crypto=True,  entry_price=100.0, atr=2.0)
        stop_dist_eq = 100.0 - r_eq["stop_loss"]
        stop_dist_cr = 100.0 - r_cr["stop_loss"]
        assert stop_dist_cr > stop_dist_eq

    def test_bull_regime_tp_gt_bear(self):
        r_bull = self._base(regime="bull")
        r_bear = self._base(regime="bear")
        tp_bull = r_bull["take_profit"] - 100.0
        tp_bear = r_bear["take_profit"] - 100.0
        assert tp_bull > tp_bear

    def test_stop_dist_clamped_min(self):
        # Very small ATR relative to price → clamped at 0.5%
        r = self._base(entry_price=10000.0, atr=0.001)
        stop_dist_pct = (10000.0 - r["stop_loss"]) / 10000.0
        assert stop_dist_pct >= 0.005 - 1e-6

    def test_tp_clamped_max(self):
        # Very high confidence + bull → tp capped at 30%
        r = self._base(confidence=1.0, regime="strong_bull", entry_price=100.0, atr=50.0)
        tp_pct = (r["take_profit"] - 100.0) / 100.0
        assert tp_pct <= 0.30 + 1e-6

    def test_trailing_activation_below_tp(self):
        r = self._base()
        tp_dist = r["take_profit"] - 100.0
        trail_act = r["trailing_activation_pct"]
        # trailing_activation_pct is a fraction of entry, confirm < tp distance
        assert trail_act < tp_dist / 100.0 + 1e-6

    def test_invalid_entry_returns_empty(self):
        assert compute_asymmetric_levels(entry_price=0, atr=1.5, signal=0.2, confidence=0.7) == {}
        assert compute_asymmetric_levels(entry_price=100, atr=0, signal=0.2, confidence=0.7) == {}

    def test_high_conf_uses_high_signal_threshold(self):
        r = self._base(confidence=0.80)
        assert r["signal_hold_threshold"] == pytest.approx(-0.25, abs=1e-6)

    def test_low_conf_uses_base_signal_threshold(self):
        r = self._base(confidence=0.50)
        assert r["signal_hold_threshold"] == pytest.approx(-0.10, abs=1e-6)

    def test_atr_preserved_in_result(self):
        r = self._base(atr=2.75)
        assert r["atr"] == pytest.approx(2.75)

    def test_strong_bear_regime(self):
        r = self._base(regime="strong_bear")
        # TP distance should be tighter than neutral
        r_neutral = self._base(regime="neutral")
        assert r["take_profit"] < r_neutral["take_profit"]


# ── update_breakeven_lock ────────────────────────────────────────────────────

class TestUpdateBreakevenLock:

    def _make_stops(self, entry=100.0, trigger_pct=0.02):
        return {
            "asym_managed": True,
            "breakeven_locked": False,
            "stop_loss": entry * 0.97,
            "breakeven_trigger_pct": trigger_pct,
        }

    def test_lock_applied_when_pnl_exceeds_trigger(self):
        ps = self._make_stops(entry=100.0, trigger_pct=0.02)
        locked = update_breakeven_lock(ps, current_price=103.0, entry_price=100.0, is_long=True)
        assert locked is True
        assert ps["breakeven_locked"] is True
        assert ps["stop_loss"] == pytest.approx(100.0)

    def test_lock_not_applied_when_below_trigger(self):
        ps = self._make_stops(entry=100.0, trigger_pct=0.02)
        locked = update_breakeven_lock(ps, current_price=100.5, entry_price=100.0, is_long=True)
        assert locked is False
        assert ps["breakeven_locked"] is False

    def test_already_locked_skips(self):
        ps = self._make_stops(entry=100.0)
        ps["breakeven_locked"] = True
        ps["stop_loss"] = 98.0  # someone else's value
        locked = update_breakeven_lock(ps, current_price=110.0, entry_price=100.0, is_long=True)
        assert locked is False
        assert ps["stop_loss"] == pytest.approx(98.0)  # unchanged

    def test_non_asym_managed_skips(self):
        ps = self._make_stops(entry=100.0)
        ps["asym_managed"] = False
        locked = update_breakeven_lock(ps, current_price=110.0, entry_price=100.0, is_long=True)
        assert locked is False

    def test_short_position_lock(self):
        ps = self._make_stops(entry=100.0, trigger_pct=0.02)
        ps["stop_loss"] = 103.0  # short: stop above entry
        # Short profit: price drops to 97, trigger = 2% → pnl = 3% > 2%
        locked = update_breakeven_lock(ps, current_price=97.0, entry_price=100.0, is_long=False)
        assert locked is True
        assert ps["stop_loss"] == pytest.approx(100.0)

    def test_zero_entry_price_skips(self):
        ps = self._make_stops(entry=100.0)
        locked = update_breakeven_lock(ps, current_price=110.0, entry_price=0.0, is_long=True)
        assert locked is False


# ── check_signal_hold ────────────────────────────────────────────────────────

class TestCheckSignalHold:

    def _make_stops(self, threshold=-0.10):
        return {"asym_managed": True, "signal_hold_threshold": threshold}

    def test_holds_when_in_profit_and_weak_signal(self):
        ps = self._make_stops(threshold=-0.10)
        # signal = -0.05 (above -0.10 threshold), pnl positive → hold
        assert check_signal_hold(ps, current_signal=-0.05, pnl_pct=0.03) is True

    def test_does_not_hold_when_signal_strongly_negative(self):
        ps = self._make_stops(threshold=-0.10)
        # signal = -0.20 (below -0.10 threshold) → allow exit
        assert check_signal_hold(ps, current_signal=-0.20, pnl_pct=0.03) is False

    def test_does_not_hold_when_in_loss(self):
        ps = self._make_stops(threshold=-0.10)
        # pnl negative → never suppress exit
        assert check_signal_hold(ps, current_signal=-0.05, pnl_pct=-0.01) is False

    def test_non_asym_managed_always_false(self):
        ps = {"asym_managed": False, "signal_hold_threshold": -0.10}
        assert check_signal_hold(ps, current_signal=0.0, pnl_pct=0.05) is False

    def test_positive_signal_in_profit_holds(self):
        ps = self._make_stops(threshold=-0.10)
        # signal still positive → definitely hold
        assert check_signal_hold(ps, current_signal=0.15, pnl_pct=0.05) is True

    def test_high_conf_threshold_stricter(self):
        ps = self._make_stops(threshold=-0.25)
        # -0.20 is above -0.25 → still hold
        assert check_signal_hold(ps, current_signal=-0.20, pnl_pct=0.04) is True
        # -0.30 is below -0.25 → allow exit
        assert check_signal_hold(ps, current_signal=-0.30, pnl_pct=0.04) is False

    def test_empty_stops_dict(self):
        # No asym_managed key → returns False
        assert check_signal_hold({}, current_signal=-0.05, pnl_pct=0.05) is False
