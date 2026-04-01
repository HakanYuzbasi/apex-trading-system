"""
risk/asymmetric_sizing.py — Asymmetric Exit Level Calculator

Fixes the "avg win ≈ avg loss" problem by:
1. Confidence-scaled TP:   higher conviction → wider TP (up to 3:1 R:R)
2. Deferred trailing stop: only activates when price > 50% of way to TP
3. Wider trailing distance: 1.5× the initial stop distance (not a fixed 2%)
4. Breakeven lock:          stop moves to entry once price reaches 1× risk distance
5. Signal-hold override:    high-confidence positions require stronger signal reversal

All adjustments are opt-in via ApexConfig flags so they can be A/B tested.

Usage (called at entry time):
    from risk.asymmetric_sizing import compute_asymmetric_levels
    levels = compute_asymmetric_levels(
        entry_price=150.0, atr=2.5, signal=0.28, confidence=0.72,
        regime='bull', is_crypto=False
    )
    position_stops[symbol] = {**position_stops[symbol], **levels}

Returns a dict compatible with execution_loop position_stops format.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Config defaults (overridden by ApexConfig when available) ────────────────
_DEF = {
    "ASYMMETRIC_SIZING_ENABLED":        True,
    # ATR multipliers for stop distance
    "ASYM_ATR_STOP_MULT":               1.5,    # 1.5× ATR for stop
    "ASYM_ATR_STOP_MULT_CRYPTO":        2.0,    # crypto needs wider stops
    # R:R ratio  —  TP = stop_distance × RR_BASE + confidence × RR_CONF_SCALE
    "ASYM_RR_BASE":                     2.0,    # minimum 2:1 R:R
    "ASYM_RR_CONF_SCALE":               1.5,    # conf=1.0 → RR = 2.0+1.5 = 3.5:1
    # Trailing stop only activates when unrealised PnL > this fraction of TP distance
    "ASYM_TRAIL_ACTIVATION_FRAC":       0.55,   # must be >55% to TP before trailing kicks in
    # Trailing distance = stop_distance × this multiplier
    "ASYM_TRAIL_DIST_MULT":             1.2,    # slightly wider than the initial stop
    # Breakeven lock: move stop to entry when PnL > this fraction of stop_distance
    "ASYM_BREAKEVEN_LOCK_FRAC":         1.0,    # PnL > 1× stop distance → lock breakeven
    # Signal override: when in profit, require this signal level before exiting
    "ASYM_HOLD_SIGNAL_THRESH_HIGH":     -0.25,  # high-conf: need signal < -0.25 to exit
    "ASYM_HOLD_SIGNAL_THRESH_BASE":     -0.10,  # base: need signal < -0.10 to exit
    "ASYM_HIGH_CONF_THRESHOLD":         0.68,   # confidence above this = "high conviction"
    # Regime TP multipliers (bear regimes = tighter targets)
    "ASYM_REGIME_TP_MULT": {
        "strong_bull": 1.20,
        "bull":        1.10,
        "neutral":     1.00,
        "bear":        0.80,
        "strong_bear": 0.70,
        "volatile":    0.85,
        "crisis":      0.60,
    },
}


def _cfg(key: str):
    """Read from ApexConfig if available, else use defaults."""
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        if v is None:
            v = _DEF.get(key)
        return v
    except Exception:
        return _DEF.get(key)


def compute_asymmetric_levels(
    entry_price: float,
    atr: float,
    signal: float,
    confidence: float,
    regime: str = "neutral",
    is_crypto: bool = False,
    is_long: bool = True,
) -> Dict:
    """
    Compute asymmetric stop / TP / trailing levels for a new entry.

    Returns dict with keys matching execution_loop position_stops format:
        stop_loss, take_profit, trailing_stop_pct, breakeven_price,
        signal_hold_threshold, atr, rr_ratio, asym_managed
    """
    if not _cfg("ASYMMETRIC_SIZING_ENABLED"):
        return {}

    if entry_price <= 0 or atr <= 0:
        return {}

    # 1. Stop distance (ATR-based, clamped 0.5%–8%)
    atr_mult = _cfg("ASYM_ATR_STOP_MULT_CRYPTO") if is_crypto else _cfg("ASYM_ATR_STOP_MULT")
    stop_dist_pct = min(max(atr / entry_price * atr_mult, 0.005), 0.08)

    # 2. TP distance: confidence-scaled R:R, regime-adjusted
    rr = _cfg("ASYM_RR_BASE") + confidence * _cfg("ASYM_RR_CONF_SCALE")
    regime_mult = _cfg("ASYM_REGIME_TP_MULT").get(regime.lower(), 1.0)
    tp_dist_pct = stop_dist_pct * rr * regime_mult
    tp_dist_pct = min(max(tp_dist_pct, 0.02), 0.30)  # 2%–30%
    rr_actual = tp_dist_pct / stop_dist_pct

    # 3. Trailing activation: only once we're > 55% of the way to TP
    trail_activation_frac = _cfg("ASYM_TRAIL_ACTIVATION_FRAC")
    trailing_activation_pct = tp_dist_pct * trail_activation_frac

    # 4. Trailing distance: 1.2× the initial stop distance
    trailing_stop_pct = stop_dist_pct * _cfg("ASYM_TRAIL_DIST_MULT")

    # 5. Breakeven lock: move stop to entry once PnL > 1× stop distance
    breakeven_trigger_pct = stop_dist_pct * _cfg("ASYM_BREAKEVEN_LOCK_FRAC")
    if is_long:
        breakeven_price = entry_price  # stop moves to entry when profit > stop_dist
    else:
        breakeven_price = entry_price

    # 6. Signal hold threshold (how negative must signal be before we exit a winner)
    high_conf = confidence >= _cfg("ASYM_HIGH_CONF_THRESHOLD")
    signal_hold_threshold = (
        _cfg("ASYM_HOLD_SIGNAL_THRESH_HIGH") if high_conf
        else _cfg("ASYM_HOLD_SIGNAL_THRESH_BASE")
    )

    # Absolute levels
    direction = 1 if is_long else -1
    stop_loss  = round(entry_price * (1 - direction * stop_dist_pct), 6)
    take_profit = round(entry_price * (1 + direction * tp_dist_pct), 6)

    result = {
        "stop_loss":               stop_loss,
        "take_profit":             take_profit,
        "trailing_stop_pct":       round(trailing_stop_pct, 5),
        "trailing_activation_pct": round(trailing_activation_pct, 5),
        "breakeven_trigger_pct":   round(breakeven_trigger_pct, 5),
        "breakeven_price":         entry_price,
        "breakeven_locked":        False,
        "signal_hold_threshold":   signal_hold_threshold,
        "atr":                     atr,
        "rr_ratio":                round(rr_actual, 2),
        "asym_managed":            True,
    }

    logger.debug(
        "AsymLevels %s: entry=%.4f SL=%.4f TP=%.4f (RR=%.2f) trail=%.2f%% trail_act=%.2f%%",
        "LONG" if is_long else "SHORT", entry_price,
        stop_loss, take_profit, rr_actual,
        trailing_stop_pct * 100, trailing_activation_pct * 100,
    )

    return result


def update_breakeven_lock(
    pos_stops: Dict,
    current_price: float,
    entry_price: float,
    is_long: bool,
) -> bool:
    """
    Check and apply breakeven lock to position_stops in-place.

    Returns True if lock was just applied this call.
    """
    if not pos_stops.get("asym_managed") or pos_stops.get("breakeven_locked"):
        return False

    trigger = pos_stops.get("breakeven_trigger_pct", 0.0)
    if trigger <= 0 or entry_price <= 0:
        return False

    pnl_pct = (current_price / entry_price - 1) if is_long else (entry_price / current_price - 1)
    if pnl_pct >= trigger:
        pos_stops["stop_loss"] = round(entry_price, 6)
        pos_stops["breakeven_locked"] = True
        return True
    return False


def check_signal_hold(
    pos_stops: Dict,
    current_signal: float,
    pnl_pct: float,
) -> bool:
    """
    Return True if the asymmetric manager wants to HOLD (suppress early exit).

    Suppresses exits when:
    - Position is in profit (pnl_pct > 0)
    - Signal is negative but not strongly so (above signal_hold_threshold)
    """
    if not pos_stops.get("asym_managed"):
        return False
    if pnl_pct <= 0:
        return False  # Only protect winners

    threshold = pos_stops.get("signal_hold_threshold", -0.10)
    # If signal is negative but weaker than threshold, hold
    return current_signal >= threshold
