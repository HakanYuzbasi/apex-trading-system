"""
risk/adaptive_atr_stops.py — Adaptive ATR Stop Manager

Dynamically updates stop-loss distances for open positions each cycle based on:

  1. Regime — trending markets use wider ATR multiples (room to breathe),
     mean-reverting / volatile use tighter multiples (protect gains quickly).
  2. VIX level — elevated VIX widens stops to avoid noise-driven stop-outs.
  3. Profit lock-in — positions > 2% in profit get tightened trailing stops
     to ratchet in realised gains (higher floor as price advances).
  4. Ratchet rule — hard stops only move in the favourable direction; they
     are NEVER loosened for losing positions.

Wire-in (execution_loop, every N cycles for open positions):
    from risk.adaptive_atr_stops import AdaptiveATRStops
    _atr_mgr = AdaptiveATRStops()
    ...
    if cycle % ATR_UPDATE_INTERVAL == 0:
        for sym, qty in self.positions.items():
            if qty == 0 or sym not in self.historical_data:
                continue
            ps = self.position_stops.get(sym, {})
            if not ps:
                continue
            ep = self.position_entry_prices.get(sym, 0)
            cp = self.historical_data[sym]["Close"].iloc[-1]
            pnl_pct = (cp - ep) / ep if ep > 0 else 0.0
            if qty < 0:
                pnl_pct = -pnl_pct
            updated = _atr_mgr.update_stop(
                symbol=sym,
                pos_stops=ps,
                current_price=cp,
                prices=self.historical_data[sym]["Close"],
                regime=self._current_regime or "neutral",
                vix=self._current_vix or 20.0,
                pnl_pct=pnl_pct,
                is_long=qty > 0,
            )
            self.position_stops[sym] = updated
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── ATR multiples by regime ────────────────────────────────────────────────────
# trending regimes → wider stops (let positions breathe with trend)
# mean-reverting / volatile → tighter (lock in gains fast)
_ATR_MULT: Dict[str, float] = {
    "strong_bull": 2.00,
    "bull":        1.80,
    "neutral":     1.50,   # mean-reverting regime → tightest
    "bear":        1.80,
    "strong_bear": 2.00,
    "volatile":    2.40,   # wide: high noise, avoid stop-outs
    "crisis":      2.80,   # very wide: intraday swings huge
}
_DEFAULT_ATR_MULT = 1.80

# Stop bounds (fraction of price)
_MIN_STOP_PCT = 0.004   # 0.4% minimum stop distance
_MAX_STOP_PCT = 0.12    # 12% maximum stop distance

# Profit tiers for trailing tightening
_PROFIT_TIERS = [
    (0.08, 0.50),  # pnl > 8% → trailing = 50% of normal distance
    (0.05, 0.65),  # pnl > 5% → trailing = 65%
    (0.03, 0.80),  # pnl > 3% → trailing = 80%
    (0.015, 0.90), # pnl > 1.5% → trailing = 90%
]


class AdaptiveATRStops:
    """
    Regime-aware, profit-ratcheting ATR stop manager.
    Stateless — one instance can be shared across all symbols.
    """

    def __init__(
        self,
        atr_period: int = 14,
        min_stop_pct: float = _MIN_STOP_PCT,
        max_stop_pct: float = _MAX_STOP_PCT,
    ) -> None:
        self._atr_period = atr_period
        self._min_stop_pct = min_stop_pct
        self._max_stop_pct = max_stop_pct

    @property
    def min_stop_pct(self) -> float:
        return self._min_stop_pct

    @property
    def max_stop_pct(self) -> float:
        return self._max_stop_pct

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_stop_distance(
        self,
        atr_pct: float,
        regime: str,
        vix: float,
        pnl_pct: float = 0.0,
    ) -> float:
        """
        Compute the adaptive stop distance as a fraction of current price.

        Args:
            atr_pct: ATR expressed as a fraction of current price.
            regime:  Current market regime string.
            vix:     Current VIX level.
            pnl_pct: Current P&L as a fraction of entry price (positive=profit).

        Returns:
            Stop distance fraction [min_stop_pct, max_stop_pct].
        """
        base_mult = _ATR_MULT.get(regime.lower(), _DEFAULT_ATR_MULT)

        # VIX overlay: widen when VIX elevated above 20 baseline
        # VIX=20 → factor=1.0, VIX=40 → factor=1.50, VIX=60 → factor=2.0
        vix_factor = 1.0 + max(0.0, (vix - 20.0) / 40.0)
        vix_factor = min(vix_factor, 2.0)

        # Profit tightening: lock in gains by trailing closer
        profit_factor = 1.0
        for threshold, factor in _PROFIT_TIERS:
            if pnl_pct >= threshold:
                profit_factor = factor
                break

        raw = atr_pct * base_mult * vix_factor * profit_factor
        return float(np.clip(raw, self._min_stop_pct, self._max_stop_pct))

    def update_stop(
        self,
        symbol: str,
        pos_stops: Dict,
        current_price: float,
        prices: "pd.Series",
        regime: str,
        vix: float,
        pnl_pct: float,
        is_long: bool = True,
    ) -> Dict:
        """
        Recompute and update stops for one open position.

        Ratchet rule:
          - Hard stop only moves in the favourable direction (never loosens).
          - Trailing stop percentage is always refreshed to the latest ATR regime.

        Args:
            symbol:        Symbol (for logging only).
            pos_stops:     Current stops dict {stop_loss, take_profit, trailing_stop_pct, atr}.
            current_price: Latest market price.
            prices:        Close price Series for ATR calculation.
            regime:        Current regime string.
            vix:           Current VIX.
            pnl_pct:       Unrealised P&L fraction (positive = position is winning).
            is_long:       True for LONG positions.

        Returns:
            Updated stops dict (copy — original is not mutated).
        """
        atr = self._calc_atr(prices)
        atr_pct = atr / current_price if current_price > 0 else 0.02

        stop_distance = self.compute_stop_distance(atr_pct, regime, vix, pnl_pct)

        updated = dict(pos_stops)   # shallow copy — never mutate caller's dict
        updated["atr"] = round(float(atr), 6)
        updated["trailing_stop_pct"] = round(stop_distance, 6)

        # Ratchet hard stop
        if is_long:
            new_floor = current_price * (1.0 - stop_distance)
            existing = float(pos_stops.get("stop_loss") or 0.0)
            if pnl_pct > 0.0:
                # In profit → ratchet stop up (never down)
                updated["stop_loss"] = round(max(existing, new_floor), 6)
            # Losing position → leave stop unchanged to avoid noise stop-out
        else:
            new_ceiling = current_price * (1.0 + stop_distance)
            existing = float(pos_stops.get("stop_loss") or float("inf"))
            if pnl_pct > 0.0:
                # Short in profit → ratchet stop down (never up)
                updated["stop_loss"] = round(min(existing, new_ceiling), 6)

        logger.debug(
            "ATR stop update [%s]: atr_pct=%.3f%% regime=%s vix=%.1f pnl=%.2f%% "
            "→ dist=%.3f%% stop=%.4f trail=%.3f%%",
            symbol,
            atr_pct * 100,
            regime,
            vix,
            pnl_pct * 100,
            stop_distance * 100,
            updated.get("stop_loss", 0.0),
            stop_distance * 100,
        )
        return updated

    def update_all(
        self,
        positions: Dict[str, float],
        position_stops: Dict[str, Dict],
        entry_prices: Dict[str, float],
        historical_data: Dict,
        regime: str,
        vix: float,
    ) -> Dict[str, Dict]:
        """
        Convenience batch updater — returns a dict of symbol → updated stops.
        Only processes symbols that already have existing stops.
        """
        updated_stops: Dict[str, Dict] = {}
        for symbol, qty in positions.items():
            if qty == 0:
                continue
            ps = position_stops.get(symbol)
            if not ps:
                continue
            hdata = historical_data.get(symbol)
            if hdata is None:
                continue
            prices = hdata.get("Close") if isinstance(hdata, dict) else getattr(hdata, "Close", None)
            if prices is None or len(prices) < self._atr_period + 1:
                continue
            ep = float(entry_prices.get(symbol) or prices.iloc[-1])
            cp = float(prices.iloc[-1])
            pnl_pct = (cp - ep) / ep if ep > 0 else 0.0
            if qty < 0:
                pnl_pct = -pnl_pct
            updated_stops[symbol] = self.update_stop(
                symbol=symbol,
                pos_stops=ps,
                current_price=cp,
                prices=prices,
                regime=regime,
                vix=vix,
                pnl_pct=pnl_pct,
                is_long=qty > 0,
            )
        return updated_stops

    # ── Internals ─────────────────────────────────────────────────────────────

    def _calc_atr(self, prices: "pd.Series") -> float:
        """
        ATR approximation using close-to-close (no high/low available).
        Falls back to 2% of last close when insufficient data.
        """
        n = min(self._atr_period, len(prices) - 1)
        if n < 2:
            return float(prices.iloc[-1]) * 0.02
        diffs = np.abs(np.diff(prices.values[-n - 1:]))
        return float(np.mean(diffs))

    @staticmethod
    def regime_is_trending(regime: str) -> bool:
        """True for directional trend regimes (wider stops expected)."""
        return regime.lower() in ("strong_bull", "bull", "bear", "strong_bear")

    @staticmethod
    def regime_is_mean_reverting(regime: str) -> bool:
        """True for neutral/mean-reverting (tighter stops expected)."""
        return regime.lower() == "neutral"
