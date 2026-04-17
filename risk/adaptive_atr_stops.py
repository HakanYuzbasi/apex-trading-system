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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import ApexConfig

logger = logging.getLogger(__name__)


def _regime_mult_map() -> Dict[str, float]:
    """Build the regime → ATR-multiple map from ApexConfig on every read.

    Reading fresh from config allows hot-reload of env overrides without
    restarting the process. Includes ``volatile`` and ``high_volatility``
    aliases so the dynamic_exit_manager regime vocabulary maps correctly.
    """
    vol_mult = float(ApexConfig.ATR_REGIME_MULT_VOLATILE)
    return {
        "strong_bull":     float(ApexConfig.ATR_REGIME_MULT_STRONG_BULL),
        "bull":            float(ApexConfig.ATR_REGIME_MULT_BULL),
        "neutral":         float(ApexConfig.ATR_REGIME_MULT_NEUTRAL),
        "bear":            float(ApexConfig.ATR_REGIME_MULT_BEAR),
        "strong_bear":     float(ApexConfig.ATR_REGIME_MULT_STRONG_BEAR),
        "volatile":        vol_mult,
        "high_volatility": vol_mult,   # alias for dynamic_exit_manager key
        "crisis":          float(ApexConfig.ATR_REGIME_MULT_CRISIS),
    }


def _parse_profit_tiers(raw: str) -> List[Tuple[float, float]]:
    """
    Parse ``"0.12:0.40,0.08:0.50,..."`` into a descending-threshold tier list.

    Args:
        raw: Comma-separated ``threshold:factor`` pairs.

    Returns:
        List of ``(threshold, tighten_factor)`` pairs sorted by threshold
        descending so the first match on iteration is the deepest tier.

    Raises:
        ValueError: If a pair is malformed or values are out of ``[0, 1]``.
    """
    tiers: List[Tuple[float, float]] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"ATR_PROFIT_TIERS entry missing ':': {piece!r}")
        t_s, f_s = piece.split(":", 1)
        try:
            t = float(t_s)
            f = float(f_s)
        except ValueError as exc:
            raise ValueError(f"ATR_PROFIT_TIERS parse error on {piece!r}: {exc}") from exc
        if not (0.0 <= t <= 1.0) or not (0.0 < f <= 1.0):
            raise ValueError(f"ATR_PROFIT_TIERS out of range: {piece!r}")
        tiers.append((t, f))
    tiers.sort(key=lambda x: x[0], reverse=True)
    return tiers


class AdaptiveATRStops:
    """
    Regime-aware, profit-ratcheting ATR stop manager.
    Stateless — one instance can be shared across all symbols.

    Uses Wilder's True-Range ATR when High/Low series are available
    (preserving intraday range information) and falls back to an EWMA of
    absolute close-to-close differences for Close-only inputs.
    """

    def __init__(
        self,
        atr_period: Optional[int] = None,
        min_stop_pct: Optional[float] = None,
        max_stop_pct: Optional[float] = None,
    ) -> None:
        self._atr_period: int = int(
            atr_period if atr_period is not None else ApexConfig.ATR_STOP_PERIOD
        )
        self._min_stop_pct: float = float(
            min_stop_pct if min_stop_pct is not None else ApexConfig.ATR_MIN_STOP_PCT
        )
        self._max_stop_pct: float = float(
            max_stop_pct if max_stop_pct is not None else ApexConfig.ATR_MAX_STOP_PCT
        )
        if self._atr_period < 2:
            raise ValueError(f"atr_period must be >= 2, got {self._atr_period}")
        if self._min_stop_pct <= 0.0 or self._max_stop_pct <= self._min_stop_pct:
            raise ValueError(
                f"invalid stop bounds min={self._min_stop_pct} max={self._max_stop_pct}"
            )

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
            atr_pct: ATR expressed as a fraction of current price. Clamped
                at ``>= 0``.
            regime:  Current market regime string (case-insensitive).
                Unknown regimes fall back to ``ATR_REGIME_MULT_DEFAULT``.
            vix:     Current VIX level. Negative values are treated as 0.
            pnl_pct: Current P&L as a fraction of entry price (positive=profit).

        Returns:
            Stop distance fraction in ``[min_stop_pct, max_stop_pct]``.
        """
        _atr_pct = max(0.0, float(atr_pct))
        base_mult = _regime_mult_map().get(
            str(regime).lower(),
            float(ApexConfig.ATR_REGIME_MULT_DEFAULT),
        )

        baseline = float(ApexConfig.ATR_VIX_BASELINE)
        scale = float(ApexConfig.ATR_VIX_SCALE)
        cap = float(ApexConfig.ATR_VIX_FACTOR_MAX)
        safe_vix = max(0.0, float(vix)) if np.isfinite(vix) else baseline
        vix_factor = 1.0 + max(0.0, (safe_vix - baseline) / max(scale, 1e-8))
        vix_factor = min(vix_factor, cap)

        # Profit tightening: lock in gains by trailing closer
        profit_factor = 1.0
        try:
            tiers = _parse_profit_tiers(str(ApexConfig.ATR_PROFIT_TIERS))
        except ValueError as exc:
            logger.warning("ATR_PROFIT_TIERS malformed (%s) — using flat 1.0", exc)
            tiers = []
        for threshold, factor in tiers:
            if pnl_pct >= threshold:
                profit_factor = factor
                break

        raw = _atr_pct * base_mult * vix_factor * profit_factor
        return float(np.clip(raw, self._min_stop_pct, self._max_stop_pct))

    def update_stop(
        self,
        symbol: str,
        pos_stops: Dict,
        current_price: float,
        prices: "Union[pd.Series, pd.DataFrame]",
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
            prices:        Either a Close ``pd.Series`` OR a full OHLC
                ``pd.DataFrame``. A DataFrame unlocks True-Range ATR
                (Wilder) — the preferred form.
            regime:        Current regime string.
            vix:           Current VIX.
            pnl_pct:       Unrealised P&L fraction (positive = position is winning).
            is_long:       True for LONG positions.

        Returns:
            Updated stops dict (copy — original is not mutated).
        """
        closes, highs, lows = self._extract_ohlc(prices)
        atr = self._calc_atr(closes, highs, lows)
        atr_pct = (
            atr / current_price
            if current_price > 0 and np.isfinite(current_price)
            else float(ApexConfig.ATR_FALLBACK_PCT)
        )

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
            # Pass the full OHLC container when available — unlocks True-Range ATR.
            if isinstance(hdata, pd.DataFrame) and {"Close"}.issubset(hdata.columns):
                ohlc_for_atr: "Union[pd.Series, pd.DataFrame]" = hdata
                closes_only = hdata["Close"]
            elif isinstance(hdata, dict) and "Close" in hdata:
                # Rebuild a minimal DataFrame if H/L are present, else pass Close series.
                cols = {"Close": hdata["Close"]}
                if "High" in hdata and "Low" in hdata:
                    cols["High"] = hdata["High"]
                    cols["Low"] = hdata["Low"]
                    ohlc_for_atr = pd.DataFrame(cols)
                else:
                    ohlc_for_atr = hdata["Close"]
                closes_only = hdata["Close"]
            else:
                closes_only = getattr(hdata, "Close", None)
                if closes_only is None:
                    continue
                ohlc_for_atr = closes_only

            if closes_only is None or len(closes_only) < self._atr_period + 1:
                continue
            ep = float(entry_prices.get(symbol) or closes_only.iloc[-1])
            cp = float(closes_only.iloc[-1])
            pnl_pct = (cp - ep) / ep if ep > 0 else 0.0
            if qty < 0:
                pnl_pct = -pnl_pct
            updated_stops[symbol] = self.update_stop(
                symbol=symbol,
                pos_stops=ps,
                current_price=cp,
                prices=ohlc_for_atr,
                regime=regime,
                vix=vix,
                pnl_pct=pnl_pct,
                is_long=qty > 0,
            )
        return updated_stops

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_ohlc(
        prices: "Union[pd.Series, pd.DataFrame]",
    ) -> Tuple["pd.Series", Optional["pd.Series"], Optional["pd.Series"]]:
        """
        Extract ``(closes, highs, lows)`` from a DataFrame/Series input.

        Args:
            prices: Either a Close ``pd.Series`` or a DataFrame with at
                least a ``Close`` column (optionally ``High`` and ``Low``).

        Returns:
            Tuple ``(closes, highs, lows)``. ``highs`` and ``lows`` are
            ``None`` when they cannot be recovered.

        Raises:
            ValueError: If ``prices`` is neither a Series nor a DataFrame
                exposing a Close column.
        """
        if isinstance(prices, pd.DataFrame):
            if "Close" not in prices.columns:
                raise ValueError("DataFrame passed to _calc_atr must have 'Close' column")
            closes = prices["Close"]
            highs = prices["High"] if "High" in prices.columns else None
            lows = prices["Low"] if "Low" in prices.columns else None
            return closes, highs, lows
        if isinstance(prices, pd.Series):
            return prices, None, None
        raise ValueError(f"prices must be pd.Series or pd.DataFrame, got {type(prices).__name__}")

    def _calc_atr(
        self,
        closes: "pd.Series",
        highs: Optional["pd.Series"] = None,
        lows: Optional["pd.Series"] = None,
    ) -> float:
        """
        Average True Range (Wilder) when full OHLC is available, or EWMA
        of |close-to-close| differences when only closes are provided.

        ``TR_i = max(H_i - L_i, |H_i - C_{i-1}|, |L_i - C_{i-1}|)``

        Close-to-close differencing — the previous behaviour of this method
        — systematically understates realised range by ~30–50% because it
        ignores intraday swings, causing stops to be placed too close to
        price and triggering false noise exits on winners.

        Args:
            closes: Close price series (required).
            highs:  High price series (optional — enables True Range).
            lows:   Low price series (optional — enables True Range).

        Returns:
            ATR in **price units** (same scale as closes). Returns
            ``last_close * ATR_FALLBACK_PCT`` when there is not enough
            data to form a valid ATR.
        """
        if closes is None or len(closes) < 2:
            if closes is not None and len(closes) >= 1:
                return float(closes.iloc[-1]) * float(ApexConfig.ATR_FALLBACK_PCT)
            return float(ApexConfig.ATR_FALLBACK_PCT)

        period = int(self._atr_period)
        closes_arr = np.asarray(closes, dtype=float)

        if highs is not None and lows is not None and len(highs) == len(closes) and len(lows) == len(closes):
            highs_arr = np.asarray(highs, dtype=float)
            lows_arr = np.asarray(lows, dtype=float)
            # True Range with proper prev-close alignment
            prev_close = np.concatenate([[closes_arr[0]], closes_arr[:-1]])
            tr1 = highs_arr - lows_arr
            tr2 = np.abs(highs_arr - prev_close)
            tr3 = np.abs(lows_arr - prev_close)
            true_range = np.maximum.reduce([tr1, tr2, tr3])
            # Drop the synthetic first bar (prev_close==close) to avoid bias
            true_range = true_range[1:]
            if true_range.size < 2:
                return float(closes_arr[-1]) * float(ApexConfig.ATR_FALLBACK_PCT)
            window = min(period, true_range.size)
            # Wilder's smoothing: RMA = EWMA with alpha = 1/period
            series = pd.Series(true_range)
            atr_val = float(series.ewm(alpha=1.0 / float(window), adjust=False).mean().iloc[-1])
            if not np.isfinite(atr_val) or atr_val <= 0.0:
                return float(closes_arr[-1]) * float(ApexConfig.ATR_FALLBACK_PCT)
            return atr_val

        # Close-only fallback: Wilder-smoothed absolute close-to-close diff.
        diffs = np.abs(np.diff(closes_arr))
        if diffs.size < 2:
            return float(closes_arr[-1]) * float(ApexConfig.ATR_FALLBACK_PCT)
        window = min(period, diffs.size)
        series = pd.Series(diffs)
        atr_val = float(series.ewm(alpha=1.0 / float(window), adjust=False).mean().iloc[-1])
        if not np.isfinite(atr_val) or atr_val <= 0.0:
            return float(closes_arr[-1]) * float(ApexConfig.ATR_FALLBACK_PCT)
        return atr_val

    @staticmethod
    def regime_is_trending(regime: str) -> bool:
        """True for directional trend regimes (wider stops expected)."""
        return regime.lower() in ("strong_bull", "bull", "bear", "strong_bear")

    @staticmethod
    def regime_is_mean_reverting(regime: str) -> bool:
        """True for neutral/mean-reverting (tighter stops expected)."""
        return regime.lower() == "neutral"
