"""
core/signal_cascade.py
──────────────────────
Cross-Asset Signal Cascade Engine.

When a major macro asset (SPY = equities, BTC/USD = crypto) moves sharply in
the last N bars, this engine propagates a signal adjustment to all correlated
instruments so they can react even before their own signal refreshes.

Design
──────
  1. call `record_price(symbol, price)` every cycle for all symbols.
  2. call `get_cascade_adjustment(symbol)` to get a float multiplier in
     [cascade_floor, cascade_ceiling] applied to the symbol's composite signal.
     Returns 1.0 (neutral) when conditions are not met.
  3. Optionally call `record_signal_update(symbol, signal)` so the engine can
     learn which symbols respond most to cascade events.

Correlation tracking
────────────────────
  • Rolling `window_bars` bars of log-returns per symbol.
  • Pearson correlation computed on demand between anchor (SPY/BTC) and each
    tracked symbol.  Cached for `correlation_cache_cycles` cycles.
  • Correlation is sign-corrected: a strong negative correlation (e.g. a bond
    vs equities) still propagates the cascade with inverted direction.

Cascade trigger
───────────────
  • Compute the anchor's N-bar cumulative log-return.
  • If |return| > threshold  → cascade active.
  • Each symbol's adjustment = 1.0  +  corr × anchor_return × gain
    clamped to [floor, ceiling].

Python stdlib only — no pandas/scipy required.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_EQUITY_ANCHOR = "SPY"
_CRYPTO_ANCHOR = "BTC/USD"
_CRYPTO_PREFIX = "CRYPTO:"

_DEFAULT_WINDOW_BARS        = 6      # look-back window for cascade detection
_DEFAULT_TRIGGER_THRESHOLD  = 0.005  # 0.5% cumulative move triggers cascade
_DEFAULT_CASCADE_GAIN       = 2.0    # how much to amplify the correlation effect
_DEFAULT_FLOOR              = 0.50   # minimum multiplier (strong dampen)
_DEFAULT_CEILING            = 1.50   # maximum multiplier (strong amplify)
_DEFAULT_MIN_CORRELATION    = 0.30   # |corr| below this → no adjustment
_DEFAULT_CACHE_CYCLES       = 5      # recalculate correlation every N cycles

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_log_return(prev: float, curr: float) -> float:
    if prev <= 0 or curr <= 0:
        return 0.0
    return math.log(curr / prev)


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x < 1e-12 or denom_y < 1e-12:
        return 0.0
    return num / (denom_x * denom_y)


# ── Main class ─────────────────────────────────────────────────────────────────

class SignalCascadeEngine:
    """
    Cross-asset signal cascade engine.

    Parameters
    ──────────
    window_bars         : look-back bars for cumulative return
    trigger_threshold   : |cumulative return| that activates cascade
    cascade_gain        : amplification factor for the adjustment
    floor / ceiling     : clamp bounds for the adjustment multiplier
    min_correlation     : |corr| below this → symbol is not affected
    correlation_cache_cycles : recalculate correlations every N cycles
    max_price_history   : rolling buffer depth per symbol
    """

    def __init__(
        self,
        window_bars: int             = _DEFAULT_WINDOW_BARS,
        trigger_threshold: float     = _DEFAULT_TRIGGER_THRESHOLD,
        cascade_gain: float          = _DEFAULT_CASCADE_GAIN,
        floor: float                 = _DEFAULT_FLOOR,
        ceiling: float               = _DEFAULT_CEILING,
        min_correlation: float       = _DEFAULT_MIN_CORRELATION,
        correlation_cache_cycles: int = _DEFAULT_CACHE_CYCLES,
        max_price_history: int       = 200,
    ) -> None:
        self.window_bars              = window_bars
        self.trigger_threshold        = trigger_threshold
        self.cascade_gain             = cascade_gain
        self.floor                    = floor
        self.ceiling                  = ceiling
        self.min_correlation          = min_correlation
        self.correlation_cache_cycles = correlation_cache_cycles
        self._max_history             = max_price_history

        # price history per symbol: deque of floats
        self._prices: Dict[str, deque] = {}

        # cached log-return series per symbol
        self._returns_cache: Dict[str, List[float]] = {}

        # cached correlations: (anchor_symbol → {symbol: corr})
        self._corr_cache: Dict[str, Dict[str, float]] = {}
        self._corr_cache_age: Dict[str, int] = {}   # cycles since last compute
        self._cycle: int = 0

        # cascade state: last active cascade per anchor
        self._active_cascade: Dict[str, Tuple[float, float]] = {}
        # {anchor: (anchor_return, timestamp)}

    # ── Price ingestion ────────────────────────────────────────────────────────

    def record_price(self, symbol: str, price: float) -> None:
        """Record the latest price for a symbol.  Call once per cycle."""
        if price <= 0:
            return
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self._max_history)
        self._prices[symbol].append(float(price))
        # invalidate return cache
        if symbol in self._returns_cache:
            del self._returns_cache[symbol]

    def tick(self) -> None:
        """Advance the internal cycle counter (call once per main loop cycle)."""
        self._cycle += 1
        # Age out correlation caches
        for anchor in list(self._corr_cache_age):
            self._corr_cache_age[anchor] += 1

    # ── Log returns ────────────────────────────────────────────────────────────

    def _get_returns(self, symbol: str) -> List[float]:
        """Return the log-return series for `symbol`, using cache."""
        if symbol in self._returns_cache:
            return self._returns_cache[symbol]
        prices = list(self._prices.get(symbol, []))
        if len(prices) < 2:
            return []
        rets = [_safe_log_return(prices[i - 1], prices[i]) for i in range(1, len(prices))]
        self._returns_cache[symbol] = rets
        return rets

    def _anchor_return(self, anchor: str) -> Optional[float]:
        """Cumulative log-return over last `window_bars` bars for anchor."""
        rets = self._get_returns(anchor)
        if len(rets) < self.window_bars:
            return None
        window = rets[-self.window_bars:]
        return sum(window)

    # ── Correlation ────────────────────────────────────────────────────────────

    def _get_correlation(self, anchor: str, symbol: str) -> float:
        """Return cached Pearson correlation between anchor and symbol."""
        age = self._corr_cache_age.get(anchor, 999)
        if age < self.correlation_cache_cycles and anchor in self._corr_cache:
            return self._corr_cache[anchor].get(symbol, 0.0)

        # Recompute full batch
        self._recompute_correlations(anchor)
        return self._corr_cache.get(anchor, {}).get(symbol, 0.0)

    def _recompute_correlations(self, anchor: str) -> None:
        """Recompute correlations of all tracked symbols against `anchor`."""
        anchor_rets = self._get_returns(anchor)
        if len(anchor_rets) < max(10, self.window_bars):
            self._corr_cache[anchor] = {}
            self._corr_cache_age[anchor] = 0
            return

        corrs: Dict[str, float] = {}
        for sym, _ in self._prices.items():
            if sym == anchor:
                continue
            sym_rets = self._get_returns(sym)
            min_len  = min(len(anchor_rets), len(sym_rets))
            if min_len < 10:
                continue
            corrs[sym] = _pearson(anchor_rets[-min_len:], sym_rets[-min_len:])

        self._corr_cache[anchor]     = corrs
        self._corr_cache_age[anchor] = 0
        logger.debug(
            "SignalCascade: recomputed %d correlations for anchor=%s",
            len(corrs), anchor,
        )

    # ── Main API ───────────────────────────────────────────────────────────────

    def _is_crypto(self, symbol: str) -> bool:
        return symbol.startswith(_CRYPTO_PREFIX) or "/" in symbol

    def _anchor_for(self, symbol: str) -> str:
        return _CRYPTO_ANCHOR if self._is_crypto(symbol) else _EQUITY_ANCHOR

    def get_cascade_adjustment(self, symbol: str) -> float:
        """
        Return a multiplier in [floor, ceiling] for `symbol`'s signal.
        Returns 1.0 when no cascade is active or symbol is uncorrelated.
        """
        anchor = self._anchor_for(symbol)
        if anchor == symbol:
            return 1.0  # anchor adjusts itself — skip

        anchor_ret = self._anchor_return(anchor)
        if anchor_ret is None or abs(anchor_ret) < self.trigger_threshold:
            return 1.0  # no significant macro move

        corr = self._get_correlation(anchor, symbol)
        if abs(corr) < self.min_correlation:
            return 1.0  # uncorrelated

        # adjustment = corr × anchor_return × gain
        raw_adj  = corr * anchor_ret * self.cascade_gain
        mult     = 1.0 + raw_adj
        clamped  = max(self.floor, min(self.ceiling, mult))

        logger.debug(
            "SignalCascade %s anchor=%s ret=%.4f corr=%.3f adj=%.3f mult=%.3f",
            symbol, anchor, anchor_ret, corr, raw_adj, clamped,
        )
        return clamped

    def get_cascade_info(self, symbol: str) -> Dict:
        """Diagnostic snapshot for a symbol."""
        anchor     = self._anchor_for(symbol)
        anchor_ret = self._anchor_return(anchor)
        corr       = self._get_correlation(anchor, symbol) if anchor_ret is not None else 0.0
        mult       = self.get_cascade_adjustment(symbol)
        return {
            "symbol":       symbol,
            "anchor":       anchor,
            "anchor_return": anchor_ret,
            "correlation":  corr,
            "multiplier":   mult,
            "cascade_active": anchor_ret is not None and abs(anchor_ret) >= self.trigger_threshold,
            "price_bars":   len(self._prices.get(symbol, [])),
            "anchor_bars":  len(self._prices.get(anchor, [])),
        }

    def get_all_adjustments(self) -> Dict[str, float]:
        """Return cascade multipliers for all tracked symbols."""
        return {sym: self.get_cascade_adjustment(sym) for sym in self._prices}

    def summary(self) -> Dict:
        """High-level status snapshot."""
        return {
            "cycle":           self._cycle,
            "tracked_symbols": len(self._prices),
            "equity_anchor_bars": len(self._prices.get(_EQUITY_ANCHOR, [])),
            "crypto_anchor_bars": len(self._prices.get(_CRYPTO_ANCHOR, [])),
            "equity_anchor_return": self._anchor_return(_EQUITY_ANCHOR),
            "crypto_anchor_return": self._anchor_return(_CRYPTO_ANCHOR),
            "equity_cascade_active": (
                (self._anchor_return(_EQUITY_ANCHOR) or 0.0)
                >= self.trigger_threshold
            ),
            "crypto_cascade_active": (
                (self._anchor_return(_CRYPTO_ANCHOR) or 0.0)
                >= self.trigger_threshold
            ),
        }
