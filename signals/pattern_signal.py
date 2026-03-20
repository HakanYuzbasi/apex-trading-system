"""
signals/pattern_signal.py — Candlestick Pattern Signal
=======================================================
Detects 7 high-value candlestick reversal patterns on OHLCV data.
Pure numpy/pandas — ZERO external dependencies, zero API calls.

Patterns fire only when trend context is appropriate:
  - Bullish patterns: only in downtrend (last N bars negative)
  - Bearish patterns: only in uptrend (last N bars positive)

This is discrete OHLCV event detection — completely independent
from the ML feature matrix which uses continuous indicators.

Usage:
    ps = PatternSignal()
    result = ps.get_signal("BTC/USD", df_ohlcv)
    print(result.signal, result.patterns_found)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
@dataclass
class PatternResult:
    symbol: str
    signal: float              # [-1, 1]: +1 = strong bullish, -1 = strong bearish
    confidence: float          # [0, 1]
    patterns_found: List[str]  # names of detected patterns
    dominant: str              # strongest-confidence pattern or "none"
    direction_votes: Dict[str, float] = field(default_factory=dict)


def _neutral_pattern(symbol: str) -> PatternResult:
    return PatternResult(
        symbol=symbol, signal=0.0, confidence=0.0,
        patterns_found=[], dominant="none",
    )


# ---------------------------------------------------------------------------
# Candle helpers (all operate on numpy scalars)
# ---------------------------------------------------------------------------

def _body(o: float, c: float) -> float:
    return abs(c - o)


def _upper_shadow(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_shadow(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def _range(h: float, l: float) -> float:
    return h - l


def _is_bullish(o: float, c: float) -> bool:
    return c > o


def _is_bearish(o: float, c: float) -> bool:
    return c < o


# ---------------------------------------------------------------------------
# Trend context
# ---------------------------------------------------------------------------

def _trend_slope(closes: np.ndarray, window: int = 5) -> float:
    """
    Normalised linear-regression slope over last `window` bars.
    Positive = uptrend, negative = downtrend.
    """
    if len(closes) < window:
        return 0.0
    seg = closes[-window:].astype(float)
    mean_px = float(np.mean(seg)) or 1.0
    x = np.arange(len(seg), dtype=float)
    try:
        slope = float(np.polyfit(x, seg, 1)[0])
        return slope / mean_px
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Individual pattern detectors
# Each returns (signal_direction: float[-1,1], raw_confidence: float[0,1])
# or (0, 0) if pattern not found.
# ---------------------------------------------------------------------------

def _hammer(o: float, h: float, l: float, c: float, trend: float) -> Tuple[float, float]:
    """Hammer (bullish) — small body near top, long lower shadow. Needs downtrend."""
    if trend >= 0:  # require downtrend
        return 0.0, 0.0
    r = _range(h, l)
    if r == 0:
        return 0.0, 0.0
    body = _body(o, c)
    lower = _lower_shadow(o, l, c)
    upper = _upper_shadow(o, h, c)
    # Hammer: lower shadow ≥ 2× body, upper shadow ≤ 0.5× body, body ≤ 40% of range
    if lower >= 2.0 * body and upper <= 0.5 * body and body / r <= 0.40:
        conf = float(min(lower / (2.0 * max(body, r * 0.01)), 1.0)) * 0.80
        return 1.0, conf
    return 0.0, 0.0


def _shooting_star(o: float, h: float, l: float, c: float, trend: float) -> Tuple[float, float]:
    """Shooting Star (bearish) — small body near bottom, long upper shadow. Needs uptrend."""
    if trend <= 0:  # require uptrend
        return 0.0, 0.0
    r = _range(h, l)
    if r == 0:
        return 0.0, 0.0
    body = _body(o, c)
    upper = _upper_shadow(o, h, c)
    lower = _lower_shadow(o, l, c)
    if upper >= 2.0 * body and lower <= 0.5 * body and body / r <= 0.40:
        conf = float(min(upper / (2.0 * max(body, r * 0.01)), 1.0)) * 0.80
        return -1.0, conf
    return 0.0, 0.0


def _bullish_engulfing(
    o0: float, c0: float,  # prior candle
    o1: float, c1: float,  # current candle
    trend: float,
) -> Tuple[float, float]:
    """Bullish Engulfing — current bullish body engulfs prior bearish body. Needs downtrend."""
    if trend >= 0:
        return 0.0, 0.0
    if not (_is_bearish(o0, c0) and _is_bullish(o1, c1)):
        return 0.0, 0.0
    prior_body = _body(o0, c0)
    curr_body = _body(o1, c1)
    # Current body must fully contain prior body
    if c1 > o0 and o1 < c0 and curr_body >= prior_body:
        conf = float(min(curr_body / max(prior_body, 1e-8), 2.0) / 2.0) * 0.85
        return 1.0, conf
    return 0.0, 0.0


def _bearish_engulfing(
    o0: float, c0: float,
    o1: float, c1: float,
    trend: float,
) -> Tuple[float, float]:
    """Bearish Engulfing — current bearish body engulfs prior bullish body. Needs uptrend."""
    if trend <= 0:
        return 0.0, 0.0
    if not (_is_bullish(o0, c0) and _is_bearish(o1, c1)):
        return 0.0, 0.0
    prior_body = _body(o0, c0)
    curr_body = _body(o1, c1)
    if o1 > c0 and c1 < o0 and curr_body >= prior_body:
        conf = float(min(curr_body / max(prior_body, 1e-8), 2.0) / 2.0) * 0.85
        return -1.0, conf
    return 0.0, 0.0


def _doji(o: float, h: float, l: float, c: float, trend: float) -> Tuple[float, float]:
    """
    Doji — body < 5% of total range (indecision).
    Direction is context-driven: doji in downtrend = potential bullish reversal, vice versa.
    Lower confidence than directional patterns.
    """
    r = _range(h, l)
    if r == 0:
        return 0.0, 0.0
    body = _body(o, c)
    if body / r < 0.05:
        direction = 1.0 if trend < 0 else -1.0 if trend > 0 else 0.0
        conf = 0.30  # doji alone is weak signal
        return direction, conf
    return 0.0, 0.0


def _morning_star(
    o0: float, c0: float,             # bar -2: large bearish
    o1: float, h1: float, l1: float, c1: float,   # bar -1: small body (star)
    o2: float, c2: float,             # bar 0: large bullish
    trend: float,
) -> Tuple[float, float]:
    """Morning Star (3-bar bullish reversal). Needs downtrend."""
    if trend >= 0:
        return 0.0, 0.0
    body0 = _body(o0, c0)
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    star_range = _range(h1, l1)
    if star_range == 0:
        return 0.0, 0.0
    # Bar 0: large bearish, Bar 1: small star (< 30% range), Bar 2: large bullish, closes into bar 0
    if (
        _is_bearish(o0, c0)
        and body1 / star_range < 0.30
        and _is_bullish(o2, c2)
        and c2 > (o0 + c0) / 2.0  # closes above midpoint of bar 0
        and body2 >= body0 * 0.6
    ):
        conf = float(min(body2 / max(body0, 1e-8), 1.5) / 1.5) * 0.90
        return 1.0, conf
    return 0.0, 0.0


def _evening_star(
    o0: float, c0: float,
    o1: float, h1: float, l1: float, c1: float,
    o2: float, c2: float,
    trend: float,
) -> Tuple[float, float]:
    """Evening Star (3-bar bearish reversal). Needs uptrend."""
    if trend <= 0:
        return 0.0, 0.0
    body0 = _body(o0, c0)
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    star_range = _range(h1, l1)
    if star_range == 0:
        return 0.0, 0.0
    if (
        _is_bullish(o0, c0)
        and body1 / star_range < 0.30
        and _is_bearish(o2, c2)
        and c2 < (o0 + c0) / 2.0
        and body2 >= body0 * 0.6
    ):
        conf = float(min(body2 / max(body0, 1e-8), 1.5) / 1.5) * 0.90
        return -1.0, conf
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PatternSignal:
    """
    Synchronous candlestick pattern signal source.
    Called inline — no async, no I/O.
    """

    def __init__(self) -> None:
        try:
            from config import ApexConfig
            self._trend_window: int = int(
                getattr(ApexConfig, "PATTERN_TREND_WINDOW", 5)
            )
        except Exception:
            self._trend_window = 5

    def get_signal(self, symbol: str, data: pd.DataFrame) -> PatternResult:
        """
        Detect patterns on `data` (must have Open, High, Low, Close columns).
        Requires at least 3 bars.
        """
        if data is None or not isinstance(data, pd.DataFrame):
            return _neutral_pattern(symbol)

        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(data.columns):
            return _neutral_pattern(symbol)

        if len(data) < 3:
            return _neutral_pattern(symbol)

        try:
            return self._detect(symbol, data)
        except Exception as exc:
            logger.debug("PatternSignal error (%s): %s", symbol, exc)
            return _neutral_pattern(symbol)

    def _detect(self, symbol: str, data: pd.DataFrame) -> PatternResult:
        closes = data["Close"].values.astype(float)
        opens = data["Open"].values.astype(float)
        highs = data["High"].values.astype(float)
        lows = data["Low"].values.astype(float)

        trend = _trend_slope(closes, self._trend_window)

        # Last bar
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]

        found: List[Tuple[str, float, float]] = []  # (name, direction, confidence)

        # --- 1-bar patterns ---
        s, cf = _hammer(o, h, l, c, trend)
        if cf > 0:
            found.append(("hammer", s, cf))

        s, cf = _shooting_star(o, h, l, c, trend)
        if cf > 0:
            found.append(("shooting_star", s, cf))

        s, cf = _doji(o, h, l, c, trend)
        if cf > 0:
            found.append(("doji", s, cf))

        # --- 2-bar patterns (need at least 2 bars) ---
        if len(data) >= 2:
            o0, c0 = opens[-2], closes[-2]

            s, cf = _bullish_engulfing(o0, c0, o, c, trend)
            if cf > 0:
                found.append(("bullish_engulfing", s, cf))

            s, cf = _bearish_engulfing(o0, c0, o, c, trend)
            if cf > 0:
                found.append(("bearish_engulfing", s, cf))

        # --- 3-bar patterns (need at least 3 bars) ---
        if len(data) >= 3:
            o0_, c0_ = opens[-3], closes[-3]
            o1_, h1_, l1_, c1_ = opens[-2], highs[-2], lows[-2], closes[-2]

            s, cf = _morning_star(o0_, c0_, o1_, h1_, l1_, c1_, o, c, trend)
            if cf > 0:
                found.append(("morning_star", s, cf))

            s, cf = _evening_star(o0_, c0_, o1_, h1_, l1_, c1_, o, c, trend)
            if cf > 0:
                found.append(("evening_star", s, cf))

        if not found:
            return _neutral_pattern(symbol)

        # --- Aggregate ---
        # Weighted average by confidence; conflicting directions partially cancel
        total_conf = sum(cf for _, _, cf in found)
        if total_conf == 0:
            return _neutral_pattern(symbol)

        weighted_signal = sum(s * cf for _, s, cf in found) / total_conf
        # Final confidence = total_conf / len(found), capped at 1
        avg_conf = float(min(total_conf / len(found), 1.0))

        # Dominant pattern = highest confidence
        dominant_entry = max(found, key=lambda x: x[2])

        direction_votes = {name: float(s) for name, s, _ in found}

        return PatternResult(
            symbol=symbol,
            signal=float(np.clip(weighted_signal, -1.0, 1.0)),
            confidence=float(np.clip(avg_conf, 0.0, 1.0)),
            patterns_found=[name for name, _, _ in found],
            dominant=dominant_entry[0],
            direction_votes=direction_votes,
        )
