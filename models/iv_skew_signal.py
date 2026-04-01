"""
models/iv_skew_signal.py — Implied Volatility Skew Signal

Two complementary IV-based signals:

  1. Options Skew  — Put/call implied volatility spread derived from the
                     Yahoo Finance options chain (front-month ATM strikes).
                     High put IV relative to call IV = bearish hedging demand.
                     Signal: -1 (heavy put skew) to +1 (heavy call skew).

  2. VIX Term Structure — Ratio of VIX3M (3-month forward vol) to VIX (spot vol).
                     Ratio > 1 = "normal" contango → market calm, risk-on.
                     Ratio < 1 = "backwardation" → near-term fear, risk-off.
                     Signal: +1 (deep contango) to -1 (deep backwardation).

Combined signal = weighted average, cached 10 minutes (options data is slow).

Usage:
    gen = IVSkewSignal()
    signal = gen.get_signal("AAPL")   # per-symbol put/call skew
    macro = gen.get_vix_term_signal()  # VIX term structure (macro, symbol-agnostic)

Config keys:
    IV_SKEW_ENABLED             = True
    IV_SKEW_PUT_CALL_WEIGHT     = 0.60
    IV_SKEW_VIX_TERM_WEIGHT     = 0.40
    IV_SKEW_CACHE_TTL_SECONDS   = 600   # 10-min cache (options chain fetch)
    IV_SKEW_BLEND_WEIGHT        = 0.05  # blend weight in GodLevel stack
    IV_SKEW_STRIKES_AROUND_ATM  = 3     # number of strikes above/below ATM to average
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "IV_SKEW_ENABLED":            True,
    "IV_SKEW_PUT_CALL_WEIGHT":    0.60,
    "IV_SKEW_VIX_TERM_WEIGHT":    0.40,
    "IV_SKEW_CACHE_TTL_SECONDS":  600,
    "IV_SKEW_BLEND_WEIGHT":       0.05,
    "IV_SKEW_STRIKES_AROUND_ATM": 3,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


# ── Sub-signal helpers ─────────────────────────────────────────────────────────

def compute_put_call_skew(put_ivs: list, call_ivs: list) -> float:
    """
    Compute put/call IV skew signal.

    Args:
        put_ivs:  list of put implied volatilities (ATM ± N strikes)
        call_ivs: list of call implied volatilities (ATM ± N strikes)

    Returns:
        float in [-1, +1]:
            +1 = calls much more expensive than puts (bullish skew)
            -1 = puts much more expensive than calls (bearish skew)
    """
    if not put_ivs or not call_ivs:
        return 0.0

    valid_puts = [iv for iv in put_ivs if iv > 0]
    valid_calls = [iv for iv in call_ivs if iv > 0]

    if not valid_puts or not valid_calls:
        return 0.0

    avg_put = float(np.mean(valid_puts))
    avg_call = float(np.mean(valid_calls))

    if avg_put + avg_call < 1e-8:
        return 0.0

    # Skew ratio: (call_iv - put_iv) / (call_iv + put_iv)
    # Positive = calls more expensive = bullish, Negative = puts more expensive = bearish
    raw_skew = (avg_call - avg_put) / (avg_call + avg_put)
    return float(np.clip(raw_skew * 5.0, -1.0, 1.0))  # scale and clip


def compute_vix_term_signal(vix_spot: float, vix_3m: float) -> float:
    """
    VIX term structure signal.

    VIX3M / VIX > 1.15  → deep contango → risk-on → +1
    VIX3M / VIX < 0.90  → backwardation → risk-off → -1
    Near 1.0              → neutral → 0

    Returns float in [-1, +1].
    """
    if vix_spot <= 0 or vix_3m <= 0:
        return 0.0

    ratio = vix_3m / vix_spot
    # Map ratio: 0.85 → -1.0, 1.0 → 0.0, 1.15 → +1.0
    signal = (ratio - 1.0) / 0.15
    return float(np.clip(signal, -1.0, 1.0))


# ── IVSkewSignal ──────────────────────────────────────────────────────────────

@dataclass
class IVSkewState:
    """Snapshot of IV skew sub-signals for a symbol."""
    symbol: str
    put_call_skew: float = 0.0
    vix_term_signal: float = 0.0
    combined_signal: float = 0.0
    avg_put_iv: float = 0.0
    avg_call_iv: float = 0.0
    vix_spot: float = 0.0
    vix_3m: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "put_call_skew": round(self.put_call_skew, 4),
            "vix_term_signal": round(self.vix_term_signal, 4),
            "combined_signal": round(self.combined_signal, 4),
            "avg_put_iv": round(self.avg_put_iv, 4),
            "avg_call_iv": round(self.avg_call_iv, 4),
            "vix_spot": round(self.vix_spot, 2),
            "vix_3m": round(self.vix_3m, 2),
            "timestamp": self.timestamp,
            "error": self.error,
        }


class IVSkewSignal:
    """
    Implied Volatility Skew signal generator.

    Per-symbol: fetches options chain from yfinance, computes put/call IV skew.
    Macro: fetches VIX + VVIX/VIX3M for term structure signal.
    Both cached independently (TTL configurable).
    """

    def __init__(self):
        # symbol → (IVSkewState, fetch_ts)
        self._symbol_cache: Dict[str, Tuple[IVSkewState, float]] = {}
        # VIX term state
        self._vix_term_state: Optional[Tuple[float, float, float]] = None  # (vix, vix3m, ts)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_signal(self, symbol: str) -> float:
        """Return combined IV skew signal for a symbol."""
        if not _cfg("IV_SKEW_ENABLED"):
            return 0.0
        state = self._get_symbol_state(symbol)
        return state.combined_signal

    def get_vix_term_signal(self) -> float:
        """Return the VIX term structure signal (macro, symbol-agnostic)."""
        if not _cfg("IV_SKEW_ENABLED"):
            return 0.0
        vix, vix3m = self._get_vix_levels()
        return compute_vix_term_signal(vix, vix3m)

    def get_state(self, symbol: str) -> IVSkewState:
        """Return full IVSkewState for a symbol."""
        return self._get_symbol_state(symbol)

    def get_report(self, symbol: str) -> Dict:
        """Return JSON-serialisable report for a symbol."""
        return self._get_symbol_state(symbol).to_dict()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_symbol_state(self, symbol: str) -> IVSkewState:
        ttl = float(_cfg("IV_SKEW_CACHE_TTL_SECONDS"))
        now = time.monotonic()
        cached = self._symbol_cache.get(symbol)
        if cached is not None and (now - cached[1]) < ttl:
            return cached[0]

        state = self._compute_symbol(symbol)
        self._symbol_cache[symbol] = (state, now)
        return state

    def _compute_symbol(self, symbol: str) -> IVSkewState:
        try:
            return self._compute_symbol_inner(symbol)
        except Exception as e:
            logger.debug("IVSkew: compute failed for %s — %s", symbol, e)
            return IVSkewState(symbol=symbol, error=str(e))

    def _compute_symbol_inner(self, symbol: str) -> IVSkewState:
        # Skip crypto and FX (no options chain)
        clean = symbol.split(":")[-1].split("/")[0].upper()
        if "CRYPTO:" in symbol.upper() or "/" in symbol:
            return IVSkewState(symbol=symbol)

        n_strikes = int(_cfg("IV_SKEW_STRIKES_AROUND_ATM"))
        put_ivs, call_ivs, spot_price = self._fetch_options_ivs(clean, n_strikes)

        w_pc = float(_cfg("IV_SKEW_PUT_CALL_WEIGHT"))
        w_vt = float(_cfg("IV_SKEW_VIX_TERM_WEIGHT"))

        pc_skew = compute_put_call_skew(put_ivs, call_ivs)
        vix_spot, vix_3m = self._get_vix_levels()
        vt_sig = compute_vix_term_signal(vix_spot, vix_3m)

        w_total = w_pc + w_vt
        if w_total < 1e-10:
            combined = 0.0
        else:
            combined = (w_pc * pc_skew + w_vt * vt_sig) / w_total
        combined = float(np.clip(combined, -1.0, 1.0))

        avg_put = float(np.mean(put_ivs)) if put_ivs else 0.0
        avg_call = float(np.mean(call_ivs)) if call_ivs else 0.0

        logger.debug(
            "IVSkew %s: put_call=%.3f vix_term=%.3f → combined=%.3f",
            clean, pc_skew, vt_sig, combined,
        )

        return IVSkewState(
            symbol=symbol,
            put_call_skew=pc_skew,
            vix_term_signal=vt_sig,
            combined_signal=combined,
            avg_put_iv=avg_put,
            avg_call_iv=avg_call,
            vix_spot=vix_spot,
            vix_3m=vix_3m,
        )

    def _fetch_options_ivs(
        self, symbol: str, n_strikes: int
    ) -> Tuple[list, list, float]:
        """
        Fetch front-month ATM put and call IVs from yfinance.
        Returns (put_ivs, call_ivs, spot_price).
        """
        try:
            import yfinance as yf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker = yf.Ticker(symbol)
                spot_data = ticker.history(period="1d", progress=False)
                if spot_data.empty:
                    return [], [], 0.0
                spot_price = float(spot_data["Close"].iloc[-1])

                expirations = ticker.options
                if not expirations:
                    return [], [], spot_price

                # Use nearest expiry (front month)
                exp = expirations[0]
                chain = ticker.option_chain(exp)
                calls = chain.calls
                puts = chain.puts

                if calls.empty or puts.empty:
                    return [], [], spot_price

            # Find ATM strikes
            call_strikes = sorted(calls["strike"].tolist(), key=lambda x: abs(x - spot_price))
            put_strikes = sorted(puts["strike"].tolist(), key=lambda x: abs(x - spot_price))

            call_atm = call_strikes[:n_strikes]
            put_atm = put_strikes[:n_strikes]

            call_ivs = []
            for k in call_atm:
                row = calls[calls["strike"] == k]
                if not row.empty:
                    iv = float(row["impliedVolatility"].iloc[0])
                    if iv > 0:
                        call_ivs.append(iv)

            put_ivs = []
            for k in put_atm:
                row = puts[puts["strike"] == k]
                if not row.empty:
                    iv = float(row["impliedVolatility"].iloc[0])
                    if iv > 0:
                        put_ivs.append(iv)

            return put_ivs, call_ivs, spot_price

        except Exception as e:
            logger.debug("IVSkew: options fetch failed for %s — %s", symbol, e)
            return [], [], 0.0

    def _get_vix_levels(self) -> Tuple[float, float]:
        """
        Returns (VIX spot, VIX3M). Cached with same TTL as symbol cache.
        Uses ^VIX and ^VIX3M from Yahoo Finance.
        """
        ttl = float(_cfg("IV_SKEW_CACHE_TTL_SECONDS"))
        now = time.monotonic()
        if self._vix_term_state is not None and (now - self._vix_term_state[2]) < ttl:
            return self._vix_term_state[0], self._vix_term_state[1]

        try:
            import yfinance as yf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vix_data = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
                vix3m_data = yf.download("^VIX3M", period="5d", progress=False, auto_adjust=True)

            vix = float(vix_data["Close"].iloc[-1]) if not vix_data.empty else 0.0
            vix3m = float(vix3m_data["Close"].iloc[-1]) if not vix3m_data.empty else 0.0
            self._vix_term_state = (vix, vix3m, now)
            return vix, vix3m
        except Exception as e:
            logger.debug("IVSkew: VIX fetch failed — %s", e)
            return 0.0, 0.0


# ── Module-level singleton ────────────────────────────────────────────────────

_iv_skew_signal: Optional[IVSkewSignal] = None


def get_iv_skew_signal() -> IVSkewSignal:
    global _iv_skew_signal
    if _iv_skew_signal is None:
        _iv_skew_signal = IVSkewSignal()
    return _iv_skew_signal
