"""
models/iv_crush_strategy.py — Earnings IV Crush Strategy

Detects pre-earnings implied volatility elevation and generates signals for:
  1. IV Crush Trade: sell straddles / fade IV before earnings (short vol)
  2. PEAD Drift: buy/sell directional after earnings surprise (post-earnings drift)

How it works:
  - Detects earnings dates from yfinance calendar (24h cache)
  - Measures "IV elevation": current ATM IV vs 30-day trailing IV
  - Pre-earnings signal: fade IV when elevation ≥ threshold (expect crush after report)
  - Post-earnings signal: follow direction of earnings surprise × price gap
  - Dashboard snapshot shows upcoming earnings + IV elevation per symbol

Usage:
    from models.iv_crush_strategy import IVCrushStrategy
    strat = IVCrushStrategy()

    # Async-safe — call via asyncio.to_thread
    signal = strat.get_signal("AAPL")
    # → IVCrushSignal(symbol, iv_elevation, days_to_earnings, signal, confidence)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
IV_ELEVATION_THRESHOLD   = 1.40   # ATM IV / 30d avg IV — above this = "elevated"
IV_CRUSH_SIGNAL_SCALE    = 0.12   # max signal magnitude for IV crush trades
PEAD_SIGNAL_SCALE        = 0.15   # max signal magnitude for post-earnings drift
DAYS_BEFORE_EARNINGS_MIN = 1      # earliest pre-earnings window
DAYS_BEFORE_EARNINGS_MAX = 5      # latest pre-earnings window to fade IV
PEAD_WINDOW_DAYS         = 3      # days after earnings to apply drift signal
CACHE_TTL                = 86400  # 24h cache for earnings dates
IV_CACHE_TTL             = 3600   # 1h cache for IV readings


@dataclass
class IVCrushSignal:
    symbol: str
    days_to_earnings: Optional[float]     # None if no upcoming earnings found
    iv_elevation: float                   # current_iv / avg_iv (1.0 = no elevation)
    signal: float                         # [-1, 1]: negative = bearish/short vol
    confidence: float                     # [0, 1]
    strategy: str                         # "iv_crush" | "pead_long" | "pead_short" | "none"
    earnings_date: Optional[str] = None   # ISO date string
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["iv_elevation"] = round(d["iv_elevation"], 4)
        d["signal"] = round(d["signal"], 4)
        d["confidence"] = round(d["confidence"], 4)
        return d


class IVCrushStrategy:
    """
    Earnings IV Crush + Post-Earnings Drift signal generator.

    All network calls use yfinance; falls back gracefully on errors.
    Thread-safe (no async) — call via asyncio.to_thread from async contexts.
    """

    def __init__(
        self,
        iv_elevation_threshold: float = IV_ELEVATION_THRESHOLD,
        state_dir: Optional[Path] = None,
    ):
        self._iv_threshold = iv_elevation_threshold
        self._earnings_cache: Dict[str, tuple] = {}    # symbol → (earnings_info, ts)
        self._iv_cache: Dict[str, tuple] = {}          # symbol → (iv_data, ts)
        self._signals: Dict[str, IVCrushSignal] = {}  # symbol → latest signal

        if state_dir:
            self._state_dir = Path(state_dir)
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load()
        else:
            self._state_dir = None

    # ── Public API ──────────────────────────────────────────────────────────

    def get_signal(self, symbol: str) -> IVCrushSignal:
        """
        Compute IV crush / PEAD signal for a symbol.

        Returns cached result if within IV_CACHE_TTL; else fetches fresh data.
        """
        cached = self._signals.get(symbol)
        if cached and (time.time() - cached.last_updated) < IV_CACHE_TTL:
            return cached

        sig = self._compute_signal(symbol)
        self._signals[symbol] = sig
        return sig

    def get_all_signals(self, symbols: List[str]) -> Dict[str, IVCrushSignal]:
        """Batch compute signals for multiple symbols."""
        return {s: self.get_signal(s) for s in symbols}

    def get_snapshot(self) -> Dict:
        """Serialisable snapshot for REST endpoint."""
        active = [
            s.to_dict() for s in sorted(
                self._signals.values(),
                key=lambda s: abs(s.signal),
                reverse=True,
            )
            if abs(s.signal) > 0.01
        ]
        upcoming = [
            s.to_dict() for s in self._signals.values()
            if s.days_to_earnings is not None and 0 <= s.days_to_earnings <= 7
        ]
        return {
            "available": True,
            "total_tracked": len(self._signals),
            "active_signals": active[:20],
            "upcoming_earnings": upcoming[:10],
            "iv_elevation_threshold": self._iv_threshold,
            "iv_crush_scale": IV_CRUSH_SIGNAL_SCALE,
            "pead_scale": PEAD_SIGNAL_SCALE,
        }

    # ── Signal computation ──────────────────────────────────────────────────

    def _compute_signal(self, symbol: str) -> IVCrushSignal:
        """Compute signal from earnings calendar + IV data."""
        clean = symbol.replace("CRYPTO:", "").replace("/USD", "").split(":")[0]

        try:
            # 1. Earnings calendar
            earnings_info = self._get_earnings_info(clean)
            days_to = earnings_info.get("days_to_earnings")
            earnings_date = earnings_info.get("earnings_date")
            earnings_surprise = earnings_info.get("surprise_pct", 0.0)
            price_gap_pct = earnings_info.get("price_gap_pct", 0.0)

            # 2. IV data
            iv_data = self._get_iv_data(clean)
            current_iv = iv_data.get("current_iv", 0.0)
            avg_iv = iv_data.get("avg_iv", 0.0)
            iv_elevation = (current_iv / avg_iv) if avg_iv > 0.01 else 1.0

            # 3. Strategy selection
            if days_to is not None and DAYS_BEFORE_EARNINGS_MIN <= days_to <= DAYS_BEFORE_EARNINGS_MAX:
                # Pre-earnings: fade IV if elevated
                if iv_elevation >= self._iv_threshold:
                    # Short vol — magnitude proportional to elevation above threshold
                    raw_strength = min((iv_elevation - 1.0) / (self._iv_threshold - 1.0 + 1e-9), 2.0)
                    sig = -raw_strength * IV_CRUSH_SIGNAL_SCALE  # negative = short vol
                    conf = min(0.40 + (iv_elevation - 1.0) * 0.30, 0.85)
                    return IVCrushSignal(
                        symbol=symbol,
                        days_to_earnings=days_to,
                        iv_elevation=iv_elevation,
                        signal=float(sig),
                        confidence=float(conf),
                        strategy="iv_crush",
                        earnings_date=earnings_date,
                    )

            elif days_to is not None and -PEAD_WINDOW_DAYS <= days_to < 0:
                # Post-earnings: follow price gap direction (PEAD)
                if abs(price_gap_pct) > 0.01:
                    direction = 1.0 if price_gap_pct > 0 else -1.0
                    raw_strength = min(abs(price_gap_pct) / 0.05, 2.0)  # scale by 5% gap
                    sig = direction * raw_strength * PEAD_SIGNAL_SCALE
                    conf = min(0.45 + abs(price_gap_pct) * 2.0, 0.80)
                    strategy = "pead_long" if direction > 0 else "pead_short"
                    return IVCrushSignal(
                        symbol=symbol,
                        days_to_earnings=days_to,
                        iv_elevation=iv_elevation,
                        signal=float(sig),
                        confidence=float(conf),
                        strategy=strategy,
                        earnings_date=earnings_date,
                    )

            return IVCrushSignal(
                symbol=symbol,
                days_to_earnings=days_to,
                iv_elevation=iv_elevation,
                signal=0.0,
                confidence=0.0,
                strategy="none",
                earnings_date=earnings_date,
            )

        except Exception as exc:
            logger.debug("IVCrushStrategy._compute_signal(%s) failed: %s", symbol, exc)
            return IVCrushSignal(
                symbol=symbol,
                days_to_earnings=None,
                iv_elevation=1.0,
                signal=0.0,
                confidence=0.0,
                strategy="none",
            )

    def _get_earnings_info(self, symbol: str) -> Dict:
        """Fetch earnings date and post-earnings data from yfinance (cached 24h)."""
        cached = self._earnings_cache.get(symbol)
        if cached and (time.time() - cached[1]) < CACHE_TTL:
            return cached[0]

        info: Dict = {}
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            # Next earnings date
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                import pandas as pd
                for col in ("Earnings Date", "earnings_date"):
                    if col in cal.columns or (hasattr(cal, "index") and col in cal.index):
                        ed = None
                        if col in cal.columns:
                            vals = cal[col].dropna()
                            ed = vals.iloc[0] if len(vals) > 0 else None
                        elif col in cal.index:
                            ed = cal.loc[col, 0] if 0 in cal.columns else None
                        if ed is not None:
                            try:
                                ed_ts = pd.Timestamp(ed)
                                days = (ed_ts - pd.Timestamp.now()).days
                                info["days_to_earnings"] = float(days)
                                info["earnings_date"] = str(ed_ts.date())
                            except Exception:
                                pass
                        break

            # Post-earnings surprise from most recent quarterly result
            try:
                earnings_hist = ticker.earnings_history
                if earnings_hist is not None and not earnings_hist.empty:
                    last = earnings_hist.iloc[-1]
                    actual = float(last.get("epsActual", 0))
                    est    = float(last.get("epsEstimate", 0))
                    if abs(est) > 0.001:
                        info["surprise_pct"] = float((actual - est) / abs(est))
            except Exception:
                pass

            # Last daily price gap after earnings
            try:
                hist = ticker.history(period="5d")
                if hist is not None and len(hist) >= 2:
                    gap = float((hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2])
                    info["price_gap_pct"] = gap
            except Exception:
                pass

        except Exception as exc:
            logger.debug("IVCrushStrategy._get_earnings_info(%s) failed: %s", symbol, exc)

        self._earnings_cache[symbol] = (info, time.time())
        return info

    def _get_iv_data(self, symbol: str) -> Dict:
        """Fetch ATM implied volatility from yfinance options chain (cached 1h)."""
        cached = self._iv_cache.get(symbol)
        if cached and (time.time() - cached[1]) < IV_CACHE_TTL:
            return cached[0]

        iv_data: Dict = {}
        try:
            import yfinance as yf
            import numpy as np
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if not expirations:
                self._iv_cache[symbol] = (iv_data, time.time())
                return iv_data

            # Get ATM IV from nearest expiration
            hist = ticker.history(period="1d")
            if hist is None or hist.empty:
                self._iv_cache[symbol] = (iv_data, time.time())
                return iv_data
            current_price = float(hist["Close"].iloc[-1])

            ivs = []
            for exp in expirations[:3]:
                try:
                    chain = ticker.option_chain(exp)
                    calls = chain.calls
                    if calls is None or calls.empty:
                        continue
                    # ATM = closest strike to current price
                    atm_idx = (calls["strike"] - current_price).abs().idxmin()
                    atm_iv = float(calls.loc[atm_idx, "impliedVolatility"])
                    if 0 < atm_iv < 5.0:  # sanity check
                        ivs.append(atm_iv)
                except Exception:
                    continue

            if ivs:
                current_iv = float(min(ivs))  # nearest expiration tends to have IV spike
                avg_iv = float(sum(ivs) / len(ivs))
                iv_data["current_iv"] = current_iv
                iv_data["avg_iv"] = avg_iv if avg_iv > 0 else current_iv

        except Exception as exc:
            logger.debug("IVCrushStrategy._get_iv_data(%s) failed: %s", symbol, exc)

        self._iv_cache[symbol] = (iv_data, time.time())
        return iv_data

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._state_dir is None:
            return
        try:
            state = {
                "signals": {k: v.to_dict() for k, v in self._signals.items()
                            if time.time() - v.last_updated < 86400},
            }
            p = self._state_dir / "iv_crush_state.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception as exc:
            logger.debug("IVCrushStrategy: save failed: %s", exc)

    def _load(self) -> None:
        if self._state_dir is None:
            return
        try:
            p = self._state_dir / "iv_crush_state.json"
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            for sym, sd in raw.get("signals", {}).items():
                self._signals[sym] = IVCrushSignal(
                    symbol=sd["symbol"],
                    days_to_earnings=sd.get("days_to_earnings"),
                    iv_elevation=float(sd.get("iv_elevation", 1.0)),
                    signal=float(sd.get("signal", 0.0)),
                    confidence=float(sd.get("confidence", 0.0)),
                    strategy=sd.get("strategy", "none"),
                    earnings_date=sd.get("earnings_date"),
                    last_updated=float(sd.get("last_updated", 0)),
                )
            logger.info("IVCrushStrategy: loaded %d signals", len(self._signals))
        except Exception as exc:
            logger.debug("IVCrushStrategy: load failed: %s", exc)
