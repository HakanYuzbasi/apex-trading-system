"""
quant_system/strategies/directional_equity.py

DirectionalEquityStrategy — single-name long/short with ML overlay.

Signal pipeline (all computed on 5-min aggregated bars):
  1. RSI(14) + EMA(20) on 5-min closes         — primary entry filter
  2. GodLevelSignalGenerator ML overlay          — veto/boost based on ML confidence
  3. MTFSignalFuser (5m + 1h + daily proxy)     — multi-timeframe alignment gate
  4. [crypto only] Binance funding-rate scalar   — directional bias for perps

Entry sizing:
  base_notional × ml_scalar × mtf_scalar
  ml_scalar  = 1.25 if ML agrees strongly, 0.75 if weak, veto if disagrees
  mtf_scalar = 1.10 if all 3 TFs aligned, 0.85 if mixed, veto if opposed
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, date
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, TradeTick
from quant_system.strategies.base import BaseStrategy

_ET = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)

# ── Net-exposure throttle (shared across all instances) ───────────────────────

_NET_LONG_CAP  = 20_000.0   # max aggregate long  notional across all dir slots
_NET_SHORT_CAP = 20_000.0   # max aggregate short notional

class _NetExposureTracker:
    """Module-level singleton that tracks live net long/short notional."""

    def __init__(self) -> None:
        self._long: dict[str, float]  = {}   # instrument → notional
        self._short: dict[str, float] = {}

    def register(self, instrument: str, notional: float, side: str) -> None:
        self.deregister(instrument)
        if side == "long":
            self._long[instrument] = notional
        else:
            self._short[instrument] = notional

    def deregister(self, instrument: str) -> None:
        self._long.pop(instrument, None)
        self._short.pop(instrument, None)

    def can_open(self, notional: float, side: str) -> bool:
        if side == "long":
            return sum(self._long.values()) + notional <= _NET_LONG_CAP
        return sum(self._short.values()) + notional <= _NET_SHORT_CAP


_NET_EXPOSURE = _NetExposureTracker()

# ── Earnings blackout cache ───────────────────────────────────────────────────

_EARNINGS_CACHE: dict[str, tuple[date | None, float]] = {}   # sym → (next_date, ts)
_EARNINGS_TTL   = 86_400.0   # refresh once per day
# ETF/index tickers that don't have individual earnings — skip blackout check
_ETF_PREFIXES   = {"SPY", "QQQ", "GLD", "SLV", "TLT", "IEF", "USO", "XLE", "XLF", "XLK"}
_BLACKOUT_DAYS  = 2           # block entries within ±2 trading days of earnings


def _next_earnings_date(symbol: str) -> date | None:
    """Fetch next scheduled earnings date via yfinance (cached 24h)."""
    cached = _EARNINGS_CACHE.get(symbol)
    if cached is not None and time.monotonic() - cached[1] < _EARNINGS_TTL:
        return cached[0]
    result: date | None = None
    try:
        import yfinance as _yf
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            cal = _yf.Ticker(symbol).calendar
        if cal is not None:
            ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if ed is None and hasattr(cal, "loc"):
                # calendar is a DataFrame — check "Earnings Date" row
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"]
            if ed is not None:
                # May be a list/Series of one or two timestamps
                val = ed[0] if hasattr(ed, "__getitem__") else ed
                if hasattr(val, "date"):
                    result = val.date()
                elif isinstance(val, date):
                    result = val
    except Exception:
        pass
    _EARNINGS_CACHE[symbol] = (result, time.monotonic())
    return result


def _earnings_blackout(symbol: str) -> bool:
    """True if earnings are within ±_BLACKOUT_DAYS trading days."""
    if symbol in _ETF_PREFIXES or symbol.startswith("CRYPTO:") or symbol.startswith("FOREX:"):
        return False
    nxt = _next_earnings_date(symbol)
    if nxt is None:
        return False
    today = date.today()
    delta = (nxt - today).days
    return -_BLACKOUT_DAYS <= delta <= _BLACKOUT_DAYS

# ── Technical indicators (no pandas) ─────────────────────────────────────────

def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0.0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def _ema(closes: list[float], period: int) -> float:
    if not closes:
        return float("nan")
    k = 2.0 / (period + 1)
    val = closes[0]
    for p in closes[1:]:
        val = p * k + val * (1.0 - k)
    return val


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    trs = [
        max(highs[i] - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    return float(np.mean(trs[-period:])) if trs else 0.0


# ── Strategy ──────────────────────────────────────────────────────────────────

class DirectionalEquityStrategy(BaseStrategy):
    """
    Single-name directional strategy with 5-min bars, GodLevel ML overlay,
    and multi-timeframe confirmation.
    """

    RSI_PERIOD    = 14
    EMA_PERIOD    = 20
    ATR_PERIOD    = 14
    RSI_OVERSOLD  = 40.0
    RSI_OVERBOUGHT = 60.0
    RSI_EXIT      = 50.0
    STOP_ATR_MULT = 1.5
    TP_ATR_MULT   = 3.0
    MAX_5M_BARS   = 200   # ~16 h of 5-min bars
    MAX_1H_BARS   = 50

    # Trailing stop: activates after TRAIL_ACTIVATE_MULT×ATR in favor,
    # then trails at TRAIL_ATR_MULT×ATR behind the high-water mark.
    TRAIL_ACTIVATE_MULT = 1.5
    TRAIL_ATR_MULT      = 1.0

    # Staged exits: close 50% at TP1, trail the remaining 50%.
    # TP1 is intentionally the same level as TRAIL_ACTIVATE_MULT so both
    # actions fire together: half is banked, the other half starts trailing.
    TP1_ATR_MULT = 1.5

    # GodLevel: require at least 30 × 5-min bars (~2.5 h) before using ML signal
    _GL_MIN_BARS  = 30
    # Funding rate refresh interval (seconds) — crypto only
    _FR_TTL       = 900
    # Daily circuit breaker: stop new entries once realized loss exceeds this threshold.
    # Approximate: uses entry notional × P&L % (ignores partial closes, conservative side).
    _DAILY_LOSS_LIMIT = -300.0   # USD
    # Afternoon trail tighten: after 14:30 ET, trail at this fraction of TRAIL_ATR_MULT.
    _AFTERNOON_TRAIL_TIGHTEN = 0.5

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        instrument: str,
        leg_notional: float = 2_000.0,
        warmup_bars: int = 14,
    ) -> None:
        super().__init__(event_bus)
        self.instrument   = instrument
        self.leg_notional = float(leg_notional)
        self.warmup_bars  = int(warmup_bars)     # 5-min bars needed before trading

        # ── 1-min → 5-min aggregator ─────────────────────────────────────────
        self._1m_count = 0          # 1-min bars accumulated in current 5-min window
        self._5m_acc: dict[str, float] | None = None  # accumulating 5-min bar

        # 5-min OHLCV history
        self._5m_closes:  deque[float] = deque(maxlen=self.MAX_5M_BARS)
        self._5m_highs:   deque[float] = deque(maxlen=self.MAX_5M_BARS)
        self._5m_lows:    deque[float] = deque(maxlen=self.MAX_5M_BARS)
        self._5m_volumes: deque[float] = deque(maxlen=self.MAX_5M_BARS)
        self._5m_bar_count = 0

        # 1-hour OHLCV history (aggregated from 5-min, 12 bars = 1 h)
        self._1h_acc: dict[str, float] | None = None
        self._5m_in_hour = 0
        self._1h_closes:  deque[float] = deque(maxlen=self.MAX_1H_BARS)
        self._1h_highs:   deque[float] = deque(maxlen=self.MAX_1H_BARS)
        self._1h_lows:    deque[float] = deque(maxlen=self.MAX_1H_BARS)
        self._1h_volumes: deque[float] = deque(maxlen=self.MAX_1H_BARS)

        # ── ML signal overlay (lazy-init) ─────────────────────────────────────
        self._godlevel: Any | None = None
        self._godlevel_init_failed = False
        self._gl_signal: float  = 0.0    # last GodLevel signal
        self._gl_conf:   float  = 0.0    # last GodLevel confidence
        self._gl_ts:     float  = 0.0    # monotonic time of last GL call

        # ── MTF fuser (lazy-init) ─────────────────────────────────────────────
        self._mtf_fuser: Any | None = None
        self._mtf_aligned: bool = True   # default: allow entries
        self._mtf_conf_adj: float = 1.0

        # ── Crypto funding-rate cache ─────────────────────────────────────────
        self._is_crypto = instrument.startswith("CRYPTO:")
        self._funding_rate: float = 0.0
        self._funding_ts:   float = 0.0

        # ── State machine ─────────────────────────────────────────────────────
        self._state:       str   = "flat"
        self._entry_price: float = 0.0
        self._stop_price:  float = 0.0
        self._tp_price:    float = 0.0

        # Fix 4: post-stop-loss cooldown (15 min = 900 s)
        self._cooldown_until: float = 0.0

        # Trailing stop state (reset on each new entry)
        self._atr_at_entry:    float = 0.0
        self._trail_hwm:       float = 0.0   # high-water-mark for long / low-water for short
        self._trailing_active: bool  = False

        # Staged exits: track if partial TP1 has fired and original notional
        self._half_closed:   bool  = False
        self._open_notional: float = 0.0

        # Session VWAP (equity only, resets each calendar day ET)
        self._vwap_cum_pv:       float      = 0.0
        self._vwap_cum_v:        float      = 0.0
        self._vwap_session_date: date | None = None

        # Per-strategy daily circuit breaker (improvement #2)
        # Stop new entries for the day once cumulative realized loss exceeds limit.
        self._daily_realized_pnl:   float      = 0.0
        self._daily_pnl_date:       date | None = None
        self._daily_circuit_fired:  bool        = False

        self.last_z_score = 0.0

        logger.info(
            "DirectionalEquity: watching %s | notional=$%.0f | warmup=%d 5m-bars",
            self.instrument, self.leg_notional, self.warmup_bars,
        )

    # ── Bar handler ───────────────────────────────────────────────────────────

    def on_bar(self, event: BarEvent) -> None:
        if event.instrument_id != self.instrument:
            return

        close  = float(event.close_price)
        high   = float(event.high_price)
        low    = float(event.low_price)
        volume = float(event.volume)
        open_  = float(event.open_price)

        # ── Accumulate into 5-min bar ─────────────────────────────────────────
        if self._5m_acc is None:
            self._5m_acc = {"o": open_, "h": high, "l": low, "c": close, "v": volume}
        else:
            self._5m_acc["h"] = max(self._5m_acc["h"], high)
            self._5m_acc["l"] = min(self._5m_acc["l"], low)
            self._5m_acc["c"] = close
            self._5m_acc["v"] += volume
        self._1m_count += 1

        if self._1m_count < 5:
            return  # 5-min bar not yet complete

        # ── Emit 5-min bar ────────────────────────────────────────────────────
        b = self._5m_acc
        self._5m_closes.append(b["c"])
        self._5m_highs.append(b["h"])
        self._5m_lows.append(b["l"])
        self._5m_volumes.append(b["v"])
        self._5m_bar_count += 1

        # Roll into 1h accumulator
        self._accumulate_1h(b)

        # Session VWAP accumulation (equity only, reset each ET calendar day)
        if not self._is_crypto and b["v"] > 0:
            today = datetime.now(_ET).date()
            if self._vwap_session_date != today:
                self._vwap_cum_pv = 0.0
                self._vwap_cum_v  = 0.0
                self._vwap_session_date = today
            self._vwap_cum_pv += b["v"] * b["c"]
            self._vwap_cum_v  += b["v"]

        # Reset 5-min accumulator
        self._5m_acc = None
        self._1m_count = 0

        if self._5m_bar_count < self.warmup_bars:
            return

        closes = list(self._5m_closes)
        highs  = list(self._5m_highs)
        lows   = list(self._5m_lows)

        rsi   = _rsi(closes, self.RSI_PERIOD)
        ema20 = _ema(closes, self.EMA_PERIOD)
        atr   = _atr(highs, lows, closes, self.ATR_PERIOD)

        self.last_z_score = (rsi - 50.0) / 15.0

        # Refresh ML and MTF overlays (non-blocking — cached)
        self._refresh_godlevel(closes)
        self._refresh_mtf()

        if self._state == "flat":
            self._try_entry(b["c"], rsi, ema20, atr)
        else:
            self._manage_position(b["c"], rsi)

    def on_tick(self, event: TradeTick) -> None:
        pass

    # ── 1-hour accumulator ────────────────────────────────────────────────────

    def _accumulate_1h(self, b: dict) -> None:
        if self._1h_acc is None:
            self._1h_acc = {"o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]}
        else:
            self._1h_acc["h"] = max(self._1h_acc["h"], b["h"])
            self._1h_acc["l"] = min(self._1h_acc["l"], b["l"])
            self._1h_acc["c"] = b["c"]
            self._1h_acc["v"] += b["v"]
        self._5m_in_hour += 1

        if self._5m_in_hour >= 12:   # 12 × 5 min = 60 min
            h = self._1h_acc
            self._1h_closes.append(h["c"])
            self._1h_highs.append(h["h"])
            self._1h_lows.append(h["l"])
            self._1h_volumes.append(h["v"])
            self._1h_acc = None
            self._5m_in_hour = 0

    # ── Session VWAP + 1h trend direction ────────────────────────────────────

    def _session_vwap(self) -> float | None:
        """Session VWAP from 5-min bars since ET midnight. Equity only."""
        if self._is_crypto or self._vwap_cum_v < 1e-6:
            return None
        return self._vwap_cum_pv / self._vwap_cum_v

    def _1h_trend_direction(self) -> str:
        """
        1h EMA(5) vs EMA(20) crossover → 'up', 'down', or 'neutral'.
        Requires ≥22 hourly bars; otherwise returns 'neutral' (no filter).
        A 0.2% buffer prevents whipsaw around the crossover point.
        """
        closes = list(self._1h_closes)
        if len(closes) < 22:
            return "neutral"
        ema5  = _ema(closes[-20:], 5)
        ema20 = _ema(closes, 20)
        if ema5 > ema20 * 1.002:
            return "up"
        if ema5 < ema20 * 0.998:
            return "down"
        return "neutral"

    # ── GodLevel ML overlay ───────────────────────────────────────────────────

    def _refresh_godlevel(self, closes: list[float]) -> None:
        """Update GodLevel signal at most once per 5-min bar (TTL = 5 min)."""
        now = time.monotonic()
        if now - self._gl_ts < 290.0:   # refresh every ~5 min
            return
        if len(closes) < self._GL_MIN_BARS:
            return
        if self._godlevel_init_failed:
            return

        try:
            if self._godlevel is None:
                import pandas as _pd
                from models.god_level_signal_generator import GodLevelSignalGenerator
                self._godlevel = GodLevelSignalGenerator()

            import pandas as _pd
            prices_series = _pd.Series(closes[-200:], dtype=float)
            symbol = self.instrument.replace("CRYPTO:", "").replace("/USD", "")
            result = self._godlevel.generate_ml_signal(symbol, prices_series)
            self._gl_signal = float(result.get("signal", 0.0))
            self._gl_conf   = float(result.get("confidence", 0.0))
            self._gl_ts     = now
            logger.debug(
                "GodLevel [%s] signal=%.3f conf=%.2f",
                self.instrument, self._gl_signal, self._gl_conf,
            )
        except Exception as exc:
            logger.debug("GodLevel init/call failed for %s: %s — disabling", self.instrument, exc)
            self._godlevel_init_failed = True

    def _godlevel_ok(self) -> bool:
        """True when GodLevel has a fresh signal."""
        return (not self._godlevel_init_failed
                and self._gl_ts > 0.0
                and time.monotonic() - self._gl_ts < 900.0)

    # ── MTF fuser ─────────────────────────────────────────────────────────────

    def _refresh_mtf(self) -> None:
        """Fuse 5-min + 1-h + daily-proxy signals."""
        try:
            if self._mtf_fuser is None:
                from models.mtf_signal_fusion import MTFSignalFuser
                self._mtf_fuser = MTFSignalFuser()

            import pandas as _pd

            # 5-min DataFrame
            fivemin_df = None
            if len(self._5m_closes) >= 6:
                fivemin_df = _pd.DataFrame({
                    "Open":   list(self._5m_closes),  # close proxy
                    "High":   list(self._5m_highs),
                    "Low":    list(self._5m_lows),
                    "Close":  list(self._5m_closes),
                    "Volume": list(self._5m_volumes),
                })

            # 1-hour DataFrame
            hourly_df = None
            if len(self._1h_closes) >= 5:
                hourly_df = _pd.DataFrame({
                    "Open":   list(self._1h_closes),
                    "High":   list(self._1h_highs),
                    "Low":    list(self._1h_lows),
                    "Close":  list(self._1h_closes),
                    "Volume": list(self._1h_volumes),
                })

            # Daily proxy: use GodLevel signal if available, else 0.0
            daily_signal = self._gl_signal if self._godlevel_ok() else 0.0

            # Get regime from global router
            regime = "neutral"
            try:
                from risk.regime_router import get_global_regime_router
                rr = get_global_regime_router()
                if rr is not None:
                    regime = getattr(rr.last_regime, "value", "neutral")
            except Exception:
                pass

            fused = self._mtf_fuser.fuse(
                daily_signal=daily_signal,
                hourly_df=hourly_df,
                fivemin_df=fivemin_df,
                regime=regime,
            )
            self._mtf_aligned   = fused.aligned
            self._mtf_conf_adj  = fused.confidence_adj
        except Exception as exc:
            logger.debug("MTF refresh failed for %s: %s", self.instrument, exc)

    # ── Crypto funding rate ───────────────────────────────────────────────────

    def _refresh_funding_rate(self) -> None:
        """Fetch Binance funding rate (crypto only, TTL 15 min)."""
        if not self._is_crypto:
            return
        if time.monotonic() - self._funding_ts < self._FR_TTL:
            return
        try:
            import urllib.request as _ur, json as _json
            raw = self.instrument.replace("CRYPTO:", "").replace("/", "")
            symbol_binance = raw.replace("USD", "USDT")
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol_binance}&limit=1"
            with _ur.urlopen(url, timeout=3) as resp:
                data = _json.loads(resp.read())
                if data:
                    self._funding_rate = float(data[-1].get("fundingRate", 0.0))
                    self._funding_ts   = time.monotonic()
                    logger.debug("Funding rate [%s] = %.6f", self.instrument, self._funding_rate)
        except Exception:
            self._funding_ts = time.monotonic()   # suppress retry for TTL

    def _funding_rate_scalar(self) -> float:
        """
        Funding rate → directional bias scalar [-0.3, +0.3].
        Extreme positive funding (longs pay shorts) → short pressure.
        Extreme negative → long pressure.
        Returns additive signal to blend with RSI direction.
        """
        if not self._is_crypto:
            return 0.0
        self._refresh_funding_rate()
        fr = self._funding_rate
        # Typical range is ±0.01% to ±0.1% per 8h; extreme = > ±0.05%
        scaled = float(np.clip(fr / 0.0005, -1.0, 1.0))  # 0.05% → full signal
        return -scaled * 0.3   # positive FR → negative (short) bias

    # ── Session gate ─────────────────────────────────────────────────────────

    def _is_equity_auction_window(self) -> bool:
        if self._is_crypto or self.instrument.startswith("FOREX:"):
            return False
        now = datetime.now(_ET)
        mins = now.hour * 60 + now.minute
        # Fix 6: extend opening block to 10:00 (full high-volatility open window).
        # Keep 3:30–4:00 block to avoid EOD positioning (separate from EOD close).
        return (9 * 60 + 30 <= mins < 10 * 60) or (15 * 60 + 30 <= mins < 16 * 60)

    def _is_eod_close_window(self) -> bool:
        """True for equity at ≥15:45 ET — triggers forced close of open positions."""
        if self._is_crypto or self.instrument.startswith("FOREX:"):
            return False
        now = datetime.now(_ET)
        mins = now.hour * 60 + now.minute
        return mins >= 15 * 60 + 45

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _ml_entry_scalar(self, rsi_side: int) -> float | None:
        """
        Return a notional scalar based on GodLevel signal agreement.
        rsi_side: +1 = want to go long, -1 = want to go short.
        Returns None to veto the entry.
        """
        if not self._godlevel_ok():
            return 1.0  # no ML data yet — neutral

        gl_side = 1 if self._gl_signal > 0.05 else (-1 if self._gl_signal < -0.05 else 0)

        if gl_side == 0:
            return 0.9  # ML uncertain — allow but size down slightly

        if gl_side == rsi_side:
            # ML agrees — scale up proportionally to confidence
            return 1.0 + 0.25 * self._gl_conf   # up to 1.25× at conf=1.0
        else:
            # ML disagrees
            if self._gl_conf >= 0.65:
                return None  # strong disagreement → veto
            return 0.7       # weak disagreement → allow but size down

    def _mtf_entry_scalar(self) -> float | None:
        """Return MTF-derived notional scalar. None = veto."""
        # If MTF data is insufficient, mtf_conf_adj defaults to 1.0 — no effect
        conf_adj = self._mtf_conf_adj
        if conf_adj < 0.83:
            return None   # strong TF misalignment → veto
        return conf_adj   # 0.85–1.10 depending on alignment

    def _regime_rsi_thresholds(self) -> tuple[float, float]:
        """Return (oversold, overbought) thresholds adapted to current VIX regime."""
        try:
            from risk.vix_regime_manager import get_global_vix_manager
            state = get_global_vix_manager().get_current_state()
            vr = state.regime.value   # "normal" | "elevated" | "fear" | "panic"
            if vr in ("fear", "panic"):
                return 30.0, 70.0   # only extreme RSI when markets are fearful
            if vr == "elevated":
                return 38.0, 62.0   # slightly tighter in cautious markets
        except Exception:
            pass
        return self.RSI_OVERSOLD, self.RSI_OVERBOUGHT

    @staticmethod
    def _rsi_conviction_scalar(rsi: float, threshold: float, side: int) -> float:
        """Scale 0.70→1.00 based on how far RSI extends beyond the entry threshold.

        At exactly the threshold (borderline signal) → 0.70× notional.
        At 10+ points beyond the threshold (high conviction) → 1.00× notional.
        Linear interpolation in between.  `side` is +1 for buy, -1 for sell.
        """
        distance = (threshold - rsi) * side   # positive when beyond the threshold
        distance = max(0.0, min(distance, 10.0))
        return 0.70 + 0.30 * (distance / 10.0)

    def _volume_confirmed(self) -> bool:
        """True when current 5-min bar volume ≥ 1.2× the 20-bar rolling average.
        Returns True (no filter) if fewer than 5 volume bars are available yet.
        """
        vols = list(self._5m_volumes)
        if len(vols) < 5:
            return True   # not enough history — allow entry during warmup
        avg = sum(vols[:-1]) / len(vols[:-1])   # average excluding the current bar
        if avg <= 0:
            return True
        return vols[-1] >= avg * 1.2

    def _try_entry(self, close: float, rsi: float, ema20: float, atr: float) -> None:
        if atr <= 0.0 or self._is_equity_auction_window():
            return

        # Volume confirmation: require above-average participation on the entry bar
        if not self._volume_confirmed():
            return

        # Post-stop-loss cooldown
        if time.monotonic() < self._cooldown_until:
            return

        # Daily circuit breaker: reset counter at session boundary, then gate
        today = datetime.now(_ET).date()
        if self._daily_pnl_date != today:
            self._daily_pnl_date      = today
            self._daily_realized_pnl  = 0.0
            self._daily_circuit_fired = False
        if self._daily_realized_pnl < self._DAILY_LOSS_LIMIT:
            if not self._daily_circuit_fired:
                logger.warning(
                    "DAILY CIRCUIT [%s]: realized P&L=$%.0f < limit=$%.0f — no new entries today",
                    self.instrument, self._daily_realized_pnl, self._DAILY_LOSS_LIMIT,
                )
                self._daily_circuit_fired = True
            return

        # VIX panic block — RegimeRouter HIGH_VOL_PANIC blocks new entries entirely
        if not self._is_crypto:
            try:
                from risk.regime_router import get_global_regime_router
                decision = get_global_regime_router().evaluate()
                if decision.block_new_entries:
                    return
            except Exception:
                pass

        # Regime-adaptive RSI thresholds (VIX-tuned)
        oversold, overbought = self._regime_rsi_thresholds()

        # Determine RSI direction
        if rsi < oversold and close > ema20:
            side, rsi_side = "buy", +1
        elif rsi > overbought and close < ema20:
            side, rsi_side = "sell", -1
        else:
            return

        # 1h trend-direction alignment: avoid fighting the intraday trend
        if not self._is_crypto:
            trend = self._1h_trend_direction()
            if trend == "up" and side == "sell":
                logger.debug("Trend filter: skip short in uptrend for %s", self.instrument)
                return
            if trend == "down" and side == "buy":
                logger.debug("Trend filter: skip long in downtrend for %s", self.instrument)
                return

        # Session VWAP filter: buy below VWAP, sell above VWAP (±0.5% buffer)
        vwap = self._session_vwap()
        if vwap is not None:
            if side == "buy" and close > vwap * 1.005:
                logger.debug("VWAP filter: skip long — close %.4f > vwap %.4f for %s",
                             close, vwap, self.instrument)
                return
            if side == "sell" and close < vwap * 0.995:
                logger.debug("VWAP filter: skip short — close %.4f < vwap %.4f for %s",
                             close, vwap, self.instrument)
                return

        # Earnings blackout gate (equity single-names only)
        if _earnings_blackout(self.instrument):
            logger.debug("Earnings blackout: skipping entry for %s", self.instrument)
            return

        # Crypto funding rate as additional directional bias
        if self._is_crypto:
            fr_bias = self._funding_rate_scalar()
            if rsi_side == 1 and fr_bias < -0.2:
                return
            if rsi_side == -1 and fr_bias > 0.2:
                return

        # GodLevel ML gate
        ml_scalar = self._ml_entry_scalar(rsi_side)
        if ml_scalar is None:
            logger.debug("GodLevel VETO entry %s %s (signal=%.3f conf=%.2f)",
                         side, self.instrument, self._gl_signal, self._gl_conf)
            return

        # MTF gate
        mtf_scalar = self._mtf_entry_scalar()
        if mtf_scalar is None:
            logger.debug("MTF VETO entry %s %s (conf_adj=%.2f aligned=%s)",
                         side, self.instrument, self._mtf_conf_adj, self._mtf_aligned)
            return

        # Conviction scalar: larger positions when RSI is deep in the oversold/overbought zone
        conviction = self._rsi_conviction_scalar(rsi, oversold if side == "buy" else overbought, rsi_side)
        final_notional = self.leg_notional * ml_scalar * mtf_scalar * conviction

        # VIX-adaptive notional scaling (equity only)
        if not self._is_crypto:
            try:
                from risk.vix_regime_manager import get_global_vix_manager
                vix_state = get_global_vix_manager().get_current_state()
                final_notional *= vix_state.risk_multiplier
            except Exception:
                pass

        # Portfolio net-exposure cap
        direction = "long" if side == "buy" else "short"
        if not _NET_EXPOSURE.can_open(final_notional, direction):
            logger.debug(
                "Net-exposure cap reached: skipping %s %s (notional=$%.0f)",
                direction, self.instrument, final_notional,
            )
            return

        # Portfolio heat cap: stop_price is entry ± STOP_ATR_MULT×ATR
        stop_estimate = (close - self.STOP_ATR_MULT * atr if side == "buy"
                         else close + self.STOP_ATR_MULT * atr)
        try:
            from risk.portfolio_heat import get_portfolio_heat
            if not get_portfolio_heat().can_open(self.instrument, close, stop_estimate, final_notional):
                return
        except Exception:
            pass

        self._open(side, close, atr, final_notional)

    def _open(self, side: str, price: float, atr: float, notional: float) -> None:
        self._state       = "long" if side == "buy" else "short"
        self._entry_price = price
        self._atr_at_entry    = atr
        self._trailing_active = False
        self._trail_hwm       = price
        self._half_closed     = False
        self._open_notional   = abs(notional)
        _NET_EXPOSURE.register(self.instrument, abs(notional), self._state)
        if self._state == "long":
            self._stop_price = price - self.STOP_ATR_MULT * atr
            self._tp_price   = price + self.TP_ATR_MULT   * atr
        else:
            self._stop_price = price + self.STOP_ATR_MULT * atr
            self._tp_price   = price - self.TP_ATR_MULT   * atr

        # Register heat now that stop_price is set
        try:
            from risk.portfolio_heat import get_portfolio_heat
            get_portfolio_heat().register(self.instrument, price, self._stop_price, abs(notional))
        except Exception:
            pass

        target = notional if side == "buy" else -notional
        self._emit_signal(side, target, metadata={
            "trigger": "rsi_5m_entry",
            "ml_signal": round(self._gl_signal, 4),
            "ml_conf":   round(self._gl_conf, 4),
            "mtf_aligned": self._mtf_aligned,
        })
        logger.info(
            "DirectionalEquity ENTRY %s %s @ %.4f | stop=%.4f tp=%.4f notional=$%.0f "
            "ml=%.2f mtf_adj=%.2f",
            side.upper(), self.instrument, price,
            self._stop_price, self._tp_price, abs(target),
            self._gl_signal, self._mtf_conf_adj,
        )

    def _half_close(self, price: float, reason: str) -> None:
        """Close 50% of position — banks guaranteed P&L, lets remaining 50% trail."""
        self._half_closed = True
        remaining  = self._open_notional * 0.5
        exit_side  = "sell" if self._state == "long" else "buy"
        target     = remaining if self._state == "long" else -remaining
        self._emit_signal(exit_side, target, metadata={"trigger": reason, "partial": True})
        _NET_EXPOSURE.register(self.instrument, remaining, self._state)
        logger.info(
            "DirectionalEquity PARTIAL EXIT %s %s @ %.4f | banked 50%% | remaining=$%.0f",
            exit_side.upper(), self.instrument, price, remaining,
        )

    # ── Position management ───────────────────────────────────────────────────

    def _manage_position(self, close: float, rsi: float) -> None:
        # EOD forced close must come first
        if self._is_eod_close_window():
            self._close_position(close, "eod_close")
            return

        atr = self._atr_at_entry or 1e-6

        # After 14:30 ET, tighten trailing stop to lock in afternoon gains before
        # EOD rebalancing noise pushes price back against intraday trend positions.
        trail_mult = (self.TRAIL_ATR_MULT * self._AFTERNOON_TRAIL_TIGHTEN
                      if self._et_mins() >= 14 * 60 + 30
                      else self.TRAIL_ATR_MULT)

        if self._state == "long":
            self._trail_hwm = max(self._trail_hwm, close)

            # Staged exit: at TP1_ATR_MULT (1.5×ATR) bank 50% and start trailing
            if not self._half_closed and close >= self._entry_price + self.TP1_ATR_MULT * atr:
                self._half_close(close, "partial_tp1")
                self._trailing_active = True
                return

            if self._trailing_active:
                trail_stop = self._trail_hwm - trail_mult * atr
                if close <= trail_stop:
                    self._close_position(close, "trailing_stop")
                    return

            # After partial TP the trailing stop is the sole exit for the second
            # leg — RSI will naturally cross 50 as price rises and would close the
            # remaining position immediately, defeating the purpose.
            if not self._half_closed and rsi >= self.RSI_EXIT:
                self._close_position(close, "rsi_exit")
            elif close <= self._stop_price:
                self._close_position(close, "stop_loss")
            elif close >= self._tp_price:
                self._close_position(close, "take_profit")
        else:
            self._trail_hwm = min(self._trail_hwm, close)

            # Staged exit: at TP1_ATR_MULT (1.5×ATR) bank 50% and start trailing
            if not self._half_closed and close <= self._entry_price - self.TP1_ATR_MULT * atr:
                self._half_close(close, "partial_tp1")
                self._trailing_active = True
                return

            if self._trailing_active:
                trail_stop = self._trail_hwm + trail_mult * atr
                if close >= trail_stop:
                    self._close_position(close, "trailing_stop")
                    return

            if not self._half_closed and rsi <= self.RSI_EXIT:
                self._close_position(close, "rsi_exit")
            elif close >= self._stop_price:
                self._close_position(close, "stop_loss")
            elif close <= self._tp_price:
                self._close_position(close, "take_profit")

    def _close_position(self, price: float, reason: str) -> None:
        exit_side = "sell" if self._state == "long" else "buy"
        self._emit_signal(exit_side, 0.0, metadata={"trigger": reason})
        logger.info(
            "DirectionalEquity EXIT %s %s @ %.4f | reason=%s",
            exit_side.upper(), self.instrument, price, reason,
        )
        _NET_EXPOSURE.deregister(self.instrument)
        # Deregister heat
        try:
            from risk.portfolio_heat import get_portfolio_heat
            get_portfolio_heat().deregister(self.instrument)
        except Exception:
            pass
        # Track realized P&L for daily circuit breaker
        if self._entry_price > 0 and self._open_notional > 0:
            direction = 1.0 if self._state == "long" else -1.0
            pnl = direction * (price - self._entry_price) / self._entry_price * self._open_notional
            self._daily_realized_pnl += pnl
        # 15-min cooldown after stop-loss to prevent re-entering the same losing move
        if reason == "stop_loss":
            self._cooldown_until = time.monotonic() + 900.0
        self._state         = "flat"
        self._entry_price   = 0.0
        self._stop_price    = 0.0
        self._tp_price      = 0.0
        self._half_closed   = False
        self._open_notional = 0.0

    # ── Signal emit ───────────────────────────────────────────────────────────

    def _emit_signal(self, side: str, target_value: float, metadata: dict[str, Any] | None = None) -> None:
        conf = 0.75 * (self._mtf_conf_adj if self._mtf_conf_adj > 0 else 1.0)
        conf = float(np.clip(conf, 0.50, 1.0))
        self.emit_signal(
            instrument_id=self.instrument,
            target_type="notional",
            target_value=target_value,
            confidence=conf,
            stop_model="atr_stop",
            metadata={**(metadata or {}), "source": "directional_equity"},
        )

    def close(self) -> None:
        super().close()
        if self._state != "flat":
            self._emit_signal("flatten", 0.0, metadata={"trigger": "slot_closed"})
            _NET_EXPOSURE.deregister(self.instrument)
            self._state = "flat"
