"""
quant_system/strategies/cross_sectional_momentum_strategy.py

CrossSectionalMomentumStrategy — Jegadeesh-Titman adapted for intraday.

Signal pipeline:
  1. Accumulate 1-min → 5-min → 1-hour bars for each symbol in UNIVERSE
  2. Every RANK_INTERVAL_BARS 5-min bars (≈1h), compute cumulative log-return
     over the past LOOKBACK_HOURS 1-hour bars, skipping SKIP_HOURS most recent
     (avoids short-term reversal contamination)
  3. Rank all symbols with sufficient history
  4. Long top N_POSITIONS, short bottom N_POSITIONS
  5. Close positions that fall out of the ranked bands

Slot wiring:
  - Uses instrument_a="SPY" as placeholder (controls market-hours gate)
  - Internally watches all UNIVERSE symbols via BarEvent subscription
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, TradeTick
from quant_system.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Intraday gap filter: skip entries when stock already moved >2.5% from today's open
_GAP_FILTER_THRESHOLD = 0.025

# Equity universe to rank — all already in IEX subscription
_UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "AMD",  "V",    "MA",    "JPM",
    "BAC",  "KO",   "PEP",  "SPY",   "QQQ",
]


class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Portfolio-level cross-sectional momentum over a fixed equity universe.
    Reranks every ~1h, longs top N, shorts bottom N.
    """

    RANK_INTERVAL_BARS        = 12       # rerank every 12 × 5-min bars ≈ 1h
    LOOKBACK_HOURS            = 20       # momentum window: 20 × 1h bars ≈ 4 sessions
    SKIP_HOURS                = 1        # skip last 1h to avoid short-term reversal
    N_POSITIONS               = 3        # long top-N and short bottom-N
    MIN_HOURS_FOR_RANK        = 5        # need at least 5 hourly bars before any signal
    WALLCLOCK_RERANK_INTERVAL = 65 * 60  # fallback: rerank if SPY feed silent >65 min

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        instrument: str = "SPY",        # placeholder for market-hours gate
        leg_notional: float = 1_500.0,  # notional per long/short leg
        warmup_bars: int = 14,
    ) -> None:
        super().__init__(event_bus)
        self.instrument   = instrument   # used by harness for session gate only
        self.leg_notional = float(leg_notional)
        self.warmup_bars  = warmup_bars

        # Per-symbol 1-min → 5-min accumulator
        self._1m_count: dict[str, int] = {sym: 0 for sym in _UNIVERSE}
        self._5m_acc:   dict[str, dict[str, float] | None] = {sym: None for sym in _UNIVERSE}

        # Per-symbol 5-min bar count (rerank trigger) and 5-min close history
        self._5m_bar_count: dict[str, int] = {sym: 0 for sym in _UNIVERSE}
        self._5m_closes:    dict[str, deque[float]] = {sym: deque(maxlen=14) for sym in _UNIVERSE}

        # Per-symbol 1-hour accumulator and history
        self._5m_in_hour: dict[str, int] = {sym: 0 for sym in _UNIVERSE}
        self._1h_acc:     dict[str, dict[str, float] | None] = {sym: None for sym in _UNIVERSE}
        self._1h_closes:  dict[str, deque[float]] = {
            sym: deque(maxlen=self.LOOKBACK_HOURS + self.SKIP_HOURS + 2)
            for sym in _UNIVERSE
        }

        # Global 5-min bar counter (drives rerank cadence)
        self._global_5m = 0

        # Current open positions: symbol → side ("long" | "short")
        self._open_positions: dict[str, str] = {}

        # Intraday gap filter: track first 5-min close of each ET trading day
        self._day_open:  dict[str, float | None] = {sym: None for sym in _UNIVERSE}
        self._day_date:  dict[str, object]        = {sym: None for sym in _UNIVERSE}

        # Timestamp of last rerank (for logging and wall-clock fallback)
        self._last_rank_ts: float = 0.0

        logger.info(
            "CrossSectionalMomentum: watching %d symbols | top/bottom %d | notional=$%.0f",
            len(_UNIVERSE), self.N_POSITIONS, self.leg_notional,
        )

    # ── Bar handler ───────────────────────────────────────────────────────────

    def on_bar(self, event: BarEvent) -> None:
        sym = event.instrument_id
        if sym not in _UNIVERSE:
            return

        close  = float(event.close_price)
        high   = float(event.high_price)
        low    = float(event.low_price)
        volume = float(event.volume)
        open_  = float(event.open_price)

        # ── Accumulate into 5-min bar ─────────────────────────────────────────
        if self._5m_acc[sym] is None:
            self._5m_acc[sym] = {"o": open_, "h": high, "l": low, "c": close, "v": volume}
        else:
            acc = self._5m_acc[sym]
            acc["h"] = max(acc["h"], high)
            acc["l"] = min(acc["l"], low)
            acc["c"] = close
            acc["v"] += volume
        self._1m_count[sym] += 1

        if self._1m_count[sym] < 5:
            return  # 5-min bar not yet complete

        # ── Emit 5-min bar ────────────────────────────────────────────────────
        b = self._5m_acc[sym]
        self._5m_closes[sym].append(b["c"])
        self._5m_bar_count[sym] += 1

        # Track first 5-min close of each ET session for intraday gap filter
        today = datetime.now(_ET).date()
        if self._day_date[sym] != today:
            self._day_open[sym] = b["c"]
            self._day_date[sym] = today

        # Roll into 1-hour accumulator
        self._accumulate_1h(sym, b)

        # Reset
        self._5m_acc[sym]    = None
        self._1m_count[sym]  = 0

        # Use SPY as the clock signal so we rerank on a consistent cadence.
        # Any symbol triggers the wall-clock fallback when SPY feed is silent.
        if sym == "SPY":
            self._global_5m += 1

            # Forced EOD exit: close all momentum positions by 15:30 ET.
            # Overnight gap risk on a momentum long/short book is asymmetric —
            # longs are today's winners (highest reversal probability) and shorts
            # are today's losers (highest overnight squeeze risk).
            eod_mins = datetime.now(_ET)
            eod_mins_val = eod_mins.hour * 60 + eod_mins.minute
            if eod_mins_val >= 15 * 60 + 30 and self._open_positions:
                logger.info(
                    "XSecMomentum EOD close: forcing exit of %d positions at 15:30 ET",
                    len(self._open_positions),
                )
                for sym_pos in list(self._open_positions.keys()):
                    self._close_position(sym_pos)
                return

            if self._global_5m % self.RANK_INTERVAL_BARS == 0:
                self._rerank_and_trade()

        elif time.monotonic() - self._last_rank_ts > self.WALLCLOCK_RERANK_INTERVAL:
            # SPY feed is lagging — use wall-clock to keep reranking alive.
            logger.warning(
                "XSecMomentum: SPY feed silent >%ds — triggering wall-clock rerank",
                self.WALLCLOCK_RERANK_INTERVAL,
            )
            self._rerank_and_trade()

    def on_tick(self, event: TradeTick) -> None:
        pass

    # ── 1-hour accumulator ────────────────────────────────────────────────────

    def _accumulate_1h(self, sym: str, b: dict) -> None:
        if self._1h_acc[sym] is None:
            self._1h_acc[sym] = dict(b)
        else:
            h = self._1h_acc[sym]
            h["h"] = max(h["h"], b["h"])
            h["l"] = min(h["l"], b["l"])
            h["c"] = b["c"]
            h["v"] += b["v"]
        self._5m_in_hour[sym] += 1

        if self._5m_in_hour[sym] >= 12:
            self._1h_closes[sym].append(self._1h_acc[sym]["c"])
            self._1h_acc[sym]    = None
            self._5m_in_hour[sym] = 0

    # ── Vol-normalized notional ───────────────────────────────────────────────

    # Target $40 of 1-sigma daily P&L per leg; clamp to [0.4×, 2.5×] base notional.
    _TARGET_DAILY_VOL_USD = 40.0

    def _vol_normalized_notional(self, sym: str) -> float:
        closes = list(self._1h_closes[sym])
        if len(closes) < 5:
            return self.leg_notional
        log_rets = np.diff(np.log(np.maximum(closes, 1e-9)))
        hourly_vol = float(np.std(log_rets)) if len(log_rets) >= 2 else 0.0
        if hourly_vol < 1e-6:
            return self.leg_notional
        # ~6.5 trading hours per day
        daily_vol = hourly_vol * np.sqrt(6.5)
        price = closes[-1]
        notional = self._TARGET_DAILY_VOL_USD / (daily_vol * price) * price
        lo = self.leg_notional * 0.4
        hi = self.leg_notional * 2.5
        return float(np.clip(notional, lo, hi))

    # ── Momentum scoring ──────────────────────────────────────────────────────

    def _momentum_score(self, sym: str) -> float | None:
        """
        Cumulative log return from bar [-LOOKBACK_HOURS-SKIP_HOURS] to [-SKIP_HOURS].
        Returns None if insufficient history.
        """
        closes = list(self._1h_closes[sym])
        needed = self.LOOKBACK_HOURS + self.SKIP_HOURS + 1
        if len(closes) < needed:
            return None

        # Window: from -(LOOKBACK_HOURS + SKIP_HOURS) to -(SKIP_HOURS)
        start_close = closes[-(self.LOOKBACK_HOURS + self.SKIP_HOURS + 1)]
        end_close   = closes[-(self.SKIP_HOURS + 1)]
        if start_close <= 0.0:
            return None
        return float(np.log(end_close / start_close))

    # ── Market-regime lean ────────────────────────────────────────────────────

    def _spy_regime_lean(self) -> tuple[float, float]:
        """Return (long_mult, short_mult) based on SPY intraday position vs VWAP.

        Uses SPY's intraday move vs its day-open as a bear/bull proxy:
          SPY up >0.5% (bull lean): boost longs ×1.30, trim shorts ×0.70
          SPY down >0.5% (bear lean): trim longs ×0.70, boost shorts ×1.30
          Neutral otherwise: 1.0 / 1.0

        Limits unnecessary exposure in trending markets on the wrong side.
        """
        spy_closes   = self._5m_closes.get("SPY")
        spy_day_open = self._day_open.get("SPY")
        if not spy_closes or not spy_day_open or spy_day_open <= 0:
            return 1.0, 1.0
        intraday = (spy_closes[-1] - spy_day_open) / spy_day_open
        if intraday > 0.005:
            return 1.30, 0.70   # bull lean: ride longs, trim shorts
        if intraday < -0.005:
            return 0.70, 1.30   # bear lean: trim longs, ride shorts
        return 1.0, 1.0

    # ── Ranking and signal dispatch ────────────────────────────────────────────

    def _rerank_and_trade(self) -> None:
        scores: dict[str, float] = {}
        for sym in _UNIVERSE:
            score = self._momentum_score(sym)
            if score is not None:
                scores[sym] = score

        n_ranked = len(scores)
        if n_ranked < self.MIN_HOURS_FOR_RANK * 2:
            return  # not enough symbols ranked yet

        ranked   = sorted(scores, key=scores.__getitem__, reverse=True)
        top_n    = set(ranked[:self.N_POSITIONS])
        bottom_n = set(ranked[-self.N_POSITIONS:])

        long_mult, short_mult = self._spy_regime_lean()

        desired: dict[str, str] = {}
        for sym in top_n:
            desired[sym] = "long"
        for sym in bottom_n:
            desired[sym] = "short"

        # Close positions in middle quintile or reversed side
        for sym, current_side in list(self._open_positions.items()):
            wanted = desired.get(sym)
            if wanted is None or wanted != current_side:
                self._close_position(sym)

        # Open new desired positions
        for sym, side in desired.items():
            if sym not in self._open_positions:
                self._open_position(sym, side, scores.get(sym, 0.0),
                                    long_mult=long_mult, short_mult=short_mult)

        self._last_rank_ts = time.monotonic()
        logger.info(
            "XSecMomentum rerank: %d scored | long=%s | short=%s | lean=(%.2fx/%.2fx)",
            n_ranked, sorted(top_n), sorted(bottom_n), long_mult, short_mult,
        )

    # ── Signal helpers ────────────────────────────────────────────────────────

    def _open_position(
        self,
        sym: str,
        side: str,
        score: float,
        *,
        long_mult: float = 1.0,
        short_mult: float = 1.0,
    ) -> None:
        # Intraday gap filter: skip if stock already moved >2.5% from today's session open.
        # Prevents chasing a move that has already happened on the ranking horizon.
        closes = self._5m_closes.get(sym)
        day_open = self._day_open.get(sym)
        if closes and day_open and day_open > 0:
            current_close  = closes[-1]
            intraday_move  = (current_close - day_open) / day_open
            if side == "long" and intraday_move > _GAP_FILTER_THRESHOLD:
                logger.debug(
                    "XSec GAP FILTER: skip long %s — already +%.1f%% today",
                    sym, intraday_move * 100,
                )
                return
            if side == "short" and intraday_move < -_GAP_FILTER_THRESHOLD:
                logger.debug(
                    "XSec GAP FILTER: skip short %s — already -%.1f%% today",
                    sym, intraday_move * 100,
                )
                return

        notional = self._vol_normalized_notional(sym)
        lean = long_mult if side == "long" else short_mult
        notional *= lean
        target = notional if side == "long" else -notional

        # Portfolio heat gate: approximate stop as 2% of current price (equity momentum
        # typical intraday stop). Prevents aggregate dollar-risk exceeding the global cap.
        closes = self._5m_closes.get(sym)
        entry_price = closes[-1] if closes else 0.0
        stop_price  = entry_price * (0.98 if side == "long" else 1.02)
        try:
            from risk.portfolio_heat import get_portfolio_heat
            if not get_portfolio_heat().can_open(sym, entry_price, stop_price, abs(notional)):
                logger.debug("XSec HEAT GATE: skip %s %s — portfolio too hot", side.upper(), sym)
                return
            get_portfolio_heat().register(sym, entry_price, stop_price, abs(notional))
        except Exception:
            pass

        self.emit_signal(
            instrument_id=sym,
            target_type="notional",
            target_value=target,
            confidence=0.65,
            stop_model="rank_reversal",
            metadata={"source": "cross_sectional_momentum", "momentum_score": round(score, 5)},
        )
        self._open_positions[sym] = side
        logger.info(
            "XSecMomentum ENTRY %s %s | score=%.4f notional=$%.0f (vol-normalized)",
            side.upper(), sym, score, abs(target),
        )

    def _close_position(self, sym: str) -> None:
        current = self._open_positions.pop(sym, None)
        if current is None:
            return
        self.emit_signal(
            instrument_id=sym,
            target_type="notional",
            target_value=0.0,
            confidence=0.65,
            stop_model="rank_reversal",
            metadata={"source": "cross_sectional_momentum", "trigger": "rank_exit"},
        )
        try:
            from risk.portfolio_heat import get_portfolio_heat
            get_portfolio_heat().deregister(sym)
        except Exception:
            pass
        logger.info("XSecMomentum EXIT %s %s", sym, current)

    def close(self) -> None:
        super().close()
        for sym in list(self._open_positions.keys()):
            self._close_position(sym)

    # ── Status (for health endpoint) ─────────────────────────────────────────

    @property
    def last_z_score(self) -> float:
        return 0.0  # harness accesses this on all strategies
