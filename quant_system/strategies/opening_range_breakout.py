"""
quant_system/strategies/opening_range_breakout.py

OpeningRangeBreakoutStrategy — intraday equity ORB.

Mechanics
---------
  1. First 15 min of the equity session (9:30–9:45 ET) accumulate the range:
       ORB_high = max(high) across all 1-min bars in the window
       ORB_low  = min(low)  across all 1-min bars in the window
  2. The first 5-min close after 9:45 ET that breaks out of the range triggers
     an entry in the breakout direction:
       close > ORB_high  →  long  (stop = ORB midpoint, target = entry + 2×range)
       close < ORB_low   →  short (stop = ORB midpoint, target = entry - 2×range)
  3. One trade per day; resets at midnight ET.
  4. No new entries after 12:00 ET (stale ORB, trend usually resolved by then).
  5. Forced EOD close at 15:45 ET — same as DirectionalEquityStrategy.

Why 2×range target?  Measured R:R on ORB studies (Toby Crabel, Cooper et al.)
converges around 2:1 with a midpoint stop, giving a break-even win rate of 33%.
Historical intraday hit rates of 55–60% produce positive expectancy.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, date
from zoneinfo import ZoneInfo

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, TradeTick
from quant_system.strategies.base import BaseStrategy

_ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)

_ORB_BUILD_START  = 9 * 60 + 30   # 9:30 ET (minutes since midnight)
_ORB_BUILD_END    = 9 * 60 + 45   # 9:45 ET — end of range-building window
_ORB_ENTRY_CUTOFF = 12 * 60       # no new entries after 12:00 ET
_ORB_EOD_CLOSE    = 15 * 60 + 45  # forced close at 15:45 ET
_ORB_MAX_RANGE_PCT     = 0.02   # skip breakouts when range > 2% of mid (gap day)
_ORB_COMPRESSION_RATIO = 0.80   # skip if today's range > 80% of 10-session avg (not compressed)
_ORB_MIN_VOL_RATIO     = 0.50   # skip if build-period volume < 50% of 20-session average
_ORB_DAILY_LOSS_LIMIT  = -200.0 # per-strategy daily circuit breaker (USD)


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Classic 15-min Opening Range Breakout for a single equity instrument.
    One trade per day, always closed by 15:45 ET.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        instrument: str,
        leg_notional: float = 2_000.0,
        warmup_bars: int = 1,
    ) -> None:
        super().__init__(event_bus)
        self.instrument   = instrument
        self.leg_notional = float(leg_notional)
        self.warmup_bars  = int(warmup_bars)   # unused internally; kept for harness slot compatibility

        # ORB range (reset each session)
        self._orb_date:       date | None = None
        self._orb_high:       float       = 0.0
        self._orb_low:        float       = float("inf")
        self._orb_ready:      bool        = False
        self._trade_done:     bool        = False
        self._orb_stop_count: int         = 0
        self._orb_break_dir:  str         = ""
        self._reverse_eligible: bool      = False

        # ATR-compression quality filter (#4): rolling history of ORB ranges across sessions
        self._historical_orb_ranges: deque[float] = deque(maxlen=10)

        # Build-period volume floor (#7): track volume during 9:30–9:45 and compare to history
        self._orb_build_vol:      float         = 0.0
        self._historical_build_vols: deque[float] = deque(maxlen=20)

        # Daily circuit breaker (#2): stop new entries after cumulative loss exceeds limit
        self._daily_realized_pnl:  float      = 0.0
        self._daily_pnl_date:      date | None = None
        self._daily_circuit_fired: bool        = False

        # Position state
        self._state:       str   = "flat"
        self._entry_price: float = 0.0
        self._stop_price:  float = 0.0
        self._tp_price:    float = 0.0

        # 1-min → 5-min bar accumulator
        self._1m_count = 0
        self._5m_acc:  dict[str, float] | None = None

        # Harness compatibility (AlphaMonitorController reads this)
        self.last_z_score: float = 0.0

        logger.info(
            "OpeningRangeBreakout: watching %s | notional=$%.0f",
            self.instrument, self.leg_notional,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _et_mins() -> int:
        now = datetime.now(_ET)
        return now.hour * 60 + now.minute

    def _reset_session(self, today: date) -> None:
        # Archive yesterday's stats into rolling history before wiping
        if self._orb_build_vol > 0:
            self._historical_build_vols.append(self._orb_build_vol)
        # ORB range is archived in the "ORB ready" block when the range is confirmed

        self._orb_date         = today
        self._orb_high         = 0.0
        self._orb_low          = float("inf")
        self._orb_ready        = False
        self._trade_done       = False
        self._orb_stop_count   = 0
        self._orb_break_dir    = ""
        self._reverse_eligible = False
        self._orb_build_vol    = 0.0

    # ── Bar handler ───────────────────────────────────────────────────────────

    def on_bar(self, event: BarEvent) -> None:
        if event.instrument_id != self.instrument:
            return

        high  = float(event.high_price)
        low   = float(event.low_price)
        close = float(event.close_price)
        open_ = float(event.open_price)
        vol   = float(event.volume)

        # Accumulate into 5-min bar
        if self._5m_acc is None:
            self._5m_acc = {"o": open_, "h": high, "l": low, "c": close, "v": vol}
        else:
            self._5m_acc["h"] = max(self._5m_acc["h"], high)
            self._5m_acc["l"] = min(self._5m_acc["l"], low)
            self._5m_acc["c"] = close
            self._5m_acc["v"] += vol
        self._1m_count += 1

        # Feed ORB range and build-period volume from every 1-min bar in the window
        mins = self._et_mins()
        if _ORB_BUILD_START <= mins < _ORB_BUILD_END:
            self._orb_high      = max(self._orb_high, high)
            self._orb_low       = min(self._orb_low,  low)
            self._orb_build_vol += vol

        if self._1m_count < 5:
            return  # 5-min bar not complete yet

        b = self._5m_acc
        self._5m_acc   = None
        self._1m_count = 0

        # Session reset at ET date boundary
        today = datetime.now(_ET).date()
        if self._orb_date != today:
            self._reset_session(today)

        # Mark range ready once we are past the build window and range is valid
        if not self._orb_ready and mins >= _ORB_BUILD_END:
            if (self._orb_high > 0.0
                    and self._orb_low < float("inf")
                    and self._orb_high > self._orb_low):
                self._orb_ready = True
                rng = self._orb_high - self._orb_low
                mid = (self._orb_high + self._orb_low) / 2.0
                self._historical_orb_ranges.append(rng)   # archive for compression filter
                logger.info(
                    "ORB ready [%s]: high=%.4f low=%.4f range=%.4f (%.2f%% of mid)",
                    self.instrument, self._orb_high, self._orb_low,
                    rng, rng / mid * 100,
                )

        if self._state != "flat":
            self._manage_position(b["c"])
        elif self._orb_ready and not self._trade_done and mins < _ORB_ENTRY_CUTOFF:
            if self._reverse_eligible:
                self._try_reverse_entry(b["c"])
            else:
                self._try_entry(b["c"])

    def on_tick(self, event: TradeTick) -> None:
        pass

    # ── Entry logic ───────────────────────────────────────────────────────────

    def _try_entry(self, close: float) -> None:
        # Daily circuit breaker: reset at session boundary, then gate
        today = datetime.now(_ET).date()
        if self._daily_pnl_date != today:
            self._daily_pnl_date      = today
            self._daily_realized_pnl  = 0.0
            self._daily_circuit_fired = False
        if self._daily_realized_pnl < _ORB_DAILY_LOSS_LIMIT:
            if not self._daily_circuit_fired:
                logger.warning(
                    "ORB DAILY CIRCUIT [%s]: P&L=$%.0f < limit=$%.0f — no new entries today",
                    self.instrument, self._daily_realized_pnl, _ORB_DAILY_LOSS_LIMIT,
                )
                self._daily_circuit_fired = True
            return

        orb_range = self._orb_high - self._orb_low
        if orb_range <= 0.0:
            return

        midpoint = (self._orb_high + self._orb_low) / 2.0

        # Wide-range guard: gap days create ORBs too wide to trade profitably.
        if orb_range / midpoint > _ORB_MAX_RANGE_PCT:
            logger.debug(
                "ORB wide-range skip [%s]: %.2f%% > %.1f%%",
                self.instrument, orb_range / midpoint * 100, _ORB_MAX_RANGE_PCT * 100,
            )
            return

        # ATR-compression quality filter: only trade when the range is compressed
        # relative to recent sessions (Crabel NR concept — tight range → expansion likely).
        if len(self._historical_orb_ranges) >= 3:
            avg_hist = sum(self._historical_orb_ranges) / len(self._historical_orb_ranges)
            if avg_hist > 0 and orb_range > avg_hist * _ORB_COMPRESSION_RATIO:
                logger.debug(
                    "ORB compression skip [%s]: range=%.4f > %.0f%% of avg=%.4f",
                    self.instrument, orb_range, _ORB_COMPRESSION_RATIO * 100, avg_hist,
                )
                return

        # Volume floor: thin build-period volume means the range levels are unreliable.
        if len(self._historical_build_vols) >= 5:
            avg_vol = sum(self._historical_build_vols) / len(self._historical_build_vols)
            if avg_vol > 0 and self._orb_build_vol < avg_vol * _ORB_MIN_VOL_RATIO:
                logger.debug(
                    "ORB thin-volume skip [%s]: build_vol=%.0f < %.0f%% of avg=%.0f",
                    self.instrument, self._orb_build_vol, _ORB_MIN_VOL_RATIO * 100, avg_vol,
                )
                return

        if close > self._orb_high:
            self._open("buy",  close, stop=midpoint, target=close + 2.0 * orb_range)
        elif close < self._orb_low:
            self._open("sell", close, stop=midpoint, target=close - 2.0 * orb_range)

    def _open(
        self,
        side: str,
        price: float,
        stop: float,
        target: float,
        notional_override: float | None = None,
    ) -> None:
        self._state       = "long" if side == "buy" else "short"
        self._entry_price = price
        self._stop_price  = stop
        self._tp_price    = target
        self._trade_done  = True
        if not self._orb_break_dir:
            self._orb_break_dir = side

        base_notional = notional_override if notional_override is not None else self.leg_notional

        # Portfolio heat check before committing
        try:
            from risk.portfolio_heat import get_portfolio_heat
            if not get_portfolio_heat().can_open(self.instrument, price, stop, base_notional):
                # Undo the state changes; the position won't be opened
                self._state       = "flat"
                self._entry_price = 0.0
                self._stop_price  = 0.0
                self._tp_price    = 0.0
                self._trade_done  = False
                return
            get_portfolio_heat().register(self.instrument, price, stop, base_notional)
        except Exception:
            pass

        notional = base_notional if side == "buy" else -base_notional
        self.emit_signal(
            instrument_id=self.instrument,
            target_type="notional",
            target_value=notional,
            confidence=0.72,
            stop_model="orb_midpoint",
            metadata={
                "source":    "orb",
                "orb_high":  round(self._orb_high, 4),
                "orb_low":   round(self._orb_low,  4),
                "stop":      round(stop,   4),
                "target":    round(target, 4),
            },
        )
        logger.info(
            "ORB ENTRY %s %s @ %.4f | stop=%.4f target=%.4f | "
            "range=%.4f notional=$%.0f",
            side.upper(), self.instrument, price,
            stop, target,
            self._orb_high - self._orb_low, abs(notional),
        )

    # ── Position management ───────────────────────────────────────────────────

    def _manage_position(self, close: float) -> None:
        if self._et_mins() >= _ORB_EOD_CLOSE:
            self._close("eod_close", close)
            return

        if self._state == "long":
            if close <= self._stop_price:
                self._close("stop_loss", close)
            elif close >= self._tp_price:
                self._close("take_profit", close)
        else:
            if close >= self._stop_price:
                self._close("stop_loss", close)
            elif close <= self._tp_price:
                self._close("take_profit", close)

    def _close(self, reason: str, price: float) -> None:
        exit_side = "sell" if self._state == "long" else "buy"
        self.emit_signal(
            instrument_id=self.instrument,
            target_type="notional",
            target_value=0.0,
            confidence=1.0,
            stop_model="orb_exit",
            metadata={"source": "orb", "trigger": reason},
        )
        logger.info(
            "ORB EXIT %s %s @ %.4f | reason=%s",
            exit_side.upper(), self.instrument, price, reason,
        )
        # Track daily P&L for circuit breaker
        if self._entry_price > 0:
            direction = 1.0 if self._state == "long" else -1.0
            pnl = direction * (price - self._entry_price) / self._entry_price * self.leg_notional
            self._daily_realized_pnl += pnl

        # Deregister heat
        try:
            from risk.portfolio_heat import get_portfolio_heat
            get_portfolio_heat().deregister(self.instrument)
        except Exception:
            pass

        self._state       = "flat"
        self._entry_price = 0.0
        self._stop_price  = 0.0
        self._tp_price    = 0.0

        # Failed-breakout reverse: allow one counter-trade at half notional
        if reason == "stop_loss" and self._orb_stop_count == 0:
            self._orb_stop_count += 1
            if self._et_mins() < _ORB_ENTRY_CUTOFF:
                self._trade_done     = False   # allow the counter entry
                self._reverse_eligible = True
                logger.info(
                    "ORB failed-breakout: reverse eligible for %s (break_dir=%s)",
                    self.instrument, self._orb_break_dir,
                )

    # ── Failed-breakout counter-trade ─────────────────────────────────────────

    def _try_reverse_entry(self, close: float) -> None:
        """Enter a half-notional counter-trade when the opposite ORB boundary is crossed."""
        orb_range = self._orb_high - self._orb_low
        if orb_range <= 0.0:
            return
        midpoint = (self._orb_high + self._orb_low) / 2.0
        # Skip if the range is too wide (same guard as original entry)
        if orb_range / midpoint > _ORB_MAX_RANGE_PCT:
            self._reverse_eligible = False
            return

        half = self.leg_notional * 0.5
        if self._orb_break_dir == "buy" and close < self._orb_low:
            # Long breakout failed → price back below low: go short at half size
            self._reverse_eligible = False
            self._open("sell", close,
                       stop=midpoint, target=close - 2.0 * orb_range,
                       notional_override=half)
            logger.info(
                "ORB REVERSE short %s @ %.4f | half_notional=$%.0f",
                self.instrument, close, half,
            )
        elif self._orb_break_dir == "sell" and close > self._orb_high:
            # Short breakout failed → price back above high: go long at half size
            self._reverse_eligible = False
            self._open("buy", close,
                       stop=midpoint, target=close + 2.0 * orb_range,
                       notional_override=half)
            logger.info(
                "ORB REVERSE long %s @ %.4f | half_notional=$%.0f",
                self.instrument, close, half,
            )

    def close(self) -> None:
        super().close()
        if self._state != "flat":
            self.emit_signal(
                instrument_id=self.instrument,
                target_type="notional",
                target_value=0.0,
                confidence=1.0,
                stop_model="orb_exit",
                metadata={"source": "orb", "trigger": "slot_closed"},
            )
            self._state = "flat"

    @property
    def pair_label(self) -> str:
        return f"{self.instrument}/{self.instrument}"
