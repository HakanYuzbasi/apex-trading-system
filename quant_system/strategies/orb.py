"""
quant_system/strategies/orb.py

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
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent, TradeTick
from quant_system.strategies.base import BaseStrategy

_ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)

_ORB_BUILD_START  = 9 * 60 + 30   # 9:30 ET
_ORB_BUILD_END    = 9 * 60 + 45   # 9:45 ET
_ORB_ENTRY_CUTOFF = 12 * 60       # 12:00 ET
_ORB_EOD_CLOSE    = 15 * 60 + 45  # 15:45 ET
_ORB_MAX_RANGE_PCT     = 0.02
_ORB_COMPRESSION_RATIO = 0.80
_ORB_MIN_VOL_RATIO     = 0.50
_ORB_DAILY_LOSS_LIMIT  = -200.0

class OpeningRangeBreakoutStrategy(BaseStrategy):
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
        self.warmup_bars  = int(warmup_bars)

        self._orb_date:       date | None = None
        self._orb_high:       float       = 0.0
        self._orb_low:        float       = float("inf")
        self._orb_ready:      bool        = False
        self._trade_done:     bool        = False
        self._orb_stop_count: int         = 0
        self._orb_break_dir:  str         = ""
        self._reverse_eligible: bool      = False

        self._historical_orb_ranges: deque[float] = deque(maxlen=10)
        self._orb_build_vol:      float         = 0.0
        self._historical_build_vols: deque[float] = deque(maxlen=20)

        self._daily_realized_pnl:  float      = 0.0
        self._daily_pnl_date:      date | None = None
        self._daily_circuit_fired: bool        = False

        self._state:       str   = "flat"
        self._entry_price: float = 0.0
        self._stop_price:  float = 0.0
        self._tp_price:    float = 0.0

        self._1m_count = 0
        self._5m_acc:  dict[str, float] | None = None
        self.last_z_score: float = 0.0

    @staticmethod
    def _et_mins(ts: datetime) -> int:
        et_ts = ts.astimezone(_ET)
        return et_ts.hour * 60 + et_ts.minute

    def _reset_session(self, today: date) -> None:
        if self._orb_build_vol > 0:
            self._historical_build_vols.append(self._orb_build_vol)
        self._orb_date         = today
        self._orb_high         = 0.0
        self._orb_low          = float("inf")
        self._orb_ready        = False
        self._trade_done       = False
        self._orb_stop_count   = 0
        self._orb_break_dir    = ""
        self._reverse_eligible = False
        self._orb_build_vol    = 0.0

    def on_bar(self, event: BarEvent) -> None:
        if event.instrument_id != self.instrument:
            return

        # Session reset at ET date boundary (Deterministic)
        today = event.exchange_ts.astimezone(_ET).date()
        if self._orb_date != today:
            self._reset_session(today)

        high  = float(event.high_price)
        low   = float(event.low_price)
        close = float(event.close_price)
        open_ = float(event.open_price)
        vol   = float(event.volume)

        if self._5m_acc is None:
            self._5m_acc = {"o": open_, "h": high, "l": low, "c": close, "v": vol}
        else:
            self._5m_acc["h"] = max(self._5m_acc["h"], high)
            self._5m_acc["l"] = min(self._5m_acc["l"], low)
            self._5m_acc["c"] = close
            self._5m_acc["v"] += vol
        self._1m_count += 1

        mins = self._et_mins(event.exchange_ts)
        if _ORB_BUILD_START <= mins < _ORB_BUILD_END:
            self._orb_high      = max(self._orb_high, high)
            self._orb_low       = min(self._orb_low,  low)
            self._orb_build_vol += vol

        if self._1m_count < 5:
            return

        b = self._5m_acc
        self._5m_acc   = None
        self._1m_count = 0

        if not self._orb_ready and mins >= _ORB_BUILD_END:
            if (self._orb_high > 0.0
                    and self._orb_low < float("inf")
                    and self._orb_high > self._orb_low):
                self._orb_ready = True
                rng = self._orb_high - self._orb_low
                self._historical_orb_ranges.append(rng)
                logger.info(
                    "ORB ready [%s]: high=%.4f low=%.4f range=%.4f",
                    self.instrument, self._orb_high, self._orb_low, rng
                )

        if self._state != "flat":
            self._manage_position(b["c"], event.exchange_ts)
        elif self._orb_ready and not self._trade_done and mins < _ORB_ENTRY_CUTOFF:
            if self._reverse_eligible:
                self._try_reverse_entry(b["c"], event.exchange_ts)
            else:
                self._try_entry(b["c"], event.exchange_ts)

    def _try_entry(self, close: float, ts: datetime) -> None:
        today = ts.astimezone(_ET).date()
        if self._daily_pnl_date != today:
            self._daily_pnl_date      = today
            self._daily_realized_pnl  = 0.0
            self._daily_circuit_fired = False
        if self._daily_realized_pnl < _ORB_DAILY_LOSS_LIMIT:
            if not self._daily_circuit_fired:
                self._daily_circuit_fired = True
            return

        orb_range = self._orb_high - self._orb_low
        if orb_range <= 0.0:
            return

        midpoint = (self._orb_high + self._orb_low) / 2.0
        if orb_range / midpoint > _ORB_MAX_RANGE_PCT:
            return

        if len(self._historical_orb_ranges) < 2:
            return
        avg_hist = sum(self._historical_orb_ranges) / len(self._historical_orb_ranges)
        if avg_hist > 0 and orb_range > avg_hist * _ORB_COMPRESSION_RATIO:
            return

        if len(self._historical_build_vols) < 3:
            return
        avg_vol = sum(self._historical_build_vols) / len(self._historical_build_vols)
        if avg_vol > 0 and self._orb_build_vol < avg_vol * _ORB_MIN_VOL_RATIO:
            return

        if close > self._orb_high:
            self._open("buy",  close, stop=midpoint, target=close + 2.0 * orb_range)
        elif close < self._orb_low:
            self._open("sell", close, stop=midpoint, target=close - 2.0 * orb_range)

    def _open(self, side: str, price: float, stop: float, target: float, notional_override: float | None = None) -> None:
        self._state       = "long" if side == "buy" else "short"
        self._entry_price = price
        self._stop_price  = stop
        self._tp_price    = target
        self._trade_done  = True
        if not self._orb_break_dir:
            self._orb_break_dir = side

        base_notional = notional_override if notional_override is not None else self.leg_notional
        notional = base_notional if side == "buy" else -base_notional
        self.emit_signal(
            instrument_id=self.instrument,
            target_type="notional",
            target_value=notional,
            confidence=0.72,
            stop_model="orb_midpoint",
            metadata={
                "source": "orb",
                "stop": round(stop, 4),
                "target": round(target, 4),
            },
        )

    def _manage_position(self, close: float, ts: datetime) -> None:
        if self._et_mins(ts) >= _ORB_EOD_CLOSE:
            self._close("eod_close", close, ts)
            return

        if self._state == "long":
            if close <= self._stop_price:
                self._close("stop_loss", close, ts)
            elif close >= self._tp_price:
                self._close("take_profit", close, ts)
        else:
            if close >= self._stop_price:
                self._close("stop_loss", close, ts)
            elif close <= self._tp_price:
                self._close("take_profit", close, ts)

    def _close(self, reason: str, price: float, ts: datetime) -> None:
        exit_side = "sell" if self._state == "long" else "buy"
        self.emit_signal(
            instrument_id=self.instrument,
            target_type="notional",
            target_value=0.0,
            confidence=1.0,
            stop_model="orb_exit",
            metadata={"source": "orb", "trigger": reason},
        )
        if self._entry_price > 0:
            direction = 1.0 if self._state == "long" else -1.0
            pnl = direction * (price - self._entry_price) / self._entry_price * self.leg_notional
            self._daily_realized_pnl += pnl

        self._state       = "flat"
        self._entry_price = 0.0
        self._stop_price  = 0.0
        self._tp_price    = 0.0

        if reason == "stop_loss" and self._orb_stop_count == 0:
            self._orb_stop_count += 1
            if self._et_mins(ts) < _ORB_ENTRY_CUTOFF:
                self._trade_done     = False
                self._reverse_eligible = True

    def _try_reverse_entry(self, close: float, ts: datetime) -> None:
        orb_range = self._orb_high - self._orb_low
        if orb_range <= 0.0:
            return
        midpoint = (self._orb_high + self._orb_low) / 2.0
        half = self.leg_notional * 0.5
        if self._orb_break_dir == "buy" and close < self._orb_low:
            self._reverse_eligible = False
            self._open("sell", close, stop=midpoint, target=close - 2.0 * orb_range, notional_override=half)
        elif self._orb_break_dir == "sell" and close > self._orb_high:
            self._reverse_eligible = False
            self._open("buy", close, stop=midpoint, target=close + 2.0 * orb_range, notional_override=half)

    def on_tick(self, event: TradeTick) -> None:
        pass

    def close(self) -> None:
        super().close()
        self._state = "flat"

    @property
    def pair_label(self) -> str:
        return f"{self.instrument}/{self.instrument}"
