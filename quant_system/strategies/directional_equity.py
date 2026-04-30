"""
quant_system/strategies/directional_equity.py

DirectionalEquityStrategy — single-name long/short equity trading.

Signal logic
------------
Entry (flat → long):  RSI(14) < 35  AND  close > EMA(20)   [oversold bounce in uptrend]
Entry (flat → short): RSI(14) > 65  AND  close < EMA(20)   [overbought fade in downtrend]

Exit:
  - RSI reverts through 50 (mean-reversion complete)
  - Stop-loss:    1.5 × ATR(14) against entry
  - Take-profit:  3.0 × ATR(14) in favour of entry

Fits seamlessly into the harness StrategySlotConfig by setting
instrument_b == instrument_a (single-symbol slot convention).
All existing risk gates (FactorMonitor, RiskManager, EquityProtector)
apply automatically via the shared event bus.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent
from quant_system.events.signal import SignalEvent

_ET = ZoneInfo("America/New_York")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast technical indicators (no pandas dependency)
# ---------------------------------------------------------------------------

def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(closes: list[float], period: int) -> float:
    if len(closes) < 2:
        return closes[0] if closes else float("nan")
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
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    return float(np.mean(trs[-period:])) if trs else 0.0


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class DirectionalEquityStrategy:
    """
    Single-name directional equity strategy (RSI mean-reversion + EMA trend filter).

    Designed as a drop-in alongside KalmanPairsStrategy inside the v3 harness.
    Use ``instrument_b == instrument_a`` in the slot config to signal single-name mode.
    """

    RSI_PERIOD       = 14
    EMA_PERIOD       = 20
    ATR_PERIOD       = 14
    RSI_OVERSOLD     = 40.0   # loosened from 35 — fires 3-5× more often on large-caps
    RSI_OVERBOUGHT   = 60.0   # loosened from 65
    RSI_EXIT         = 50.0
    STOP_ATR_MULT    = 1.5
    TP_ATR_MULT      = 3.0
    MAX_HISTORY      = 120   # bars to keep in rolling buffer

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        instrument: str,
        leg_notional: float = 2_000.0,
        warmup_bars: int = 10,
    ) -> None:
        self._bus          = event_bus
        self.instrument    = instrument
        self.leg_notional  = float(leg_notional)
        self.warmup_bars   = int(warmup_bars)

        self._closes: deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._highs:  deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._lows:   deque[float] = deque(maxlen=self.MAX_HISTORY)
        self._bar_count = 0

        # State machine: "flat" | "long" | "short"
        self._state:       str   = "flat"
        self._entry_price: float = 0.0
        self._stop_price:  float = 0.0
        self._tp_price:    float = 0.0

        # Exposed for telemetry (mirrors KalmanPairsStrategy.last_z_score convention)
        self.last_z_score: float = 0.0

        self._sub = self._bus.subscribe("bar", self._on_bar)
        logger.info(
            "DirectionalEquity: watching %s | notional=$%.0f | warmup=%d bars",
            self.instrument, self.leg_notional, self.warmup_bars,
        )

    # ------------------------------------------------------------------
    # Bar handler
    # ------------------------------------------------------------------

    def _on_bar(self, event: BarEvent) -> None:
        if event.instrument_id != self.instrument:
            return
        close = float(event.close_price)
        high  = float(event.high_price)
        low   = float(event.low_price)

        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)
        self._bar_count += 1

        if self._bar_count < self.warmup_bars:
            return

        closes = list(self._closes)
        highs  = list(self._highs)
        lows   = list(self._lows)

        rsi   = _rsi(closes, self.RSI_PERIOD)
        ema20 = _ema(closes, self.EMA_PERIOD)
        atr   = _atr(highs, lows, closes, self.ATR_PERIOD)

        # Normalised RSI deviation used as telemetry z-score proxy
        self.last_z_score = (rsi - 50.0) / 15.0

        if self._state == "flat":
            self._try_entry(close, rsi, ema20, atr)
        else:
            self._manage_position(close, rsi)

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _is_equity_auction_window(self) -> bool:
        """Skip entries during the opening/closing 15-min auction windows (equities only)."""
        if self.instrument.startswith("CRYPTO:") or self.instrument.startswith("FOREX:"):
            return False
        now = datetime.now(_ET)
        mins = now.hour * 60 + now.minute
        return (9 * 60 + 30 <= mins < 9 * 60 + 45) or (15 * 60 + 45 <= mins < 16 * 60)

    def _try_entry(self, close: float, rsi: float, ema20: float, atr: float) -> None:
        if atr <= 0.0:
            return
        if self._is_equity_auction_window():
            return
        if rsi < self.RSI_OVERSOLD and close > ema20:
            self._open("buy", close, atr)
        elif rsi > self.RSI_OVERBOUGHT and close < ema20:
            self._open("sell", close, atr)

    def _open(self, side: str, price: float, atr: float) -> None:
        self._state       = "long" if side == "buy" else "short"
        self._entry_price = price
        if self._state == "long":
            self._stop_price = price - self.STOP_ATR_MULT * atr
            self._tp_price   = price + self.TP_ATR_MULT   * atr
        else:
            self._stop_price = price + self.STOP_ATR_MULT * atr
            self._tp_price   = price - self.TP_ATR_MULT   * atr

        target = self.leg_notional if side == "buy" else -self.leg_notional
        self._emit_signal(side, target, metadata={"trigger": "rsi_entry"})
        logger.info(
            "DirectionalEquity ENTRY %s %s @ %.4f | stop=%.4f | tp=%.4f | notional=$%.0f",
            side.upper(), self.instrument, price,
            self._stop_price, self._tp_price, abs(target),
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _manage_position(self, close: float, rsi: float) -> None:
        if self._state == "long":
            if rsi >= self.RSI_EXIT:
                self._close_position(close, "rsi_exit")
            elif close <= self._stop_price:
                self._close_position(close, "stop_loss")
            elif close >= self._tp_price:
                self._close_position(close, "take_profit")
        else:  # short
            if rsi <= self.RSI_EXIT:
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
        self._state       = "flat"
        self._entry_price = 0.0
        self._stop_price  = 0.0
        self._tp_price    = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_signal(self, side: str, target_value: float, metadata: dict[str, Any] | None = None) -> None:
        self._bus.publish(SignalEvent(
            instrument_id=self.instrument,
            side=side,
            target_value=target_value,
            confidence=0.75,
            metadata={**(metadata or {}), "source": "directional_equity"},
        ))

    def close(self) -> None:
        """Called by the harness when the slot is deactivated."""
        self._bus.unsubscribe(self._sub)
        if self._state != "flat":
            exit_side = "sell" if self._state == "long" else "buy"
            self._emit_signal(exit_side, 0.0, metadata={"trigger": "slot_closed"})
            self._state = "flat"
