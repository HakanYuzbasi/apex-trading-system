"""
quant_system/risk/protector.py

EquityProtector — intraday drawdown kill-switch.

Monitors total portfolio equity against the day's opening equity (recorded
at the first bar event of each UTC calendar day).  When the daily loss
limit is breached it:

  1. Publishes ``side="flatten"`` SignalEvents for **all** open positions —
     both equity and crypto — so the RiskManager converts them to market
     orders immediately.
  2. Sets an internal *halted* flag keyed to today's UTC date.
  3. Auto-saves halt state to a JSON file (if ``state_path`` is provided) so
     a process restart on the same UTC day stays halted.

The halt automatically clears on the next UTC calendar day without any
manual intervention.

Integration
-----------
Pass the instance to ``RiskManager`` (``equity_protector`` kwarg).  The
RiskManager will call ``is_halted()`` before accepting any entry signal and
return None (veto) while the bot is halted, letting only flatten/cover
signals through.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from core.symbols import parse_symbol  # noqa: F401 — imported for future use
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BarEvent
from quant_system.events.signal import SignalEvent
from quant_system.portfolio.ledger import PortfolioLedger

logger = logging.getLogger(__name__)


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EquityProtector:
    """
    Intraday drawdown kill-switch.

    Parameters
    ----------
    portfolio_ledger:
        The live ledger — used read-only to query current equity and
        to discover open positions when emitting flatten signals.
    event_bus:
        The shared event bus.  The protector subscribes to ``"bar"`` events
        and publishes ``"signal"`` events.
    daily_loss_limit:
        Negative fraction at which the kill-switch fires.
        Default: -0.02 (−2 %).  Must be in the range [-1, 0).
    state_path:
        Optional path for JSON persistence.  When provided the halt state
        is automatically written every time it changes so that a restart
        on the same UTC day re-loads the halted flag.
    """

    def __init__(
        self,
        portfolio_ledger: PortfolioLedger,
        event_bus: InMemoryEventBus,
        *,
        daily_loss_limit: float = -0.02,
        state_path: Path | str | None = None,
    ) -> None:
        if not -1.0 <= daily_loss_limit < 0.0:
            raise ValueError(
                f"daily_loss_limit must be a negative fraction in [-1, 0), got {daily_loss_limit}"
            )
        self._ledger = portfolio_ledger
        self._event_bus = event_bus
        self._daily_loss_limit = float(daily_loss_limit)
        self._state_path: Path | None = Path(state_path) if state_path else None

        # Mutable state
        self._day_start_equity: float = 0.0
        self._day_start_date: date | None = None
        self._halted_date: date | None = None

        self._subscription: Subscription = self._event_bus.subscribe(
            "bar", self._on_bar
        )

    # ── Public interface ────────────────────────────────────────────────────

    @property
    def subscription(self) -> Subscription:
        return self._subscription

    @property
    def daily_loss_limit(self) -> float:
        """Kill-switch threshold (e.g. -0.02 for -2 %)."""
        return self._daily_loss_limit

    @property
    def day_start_equity(self) -> float:
        """Equity snapshot taken at the start of the current UTC day."""
        return self._day_start_equity

    def is_halted(self) -> bool:
        """True when the daily loss limit was breached **today** (UTC)."""
        return self._halted_date == _today_utc()

    def daily_drawdown(self) -> float | None:
        """
        Current daily drawdown as a signed fraction (negative = loss).
        Returns None until the day-start equity has been established.
        """
        if self._day_start_equity <= 0.0:
            return None
        return (self._ledger.total_equity() - self._day_start_equity) / self._day_start_equity

    def close(self) -> None:
        """Unsubscribe from the event bus."""
        self._event_bus.unsubscribe(self._subscription.token)

    def manual_flatten(self) -> None:
        """Manually trigger a full portfolio flatten and halt the system for today."""
        today = _today_utc()
        self._halted_date = today
        
        logger.critical("🚨 EMERGENCY FLATTEN INITIATED MANUALLY")
        
        # Create a trigger event context for the flatten logic
        now = _utcnow()
        dummy_event = BarEvent(
            symbol="MANUAL",
            exchange_ts=now,
            processed_ts=now,
            sequence_id=0,
            interval="1m",
            open=0.0, high=0.0, low=0.0, close=0.0, volume=0.0
        )
        self._flatten_all_positions(dummy_event)
        
        if self._state_path is not None:
            self.save_state(self._state_path)

    # ── Bar handler ─────────────────────────────────────────────────────────

    def _on_bar(self, event: BarEvent) -> None:
        today = _today_utc()

        # ── Day roll: record today's opening equity ──────────────────────────
        if self._day_start_date != today:
            self._day_start_equity = self._ledger.total_equity()
            self._day_start_date = today
            logger.info(
                "EquityProtector: new UTC day %s — day_start_equity=$%.2f",
                today,
                self._day_start_equity,
            )

        # ── Already halted for today — nothing more to check ─────────────────
        if self.is_halted():
            return

        dd = self.daily_drawdown()
        if dd is None or dd >= self._daily_loss_limit:
            return  # within acceptable bounds

        # ── Limit breached ───────────────────────────────────────────────────
        self._halted_date = today
        current_equity = self._ledger.total_equity()
        logger.critical(
            "EquityProtector TRIGGERED | daily_drawdown=%.2f%% "
            "(limit=%.2f%%) | equity=$%.2f | day_start=$%.2f — "
            "flattening ALL positions and halting for today",
            dd * 100.0,
            self._daily_loss_limit * 100.0,
            current_equity,
            self._day_start_equity,
        )
        self._flatten_all_positions(event)
        if self._state_path is not None:
            self.save_state(self._state_path)

    # ── Flatten helper ───────────────────────────────────────────────────────

    def _flatten_all_positions(self, trigger_bar: BarEvent) -> None:
        open_positions = [
            instrument_id
            for instrument_id, position in self._ledger.positions.items()
            if abs(position.quantity) > 1e-12
        ]
        if not open_positions:
            logger.info("EquityProtector: no open positions to flatten")
            return

        dd = self.daily_drawdown() or 0.0
        now = _utcnow()

        for instrument_id in open_positions:
            received_ts = max(trigger_bar.exchange_ts, now)
            processed_ts = max(received_ts, trigger_bar.processed_ts, now)
            signal = SignalEvent(
                instrument_id=instrument_id,
                exchange_ts=trigger_bar.exchange_ts,
                received_ts=received_ts,
                processed_ts=processed_ts,
                sequence_id=trigger_bar.sequence_id,
                source="risk.equity_protector",
                strategy_id="EquityProtector",
                side="flatten",
                target_type="notional",
                target_value=0.0,
                confidence=1.0,
                stop_model="equity_protection",
                stop_params={"daily_drawdown": round(dd, 6)},
            )
            self._dispatch_signal(signal)
            logger.info(
                "EquityProtector: FLATTEN signal published for %s", instrument_id
            )

    def _dispatch_signal(self, event: SignalEvent) -> None:
        """Publish via sync or async path depending on existing subscribers."""
        subs = self._event_bus.subscriptions_for(event.event_type)
        has_async = any(s.is_async for s in subs)
        if not has_async:
            self._event_bus.publish(event)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._event_bus.publish_async(event))
            return
        loop.create_task(self._event_bus.publish_async(event))

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_state(self, path: Path | str) -> None:
        """
        Write halt state to *path* (JSON).

        Saved fields
        ------------
        * ``halted_date``      — ISO date if halted, else null.
        * ``day_start_equity`` — equity at the start of the last UTC day.
        * ``day_start_date``   — ISO date the snapshot was taken.

        On startup call ``load_state`` before the first bar arrives so the
        bot remains halted if the process was restarted mid-day after the
        kill-switch fired.
        """
        payload = {
            "halted_date": self._halted_date.isoformat() if self._halted_date else None,
            "day_start_equity": self._day_start_equity,
            "day_start_date": self._day_start_date.isoformat() if self._day_start_date else None,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("EquityProtector state saved to %s", path)

    def load_state(self, path: Path | str) -> None:
        """
        Restore halt state from *path*.  Silently no-ops if the file does not
        exist.  If today's date matches the persisted ``halted_date`` the bot
        starts up in halted mode.
        """
        p = Path(path)
        if not p.exists():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))

            raw_halt = payload.get("halted_date")
            self._halted_date = date.fromisoformat(raw_halt) if raw_halt else None

            self._day_start_equity = float(payload.get("day_start_equity", 0.0))

            raw_start = payload.get("day_start_date")
            self._day_start_date = date.fromisoformat(raw_start) if raw_start else None

            if self.is_halted():
                logger.warning(
                    "EquityProtector: loaded state from %s — bot is HALTED for today (%s)",
                    path,
                    _today_utc(),
                )
            elif self._halted_date is not None and self._halted_date < _today_utc():
                # Stale halt from a previous UTC session — auto-clear so it
                # doesn't carry over and silently block today's entries.
                logger.info(
                    "EquityProtector: stale halt date %s is before today %s — auto-clearing.",
                    self._halted_date, _today_utc(),
                )
                self._halted_date = None
                if self._state_path is not None:
                    self.save_state(self._state_path)
            else:
                logger.info(
                    "EquityProtector: state restored from %s (not halted)", path
                )
        except Exception:
            logger.exception("EquityProtector: failed to load state from %s", path)

    # ── Manual control ───────────────────────────────────────────────────────

    def reset_halt(self) -> None:
        """
        Clear today's halt flag after manual review confirms the system is
        healthy.  Persists the cleared state so a restart stays unhalted.
        """
        if not self.is_halted():
            logger.debug("EquityProtector.reset_halt() called but bot is not halted — no-op.")
            return
        self._halted_date = None
        if self._state_path is not None:
            self.save_state(self._state_path)
        logger.warning(
            "EquityProtector: halt manually cleared — entries re-enabled for today."
        )

    def reseed_day_equity(self, sovereign_equity: float) -> None:
        """
        Override ``day_start_equity`` with the broker-verified equity truth.

        Call this after the Master Sync block during harness startup, passing
        the confirmed Alpaca equity value.  Prevents phantom baselines from
        IBKR multi-venue seeding from poisoning the drawdown calculation.

        A reseed is skipped if ``sovereign_equity <= 0`` or if it is more than
        10× the current baseline (would indicate a data error).
        """
        if sovereign_equity <= 0:
            logger.warning(
                "EquityProtector.reseed_day_equity: sovereign_equity=%.2f is invalid — skipping.",
                sovereign_equity,
            )
            return
        if self._day_start_equity > 0 and sovereign_equity > self._day_start_equity * 10:
            logger.warning(
                "EquityProtector.reseed_day_equity: sovereign_equity=$%.2f is >10× current "
                "baseline=$%.2f — looks like a data error, skipping.",
                sovereign_equity, self._day_start_equity,
            )
            return
        old = self._day_start_equity
        self._day_start_equity = sovereign_equity
        logger.info(
            "EquityProtector: day_start_equity reseeded $%.2f → $%.2f (Alpaca sovereign truth).",
            old, sovereign_equity,
        )
