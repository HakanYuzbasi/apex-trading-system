from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from psycopg2.extras import Json, RealDictCursor

from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.data.stores.client import TimescaleDBClient
from quant_system.events import BaseEvent, ExecutionEvent
from quant_system.portfolio.ledger import PortfolioLedger, Position

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PortfolioStateSnapshot:
    state_ts: datetime
    cash: float
    positions: dict[str, dict[str, float]]
    last_prices: dict[str, float]
    metadata: dict[str, Any]


class StateManager:
    """
    Persist and recover ledger state snapshots from TimescaleDB.
    """

    def __init__(
        self,
        portfolio_ledger: PortfolioLedger,
        event_bus: InMemoryEventBus,
        db_client: TimescaleDBClient,
    ) -> None:
        self._portfolio_ledger = portfolio_ledger
        self._event_bus = event_bus
        self._db_client = db_client
        self._subscription: Subscription = self._event_bus.subscribe("execution", self._on_execution, is_async=True)

    @property
    def subscription(self) -> Subscription:
        return self._subscription

    def ensure_schema(self) -> None:
        with self._db_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_states (
                        state_ts TIMESTAMPTZ NOT NULL,
                        cash DOUBLE PRECISION NOT NULL,
                        positions JSONB NOT NULL,
                        last_prices JSONB NOT NULL DEFAULT '{}'::jsonb,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        PRIMARY KEY (state_ts)
                    );
                    """
                )
                cur.execute(
                    """
                    SELECT create_hypertable(
                        'portfolio_states',
                        'state_ts',
                        if_not_exists => TRUE,
                        migrate_data => TRUE,
                        chunk_time_interval => '7 days'::interval
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_portfolio_states_desc ON portfolio_states (state_ts DESC);"
                )

    async def close(self) -> None:
        self._event_bus.unsubscribe(self._subscription.token)

    async def _on_execution(self, event: BaseEvent) -> None:
        if not isinstance(event, ExecutionEvent):
            return
        if event.execution_status not in {"partial_fill", "filled"}:
            return
        await asyncio.to_thread(self.save_state, trigger_event=event)

    def save_state(self, *, trigger_event: ExecutionEvent | None = None) -> None:
        snapshot = self._snapshot(trigger_event=trigger_event)
        with self._db_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO portfolio_states (state_ts, cash, positions, last_prices, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        snapshot.state_ts,
                        snapshot.cash,
                        Json(snapshot.positions),
                        Json(snapshot.last_prices),
                        Json(snapshot.metadata),
                    ),
                )

    def load_latest_state(self) -> PortfolioStateSnapshot | None:
        with self._db_client.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT state_ts, cash, positions, last_prices, metadata
                    FROM portfolio_states
                    ORDER BY state_ts DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
        if row is None:
            return None
        return PortfolioStateSnapshot(
            state_ts=self._normalize_ts(row["state_ts"]),
            cash=float(row["cash"]),
            positions={str(key): dict(value) for key, value in (row["positions"] or {}).items()},
            last_prices={str(key): float(value) for key, value in (row.get("last_prices") or {}).items()},
            metadata=dict(row.get("metadata") or {}),
        )

    def restore_into_ledger(self, snapshot: PortfolioStateSnapshot) -> None:
        self._portfolio_ledger.cash = float(snapshot.cash)
        self._portfolio_ledger.positions.clear()
        self._portfolio_ledger.last_price_by_instrument.clear()
        for instrument_id, payload in snapshot.positions.items():
            self._portfolio_ledger.positions[instrument_id] = Position(
                quantity=float(payload.get("quantity", 0.0)),
                avg_price=float(payload.get("avg_price", 0.0)),
                realized_pnl=float(payload.get("realized_pnl", 0.0)),
            )
        self._portfolio_ledger.last_price_by_instrument.update(snapshot.last_prices)

    def _snapshot(self, *, trigger_event: ExecutionEvent | None) -> PortfolioStateSnapshot:
        now = datetime.now(timezone.utc)
        positions = {
            instrument_id: {
                "quantity": float(position.quantity),
                "avg_price": float(position.avg_price),
                "realized_pnl": float(position.realized_pnl),
            }
            for instrument_id, position in self._portfolio_ledger.positions.items()
            if abs(position.quantity) > 1e-12 or abs(position.realized_pnl) > 1e-12
        }
        metadata: dict[str, Any] = {
            "total_equity": self._portfolio_ledger.total_equity(),
            "total_realized_pnl": self._portfolio_ledger.total_realized_pnl(),
            "total_unrealized_pnl": self._portfolio_ledger.total_unrealized_pnl(),
        }
        if trigger_event is not None:
            metadata["trigger_execution_id"] = trigger_event.execution_id
            metadata["trigger_order_id"] = trigger_event.order_id

        return PortfolioStateSnapshot(
            state_ts=now,
            cash=float(self._portfolio_ledger.cash),
            positions=positions,
            last_prices={key: float(value) for key, value in self._portfolio_ledger.last_price_by_instrument.items()},
            metadata=metadata,
        )

    @staticmethod
    def _normalize_ts(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
