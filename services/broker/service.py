import os
import json
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from datetime import datetime

from cryptography.fernet import Fernet
from alpaca.trading.client import TradingClient
from ib_insync import IB

from models.broker import BrokerConnection, BrokerType
from core.exceptions import ApexBrokerError
from services.common.db import db_session
from services.trading.models import BrokerConnectionModel

logger = logging.getLogger(__name__)


@dataclass
class BrokerEquitySnapshot:
    """Normalized equity payload used by API/dashboard surfaces."""

    value: float
    broker: str
    stale: bool
    as_of: str
    source: str
    source_id: str


class BrokerService:
    def __init__(self):
        self._master_key = os.getenv("APEX_MASTER_KEY")
        if not self._master_key:
            logger.warning(
                "APEX_MASTER_KEY not set. Generating temporary key. "
                "DATA WILL BE LOST ON RESTART unless you set this env variable."
            )
            self._master_key = Fernet.generate_key().decode()
        key_bytes = self._master_key.encode() if isinstance(self._master_key, str) else self._master_key
        self._cipher = Fernet(key_bytes)

        # In-memory cache; populated lazily per user_id from disk
        self._connections: Dict[str, BrokerConnection] = {}
        self._loaded_users: set[str] = set()
        self._equity_cache: Dict[str, BrokerEquitySnapshot] = {}

    # ─────────────────────────────────────────
    # Disk persistence helpers
    # ─────────────────────────────────────────

    async def _load_user(self, user_id: str) -> None:
        """Loads broker connections for a user from Postgres into the in-memory cache."""
        if user_id in self._loaded_users:
            return
        self._loaded_users.add(user_id)

        try:
            async with db_session() as session:
                from sqlalchemy import select
                stmt = select(BrokerConnectionModel).filter(BrokerConnectionModel.user_id == user_id)
                result = await session.execute(stmt)
                records = result.scalars().all()

                for row in records:
                    conn = BrokerConnection(
                        id=row.id,
                        user_id=row.user_id,
                        broker_type=BrokerType(row.broker_type),
                        name=row.name,
                        environment=row.environment,
                        client_id=row.client_id,
                        credentials={"data": row.credentials_encrypted_json},
                        is_active=row.is_active,
                        created_at=row.created_at,
                        updated_at=row.updated_at
                    )
                    self._connections[conn.id] = conn
                logger.info(f"Loaded {len(records)} broker connection(s) for user '{user_id}' from database.")
        except Exception as e:  # SWALLOW: startup should continue even if one tenant load fails
            logger.exception("Failed to load broker connections for user '%s': %s", user_id, e)

    async def _save_user(self, user_id: str) -> None:
        """Atomically upserts all broker connections for a user to Postgres."""
        connections = [c for c in self._connections.values() if c.user_id == user_id]

        try:
            async with db_session() as session:
                # Merge each connection
                for conn in connections:
                    db_model = BrokerConnectionModel(
                        id=conn.id,
                        user_id=conn.user_id,
                        broker_type=conn.broker_type.value,
                        name=conn.name,
                        environment=conn.environment,
                        client_id=conn.client_id,
                        credentials_encrypted_json=conn.credentials.get("data", ""),
                        is_active=conn.is_active,
                        created_at=conn.created_at,
                        updated_at=conn.updated_at
                    )
                    await session.merge(db_model)
                await session.commit()
            logger.debug(f"Saved {len(connections)} broker connection(s) for user '{user_id}' to database.")
        except Exception as e:  # SWALLOW: persist failure is logged; caller receives in-memory state
            logger.exception("Failed to save broker connections for user '%s': %s", user_id, e)

    # ─────────────────────────────────────────
    # Credential encryption helpers
    # ─────────────────────────────────────────

    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        json_str = json.dumps(credentials)
        ciphertext = self._cipher.encrypt(json_str.encode()).decode()
        return f"v1:{ciphertext}"

    def _decrypt_credentials(self, encrypted_credentials: str) -> Dict[str, Any]:
        try:
            if encrypted_credentials.startswith("v1:"):
                payload = encrypted_credentials[3:]
            else:
                payload = encrypted_credentials # Legacy unversioned
                
            params = self._cipher.decrypt(payload.encode()).decode()
            return json.loads(params)
        except Exception as exc:
            logger.error("Credential decryption failed (key mismatch or corrupted data): %s", exc)
            raise ApexBrokerError(
                code="CREDENTIAL_DECRYPT_FAILED",
                message="Failed to decrypt stored credentials — was APEX_MASTER_KEY changed?",
                context={"error": str(exc)},
            ) from exc

    # ─────────────────────────────────────────
    # Connection validation
    # ─────────────────────────────────────────

    async def connect_alpaca(self, api_key: str, secret_key: str, environment: str = "paper") -> TradingClient:
        """Establishes and validates an Alpaca connection."""
        try:
            client = TradingClient(api_key, secret_key, paper=(environment == "paper"))
            client.get_account()  # Validates credentials immediately
            return client
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            raise ApexBrokerError(
                code="ALPACA_CONNECT_FAILED",
                message="Failed to connect to Alpaca",
                context={"environment": environment, "error": str(e)},
            ) from e

    async def connect_ibkr(self, host: str, port: int, client_id: int) -> IB:
        """Establishes and validates an IBKR connection."""
        ib = IB()
        try:
            await ib.connectAsync(host, port, clientId=client_id, timeout=5)
            if not ib.isConnected():
                raise ConnectionError("IBKR connection check failed")
            ib.disconnect()
            return ib
        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            raise ApexBrokerError(
                code="IBKR_CONNECT_FAILED",
                message="Failed to connect to IBKR",
                context={"host": host, "port": port, "client_id": client_id, "error": str(e)},
            ) from e

    async def validate_credentials(self, broker_type: BrokerType, credentials: Dict[str, Any], environment: str = "paper") -> bool:
        """Validates credentials by attempting a connection."""
        if broker_type == BrokerType.ALPACA:
            await self.connect_alpaca(
                credentials.get("api_key"),
                credentials.get("secret_key"),
                environment,
            )
        elif broker_type == BrokerType.IBKR:
            await self.connect_ibkr(
                credentials.get("host", "127.0.0.1"),
                credentials.get("port", 7497),
                credentials.get("client_id"),
            )
        else:
            raise ApexBrokerError(
                code="BROKER_CREDENTIAL_VALIDATION_UNSUPPORTED",
                message=f"Unsupported broker type: {broker_type}",
                context={"broker_type": str(broker_type)},
            )
        return True

    # ─────────────────────────────────────────
    # CRUD operations (each mutating op saves to disk)
    # ─────────────────────────────────────────

    async def create_connection(
        self,
        user_id: str,
        broker_type: BrokerType,
        name: str,
        credentials: Dict[str, Any],
        environment: str = "paper",
        client_id: Optional[int] = None,
    ) -> BrokerConnection:
        """Creates and saves a new broker connection after validation."""
        # Ensure existing connections for this user are loaded first
        await self._load_user(user_id)

        # 1. Validate credentials against the broker
        await self.validate_credentials(broker_type, credentials, environment)

        # 2. Encrypt credentials before storing
        encrypted_creds = {"data": self._encrypt_credentials(credentials)}

        # 3. Build model
        connection = BrokerConnection(
            user_id=user_id,
            broker_type=broker_type,
            name=name,
            environment=environment,
            client_id=client_id,
            credentials=encrypted_creds,
            is_active=True,
        )

        # 4. Write to in-memory cache and persist
        self._connections[connection.id] = connection
        await self._save_user(user_id)
        return connection

    async def get_connection(self, connection_id: str, user_id: str) -> Optional[BrokerConnection]:
        await self._load_user(user_id)
        conn = self._connections.get(connection_id)
        if conn and conn.user_id == user_id:
            return conn
        return None

    async def list_connections(self, user_id: str) -> List[BrokerConnection]:
        await self._load_user(user_id)
        return [c for c in self._connections.values() if c.user_id == user_id]

    async def delete_connection(self, connection_id: str, user_id: str) -> bool:
        conn = await self.get_connection(connection_id, user_id)
        if not conn:
            return False
        
        # Delete from Postgres
        try:
            async with db_session() as session:
                from sqlalchemy import delete
                stmt = delete(BrokerConnectionModel).where(BrokerConnectionModel.id == connection_id)
                await session.execute(stmt)
                await session.commit()
        except Exception as e:  # SWALLOW: delete failure returns False to endpoint for 500 mapping
            logger.exception("Failed to delete broker connection %s from DB: %s", connection_id, e)
            return False

        del self._connections[connection_id]
        return True

    async def toggle_connection(self, connection_id: str, user_id: str) -> Optional[BrokerConnection]:
        """Toggles is_active and persists the change."""
        conn = await self.get_connection(connection_id, user_id)
        if not conn:
            return None
        conn.is_active = not conn.is_active
        conn.updated_at = datetime.utcnow()
        await self._save_user(conn.user_id)
        return conn

    # ─────────────────────────────────────────
    # Portfolio aggregation
    # ─────────────────────────────────────────

    async def get_total_equity(self, user_id: str) -> float:
        """Aggregates equity across all active connections."""
        snapshot = await self.get_tenant_equity_snapshot(user_id)
        return float(snapshot["total_equity"])

    async def get_positions(self, user_id: str) -> List[Dict[str, Any]]:
        """Aggregates open positions across all active broker connections."""
        connections = await self.list_connections(user_id)
        all_positions: List[Dict[str, Any]] = []

        for conn in connections:
            if not conn.is_active:
                continue
            try:
                creds = self._decrypt_credentials(conn.credentials["data"])
                if conn.broker_type == BrokerType.ALPACA:
                    client = TradingClient(
                        creds["api_key"],
                        creds["secret_key"],
                        paper=(conn.environment == "paper"),
                    )
                    raw_positions = client.get_all_positions()
                    for pos in raw_positions:
                        all_positions.append({
                            "source": conn.name,
                            "source_id": conn.id,
                            "broker_type": conn.broker_type.value,
                            "symbol": str(pos.symbol),
                            "qty": float(pos.qty),
                            "side": str(pos.side.value) if hasattr(pos.side, "value") else str(pos.side),
                            "avg_cost": float(pos.avg_entry_price or 0),
                            "current_price": float(pos.current_price or 0),
                            "market_value": float(pos.market_value or 0),
                            "unrealized_pl": float(pos.unrealized_pl or 0),
                            "unrealized_plpc": float(pos.unrealized_plpc or 0),
                            "stale": False,
                            "as_of": datetime.utcnow().isoformat(),
                            "source_status": "ok",
                        })
                elif conn.broker_type == BrokerType.IBKR:
                    logger.debug(f"IBKR positions not yet supported for connection {conn.id}")
            except Exception as e:  # SWALLOW: isolate one broker failure from global positions view
                logger.exception("Failed to fetch positions for connection %s (%s): %s", conn.id, conn.name, e)

        return all_positions

    async def list_connection_sources(self, user_id: str) -> List[Dict[str, Any]]:
        """Returns lightweight list of active connections for the account selector dropdown."""
        connections = await self.list_connections(user_id)
        return [
            {"id": c.id, "name": c.name, "broker_type": c.broker_type.value, "environment": c.environment}
            for c in connections
            if c.is_active
        ]

    async def list_tenant_ids(self) -> List[str]:
        """Return distinct tenant IDs with active broker connections."""
        async with db_session() as session:
            from sqlalchemy import distinct, select

            result = await session.execute(
                select(distinct(BrokerConnectionModel.user_id)).where(
                    BrokerConnectionModel.is_active == True
                )
            )
            return [row[0] for row in result.all() if row[0]]

    async def get_tenant_equity_snapshot(self, tenant_id: str) -> Dict[str, Any]:
        """Aggregate tenant equity with per-source normalized snapshots."""
        snapshots = await self._collect_equity_snapshots_for_user(tenant_id)
        total_value = sum(snapshot.value for snapshot in snapshots)
        return {
            "tenant_id": tenant_id,
            "total_equity": float(total_value),
            "breakdown": [snapshot.__dict__ for snapshot in snapshots],
            "as_of": datetime.utcnow().isoformat(),
        }

    async def _collect_equity_snapshots_for_user(self, user_id: str) -> List[BrokerEquitySnapshot]:
        """Collect per-connection equity snapshots with error isolation."""
        connections = await self.list_connections(user_id)
        active_connections = [conn for conn in connections if conn.is_active]

        tasks = [
            self._fetch_equity_with_fallback(connection=connection)
            for connection in active_connections
        ]
        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: List[BrokerEquitySnapshot] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                connection = active_connections[index]
                logger.error(
                    "Equity collection failed for %s (%s): %s",
                    connection.id,
                    connection.name,
                    result,
                )
                continue
            snapshots.append(result)
        return snapshots

    async def _fetch_equity_with_fallback(self, connection: BrokerConnection) -> BrokerEquitySnapshot:
        """Fetch connection equity and fall back to last known value on failures."""
        try:
            raw_creds = connection.credentials.get("data", "")
            credentials = self._decrypt_credentials(raw_creds)
            snapshot = await self._fetch_connection_equity(connection, credentials)
            self._equity_cache[connection.id] = snapshot
            return snapshot
        except Exception as exc:  # SWALLOW: use last-known cached equity snapshot when broker is unavailable
            cached = self._equity_cache.get(connection.id)
            if cached is not None:
                return BrokerEquitySnapshot(
                    value=cached.value,
                    broker=cached.broker,
                    stale=True,
                    as_of=cached.as_of,
                    source=cached.source,
                    source_id=cached.source_id,
                )
            raise ApexBrokerError(
                code="BROKER_EQUITY_FETCH_FAILED",
                message=f"Unable to fetch equity for connection {connection.id}",
                context={"connection_id": connection.id, "broker_type": connection.broker_type.value, "error": str(exc)},
            ) from exc

    _EQUITY_FETCH_TIMEOUT_SECONDS = 10

    async def _fetch_connection_equity(
        self,
        connection: BrokerConnection,
        credentials: Dict[str, Any],
    ) -> BrokerEquitySnapshot:
        """Fetch normalized equity snapshot for a single broker connection."""
        if connection.broker_type == BrokerType.ALPACA:
            equity = await asyncio.wait_for(
                asyncio.to_thread(
                    self._fetch_alpaca_equity_blocking,
                    credentials,
                    connection.environment,
                ),
                timeout=self._EQUITY_FETCH_TIMEOUT_SECONDS,
            )
            return BrokerEquitySnapshot(
                value=float(equity),
                broker="alpaca",
                stale=False,
                as_of=datetime.utcnow().isoformat(),
                source=connection.name,
                source_id=connection.id,
            )

        if connection.broker_type == BrokerType.IBKR:
            equity = await asyncio.wait_for(
                asyncio.to_thread(
                    self._fetch_ibkr_equity_blocking,
                    credentials,
                    connection.client_id,
                ),
                timeout=self._EQUITY_FETCH_TIMEOUT_SECONDS,
            )
            return BrokerEquitySnapshot(
                value=float(equity),
                broker="ibkr",
                stale=False,
                as_of=datetime.utcnow().isoformat(),
                source=connection.name,
                source_id=connection.id,
            )

        raise ApexBrokerError(
            code="BROKER_UNSUPPORTED",
            message=f"Unsupported broker type: {connection.broker_type}",
            context={"broker_type": str(connection.broker_type)},
        )

    @staticmethod
    def _fetch_alpaca_equity_blocking(credentials: Dict[str, Any], environment: str) -> float:
        """Blocking Alpaca account equity call (run in worker thread)."""
        client = TradingClient(
            credentials["api_key"],
            credentials["secret_key"],
            paper=(environment == "paper"),
        )
        account = client.get_account()
        return float(account.equity)

    @staticmethod
    def _fetch_ibkr_equity_blocking(credentials: Dict[str, Any], client_id: Optional[int]) -> float:
        """Blocking IBKR equity fetch using NetLiquidation summary tag."""
        host = credentials.get("host", "127.0.0.1")
        port = int(credentials.get("port", 7497))
        ib_client_id = int(credentials.get("client_id") or client_id or 1)
        owned_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            # ib_insync resolves the current loop during connect(); worker threads
            # have no default loop in Python 3.11+, so create one explicitly.
            owned_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(owned_loop)
        ib = IB()
        try:
            ib.connect(host, port, clientId=ib_client_id, timeout=5, readonly=True)
            summary_rows = ib.accountSummary()
            for row in summary_rows:
                if getattr(row, "tag", "") == "NetLiquidation":
                    return float(row.value)
            raise ApexBrokerError(
                code="IBKR_EQUITY_MISSING",
                message="NetLiquidation tag not found in IBKR account summary",
                context={"host": host, "port": port, "client_id": ib_client_id},
            )
        finally:
            if ib.isConnected():
                ib.disconnect()
            if owned_loop is not None:
                asyncio.set_event_loop(None)
                owned_loop.close()


# Singleton instance
broker_service = BrokerService()
