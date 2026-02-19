import os
import json
import base64
import tempfile
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

from cryptography.fernet import Fernet
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError
from ib_insync import IB

from models.broker import BrokerConnection, BrokerType

logger = logging.getLogger(__name__)

# Root directory for user data  (relative to CWD when backend starts)
DATA_ROOT = Path(os.getenv("APEX_DATA_DIR", "data"))


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

    # ─────────────────────────────────────────
    # Disk persistence helpers
    # ─────────────────────────────────────────

    def _user_store_path(self, user_id: str) -> Path:
        """Returns the JSON file path for a given user's broker connections."""
        path = DATA_ROOT / "users" / user_id
        path.mkdir(parents=True, exist_ok=True)
        return path / "brokers.json"

    def _load_user(self, user_id: str) -> None:
        """Loads broker connections for a user from disk into the in-memory cache (idempotent)."""
        if user_id in self._loaded_users:
            return
        self._loaded_users.add(user_id)

        store = self._user_store_path(user_id)
        if not store.exists():
            return

        try:
            raw = store.read_text(encoding="utf-8")
            records: List[dict] = json.loads(raw)
            for rec in records:
                # Deserialise ISO datetime strings back to datetime objects
                for dt_field in ("created_at", "updated_at"):
                    if dt_field in rec and isinstance(rec[dt_field], str):
                        rec[dt_field] = datetime.fromisoformat(rec[dt_field])
                conn = BrokerConnection.model_validate(rec)
                self._connections[conn.id] = conn
            logger.info(f"Loaded {len(records)} broker connection(s) for user '{user_id}' from disk.")
        except Exception as e:
            logger.error(f"Failed to load broker connections for user '{user_id}': {e}")

    def _save_user(self, user_id: str) -> None:
        """Atomically writes all broker connections for a user to disk."""
        store = self._user_store_path(user_id)
        connections = [c for c in self._connections.values() if c.user_id == user_id]

        try:
            records = []
            for conn in connections:
                rec = conn.model_dump()
                # Serialise datetime objects to ISO strings for JSON
                for dt_field in ("created_at", "updated_at"):
                    if isinstance(rec.get(dt_field), datetime):
                        rec[dt_field] = rec[dt_field].isoformat()
                records.append(rec)

            # Atomic write: write to temp file then rename to avoid partial writes
            tmp_fd, tmp_path = tempfile.mkstemp(dir=store.parent, suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
                os.replace(tmp_path, store)
            except Exception:
                os.unlink(tmp_path)
                raise

            logger.debug(f"Saved {len(records)} broker connection(s) for user '{user_id}' to {store}.")
        except Exception as e:
            logger.error(f"Failed to save broker connections for user '{user_id}': {e}")

    # ─────────────────────────────────────────
    # Credential encryption helpers
    # ─────────────────────────────────────────

    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> str:
        json_str = json.dumps(credentials)
        return self._cipher.encrypt(json_str.encode()).decode()

    def _decrypt_credentials(self, encrypted_credentials: str) -> Dict[str, Any]:
        params = self._cipher.decrypt(encrypted_credentials.encode()).decode()
        return json.loads(params)

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
            raise ValueError(f"Failed to connect to Alpaca: {e}")

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
            raise ValueError(f"Failed to connect to IBKR at {host}:{port} (Client ID: {client_id}): {e}")

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
        self._load_user(user_id)

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
        self._save_user(user_id)
        return connection

    async def get_connection(self, connection_id: str) -> Optional[BrokerConnection]:
        return self._connections.get(connection_id)

    async def list_connections(self, user_id: str) -> List[BrokerConnection]:
        self._load_user(user_id)
        return [c for c in self._connections.values() if c.user_id == user_id]

    async def delete_connection(self, connection_id: str) -> bool:
        conn = self._connections.get(connection_id)
        if not conn:
            return False
        del self._connections[connection_id]
        self._save_user(conn.user_id)
        return True

    async def toggle_connection(self, connection_id: str) -> Optional[BrokerConnection]:
        """Toggles is_active and persists the change."""
        conn = self._connections.get(connection_id)
        if not conn:
            return None
        conn.is_active = not conn.is_active
        conn.updated_at = datetime.utcnow()
        self._save_user(conn.user_id)
        return conn

    # ─────────────────────────────────────────
    # Portfolio aggregation
    # ─────────────────────────────────────────

    async def get_total_equity(self, user_id: str) -> float:
        """Aggregates equity across all active connections."""
        connections = await self.list_connections(user_id)
        total_equity = 0.0

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
                    account = client.get_account()
                    total_equity += float(account.equity)
                elif conn.broker_type == BrokerType.IBKR:
                    logger.debug(f"IBKR equity not yet supported for connection {conn.id}")
            except Exception as e:
                logger.error(f"Failed to fetch equity for {conn.id}: {e}")

        return total_equity

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
                        })
                elif conn.broker_type == BrokerType.IBKR:
                    logger.debug(f"IBKR positions not yet supported for connection {conn.id}")
            except Exception as e:
                logger.error(f"Failed to fetch positions for connection {conn.id} ({conn.name}): {e}")

        return all_positions

    async def list_connection_sources(self, user_id: str) -> List[Dict[str, Any]]:
        """Returns lightweight list of active connections for the account selector dropdown."""
        connections = await self.list_connections(user_id)
        return [
            {"id": c.id, "name": c.name, "broker_type": c.broker_type.value, "environment": c.environment}
            for c in connections
            if c.is_active
        ]


# Singleton instance
broker_service = BrokerService()
