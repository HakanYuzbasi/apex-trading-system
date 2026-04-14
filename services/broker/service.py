import os
import json
import logging
import asyncio
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from datetime import datetime

from cryptography.fernet import Fernet
from alpaca.trading.client import TradingClient
from ib_insync import IB
import json
from pathlib import Path

from config import ApexConfig
from models.broker import BrokerConnection, BrokerType
from core.exceptions import ApexBrokerError
from services.common.db import db_session
from services.trading.models import BrokerConnectionModel
from execution.ibkr_lease_manager import lease_manager

logger = logging.getLogger(__name__)

# Process-wide semaphore: at most 2 concurrent IBKR TCP connections.
# TWS paper has a hard ~32 connection limit; allowing unlimited concurrent
# connect() calls during error/retry storms saturates the pool and causes
# a self-reinforcing failure cascade.  2 slots = ample headroom while
# keeping TWS healthy.
_ibkr_connect_semaphore = asyncio.Semaphore(2)


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
        self._positions_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._equity_snapshot_cache: Dict[str, Dict[str, Any]] = {}
        self._positions_snapshot_cache: Dict[str, Dict[str, Any]] = {}
        self._equity_locks: Dict[str, asyncio.Lock] = {}
        self._positions_locks: Dict[str, asyncio.Lock] = {}
        self._equity_refresh_ttl_seconds: int = int(
            os.getenv("APEX_BROKER_EQUITY_REFRESH_INTERVAL_SECONDS", "120")
        )
        self._positions_refresh_ttl_seconds: int = int(
            os.getenv("APEX_BROKER_POSITIONS_REFRESH_INTERVAL_SECONDS", "120")
        )
        self._ibkr_connect_retry_attempts: int = int(
            os.getenv("APEX_IBKR_CONNECT_RETRY_ATTEMPTS", "3")
        )

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
        last_error: Optional[ApexBrokerError] = None
        for attempt in range(self._ibkr_connect_retry_attempts):
            allocated_id = await lease_manager.allocate(preferred_id=client_id, ttl=30)
            ib = IB()
            fatal_error: Dict[str, Any] = {}

            def _on_error(req_id: int, error_code: int, error_string: str, contract: Any) -> None:
                if int(error_code) == 10141:
                    fatal_error["message"] = error_string

            error_event = getattr(ib, "errorEvent", None)
            if error_event is not None:
                error_event += _on_error
            try:
                await ib.connectAsync(host, port, clientId=allocated_id, timeout=30)
                if fatal_error.get("message"):
                    raise self._classify_ibkr_error(
                        host=host,
                        port=port,
                        client_id=allocated_id,
                        error=RuntimeError(fatal_error["message"]),
                        fatal_message=fatal_error["message"],
                    )
                if not ib.isConnected():
                    raise ConnectionError("IBKR connection check failed")
                ib.disconnect()
                return ib
            except Exception as exc:
                error = exc if isinstance(exc, ApexBrokerError) else self._classify_ibkr_error(
                    host=host,
                    port=port,
                    client_id=allocated_id,
                    error=exc,
                    fatal_message=fatal_error.get("message"),
                )
                last_error = error
                logger.error("IBKR connection failed for clientId %s: %s", allocated_id, error)
                if attempt >= self._ibkr_connect_retry_attempts - 1 or not self._is_retryable_ibkr_error(error):
                    raise error from exc
                await asyncio.sleep(1.0)
            finally:
                if ib.isConnected():
                    ib.disconnect()
                await lease_manager.release(allocated_id)

        if last_error is not None:
            raise last_error
        raise ApexBrokerError(
            code="IBKR_CONNECT_FAILED",
            message="Failed to connect to IBKR",
            context={"host": host, "port": port, "client_id": client_id},
        )

    @staticmethod
    def _classify_ibkr_error(
        host: str,
        port: int,
        client_id: Optional[int],
        error: Exception,
        fatal_message: Optional[str] = None,
    ) -> ApexBrokerError:
        detail = str(fatal_message or error or "").strip()
        lowered = detail.lower()
        context = {
            "host": host,
            "port": port,
            "client_id": client_id,
            "error": detail or type(error).__name__,
        }

        if fatal_message or "paper trading disclaimer" in lowered or "10141" in lowered:
            return ApexBrokerError(
                code="IBKR_PAPER_DISCLAIMER_REQUIRED",
                message="IBKR paper trading disclaimer must be accepted in TWS/Gateway before API access is allowed",
                context=context,
            )
        if "already in use" in lowered or ("clientid" in lowered and "in use" in lowered):
            return ApexBrokerError(
                code="IBKR_CLIENT_ID_IN_USE",
                message="IBKR rejected the API client ID because it is already in use",
                context=context,
            )
        if (
            isinstance(error, (TimeoutError, asyncio.TimeoutError))
            or "timeout" in lowered
            or "timed out" in lowered
        ):
            return ApexBrokerError(
                code="IBKR_CONNECT_TIMEOUT",
                message="IBKR API connection timed out",
                context=context,
            )
        if "connection reset by peer" in lowered or "peer closed connection" in lowered:
            return ApexBrokerError(
                code="IBKR_CONNECTION_RESET",
                message="IBKR peer reset the API connection",
                context=context,
            )
        return ApexBrokerError(
            code="IBKR_CONNECT_FAILED",
            message="Failed to connect to IBKR",
            context=context,
        )

    @staticmethod
    def _is_retryable_ibkr_error(error: Exception) -> bool:
        if not isinstance(error, ApexBrokerError):
            return False
        return error.code in {
            "IBKR_CLIENT_ID_IN_USE",
            "IBKR_CONNECT_TIMEOUT",
            "IBKR_CONNECTION_RESET",
        }

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
        cached = self._positions_snapshot_cache.get(user_id)
        if self._is_snapshot_fresh(cached, self._positions_refresh_ttl_seconds):
            return [dict(row) for row in cached.get("positions", [])]

        lock = self._positions_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            cached = self._positions_snapshot_cache.get(user_id)
            if self._is_snapshot_fresh(cached, self._positions_refresh_ttl_seconds):
                return [dict(row) for row in cached.get("positions", [])]

            connections = await self.list_connections(user_id)
            active_connections = [conn for conn in connections if conn.is_active]
            if not active_connections:
                self._positions_snapshot_cache[user_id] = {
                    "positions": [],
                    "cached_at": time.time(),
                }
                return []

            tasks = [self._fetch_positions_with_fallback(connection=conn) for conn in active_connections]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_positions: List[Dict[str, Any]] = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    conn = active_connections[idx]
                    logger.exception("Failed to fetch positions for connection %s (%s): %s", conn.id, conn.name, result)
                    continue
                all_positions.extend(result)

            self._positions_snapshot_cache[user_id] = {
                "positions": [dict(row) for row in all_positions],
                "cached_at": time.time(),
            }
            return all_positions

    async def _fetch_positions_with_fallback(self, connection: BrokerConnection) -> List[Dict[str, Any]]:
        """Fetch connection positions with stale-cache fallback."""
        try:
            raw_creds = connection.credentials.get("data", "")
            credentials = self._decrypt_credentials(raw_creds)
            positions = await self._fetch_connection_positions(connection, credentials)
            self._positions_cache[connection.id] = positions
            return positions
        except Exception as exc:  # SWALLOW: Use engine's state or last-known cache when broker fetch fails
            # 1. Try internal cache first
            cached = self._positions_cache.get(connection.id)
            if cached is not None:
                stale_positions: List[Dict[str, Any]] = []
                for row in cached:
                    stale_row = dict(row)
                    stale_row["stale"] = True
                    stale_row["source_status"] = "degraded"
                    stale_positions.append(stale_row)
                return stale_positions
            
            # 2. Hard Fallback: Check engine's trading_state.json for positions
            try:
                state_path = ApexConfig.DATA_DIR / "trading_state.json"
                if state_path.exists():
                    with open(state_path, "r") as f:
                        state_data = json.load(f)
                    
                    engine_positions = state_data.get("positions", {})
                    if engine_positions:
                        logger.info(f"Using harness state-file fallback for {connection.broker_type.value} positions")
                        rows = []
                        for sym, data in engine_positions.items():
                            # Approximate matching: if we have multiple brokers, this might be messy, 
                            # but for single-broker deployments it restores the UI.
                            rows.append({
                                "source": f"{connection.name} (Harness Fallback)",
                                "source_id": connection.id,
                                "broker_type": connection.broker_type.value,
                                "symbol": sym,
                                "qty": data.get("qty", 0),
                                "side": data.get("side", "LONG"),
                                "avg_cost": data.get("avg_price", 0),
                                "current_price": data.get("current_price", 0),
                                "stale": True,
                                "as_of": state_data.get("timestamp"),
                                "source_status": "degraded",
                            })
                        return rows
            except Exception as fallback_exc:
                logger.debug(f"Harness state position fallback failed: {fallback_exc}")

            raise ApexBrokerError(
                code="BROKER_POSITION_FETCH_FAILED",
                message=f"Unable to fetch positions for connection {connection.id}",
                context={"connection_id": connection.id, "broker_type": connection.broker_type.value, "error": str(exc)},
            ) from exc

    _POSITIONS_FETCH_TIMEOUT_SECONDS = 45

    async def _fetch_connection_positions(
        self,
        connection: BrokerConnection,
        credentials: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        as_of = datetime.utcnow().isoformat() + "Z"
        if connection.broker_type == BrokerType.ALPACA:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._fetch_alpaca_positions_blocking,
                    source_name=connection.name,
                    source_id=connection.id,
                    environment=connection.environment,
                    credentials=credentials,
                    as_of=as_of,
                ),
                timeout=self._POSITIONS_FETCH_TIMEOUT_SECONDS,
            )

        if connection.broker_type == BrokerType.IBKR:
            last_error: Optional[Exception] = None
            for attempt in range(self._ibkr_connect_retry_attempts):
                allocated_id = await lease_manager.allocate(
                    preferred_id=connection.client_id,
                    ttl=30,
                )
                try:
                    async with _ibkr_connect_semaphore:
                        return await asyncio.wait_for(
                            asyncio.to_thread(
                                self._fetch_ibkr_positions_blocking,
                                source_name=connection.name,
                                source_id=connection.id,
                                client_id=allocated_id,
                                credentials=credentials,
                                as_of=as_of,
                            ),
                            timeout=self._POSITIONS_FETCH_TIMEOUT_SECONDS,
                        )
                except Exception as exc:
                    error = exc if isinstance(exc, ApexBrokerError) else self._classify_ibkr_error(
                        host=str(credentials.get("host", "127.0.0.1")),
                        port=int(credentials.get("port", 7497)),
                        client_id=allocated_id,
                        error=exc,
                    )
                    last_error = error
                    if attempt >= self._ibkr_connect_retry_attempts - 1 or not self._is_retryable_ibkr_error(error):
                        raise error from exc
                    logger.warning(
                        "Retrying IBKR position fetch for %s after %s (attempt %d/%d)",
                        connection.id,
                        error.code,
                        attempt + 1,
                        self._ibkr_connect_retry_attempts,
                    )
                    await asyncio.sleep(1.0)
                finally:
                    await lease_manager.release(allocated_id)
            if last_error is not None:
                raise last_error

        raise ApexBrokerError(
            code="BROKER_UNSUPPORTED",
            message=f"Unsupported broker type: {connection.broker_type}",
            context={"broker_type": str(connection.broker_type)},
        )

    @staticmethod
    def _fetch_alpaca_positions_blocking(
        source_name: str,
        source_id: str,
        environment: str,
        credentials: Dict[str, Any],
        as_of: str,
    ) -> List[Dict[str, Any]]:
        client = TradingClient(
            credentials["api_key"],
            credentials["secret_key"],
            paper=(environment == "paper"),
        )
        raw_positions = client.get_all_positions()
        rows: List[Dict[str, Any]] = []
        for pos in raw_positions:
            qty = float(pos.qty or 0)
            if math.isclose(qty, 0.0, abs_tol=1e-12):
                continue
            rows.append({
                "source": source_name,
                "source_id": source_id,
                "broker_type": "alpaca",
                "symbol": str(pos.symbol),
                "security_type": "CRYPTO",
                "expiry": None,
                "strike": None,
                "right": None,
                "qty": qty,
                "side": str(pos.side.value) if hasattr(pos.side, "value") else str(pos.side),
                "avg_cost": float(pos.avg_entry_price or 0),
                "current_price": float(pos.current_price or 0),
                "market_value": float(pos.market_value or 0),
                "unrealized_pl": float(pos.unrealized_pl or 0),
                "unrealized_plpc": float(pos.unrealized_plpc or 0),
                "stale": False,
                "as_of": as_of,
                "source_status": "ok",
            })
        return rows

    @staticmethod
    def _format_ibkr_contract_symbol(contract: Any) -> str:
        sec_type = str(getattr(contract, "secType", "")).strip().upper()
        symbol = str(getattr(contract, "symbol", "")).strip()
        currency = str(getattr(contract, "currency", "")).strip().upper()
        if sec_type in {"CASH", "CRYPTO"} and symbol and currency:
            return f"{symbol}/{currency}"
        return symbol

    @staticmethod
    def _normalize_ibkr_expiry(value: Any) -> str:
        raw = str(value or "").strip()
        if len(raw) == 8 and raw.isdigit():
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        if len(raw) == 6 and raw.isdigit():
            return f"{raw[:4]}-{raw[4:6]}-01"
        return raw

    @staticmethod
    def _normalize_option_right(value: Any) -> str:
        token = str(value or "").strip().upper()
        if token == "CALL":
            return "C"
        if token == "PUT":
            return "P"
        if token in {"C", "P"}:
            return token
        return ""

    @staticmethod
    def _fetch_ibkr_positions_blocking(
        source_name: str,
        source_id: str,
        client_id: Optional[int],
        credentials: Dict[str, Any],
        as_of: str,
    ) -> List[Dict[str, Any]]:
        host = credentials.get("host", "127.0.0.1")
        port = int(credentials.get("port", 7497))
        import random
        ib_client_id = int(client_id or credentials.get("client_id") or random.randint(100, 199))
        owned_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            owned_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(owned_loop)

        ib = IB()
        fatal_error: Dict[str, Any] = {}

        def _on_error(req_id: int, error_code: int, error_string: str, contract: Any) -> None:
            if int(error_code) == 10141:
                fatal_error["message"] = error_string

        error_event = getattr(ib, "errorEvent", None)
        if error_event is not None:
            error_event += _on_error
        try:
            ib.connect(host, port, clientId=ib_client_id, timeout=30, readonly=True)
            if fatal_error.get("message"):
                raise BrokerService._classify_ibkr_error(
                    host=host,
                    port=port,
                    client_id=ib_client_id,
                    error=RuntimeError(fatal_error["message"]),
                    fatal_message=fatal_error["message"],
                )
            portfolio_by_conid: Dict[int, Any] = {}
            for item in ib.portfolio():
                con_id = int(getattr(getattr(item, "contract", None), "conId", 0) or 0)
                if con_id:
                    portfolio_by_conid[con_id] = item

            rows: List[Dict[str, Any]] = []
            for pos in ib.positions():
                contract = getattr(pos, "contract", None)
                if contract is None:
                    continue
                sec_type = str(getattr(contract, "secType", "")).strip().upper()
                qty = float(getattr(pos, "position", 0) or 0)
                if math.isclose(qty, 0.0, abs_tol=1e-12):
                    continue

                symbol = BrokerService._format_ibkr_contract_symbol(contract)
                expiry = BrokerService._normalize_ibkr_expiry(
                    getattr(contract, "lastTradeDateOrContractMonth", ""),
                )
                right = BrokerService._normalize_option_right(getattr(contract, "right", ""))
                strike = float(getattr(contract, "strike", 0.0) or 0.0) if "OPT" in sec_type else 0.0
                if "OPT" in sec_type:
                    symbol = f"{symbol} {expiry} {strike:g} {right}".strip()
                con_id = int(getattr(contract, "conId", 0) or 0)
                portfolio_item = portfolio_by_conid.get(con_id)
                avg_cost = float(getattr(pos, "avgCost", 0) or 0)
                current_price = float(getattr(portfolio_item, "marketPrice", 0) or 0)
                market_value = float(getattr(portfolio_item, "marketValue", 0) or 0)
                unrealized_pl = float(getattr(portfolio_item, "unrealizedPNL", 0) or 0)

                if math.isclose(current_price, 0.0, abs_tol=1e-12) and not math.isclose(avg_cost, 0.0, abs_tol=1e-12):
                    current_price = avg_cost
                if math.isclose(market_value, 0.0, abs_tol=1e-12):
                    market_value = qty * current_price
                basis = abs(qty * avg_cost)
                unrealized_plpc = (unrealized_pl / basis) if basis > 1e-12 else 0.0

                rows.append({
                    "source": source_name,
                    "source_id": source_id,
                    "broker_type": "ibkr",
                    "symbol": symbol,
                    "security_type": sec_type or "EQUITY",
                    "expiry": expiry or None,
                    "strike": strike if "OPT" in sec_type else None,
                    "right": right or None,
                    "qty": qty,
                    "side": "LONG" if qty > 0 else "SHORT",
                    "avg_cost": avg_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                    "unrealized_plpc": unrealized_plpc,
                    "stale": False,
                    "as_of": as_of,
                    "source_status": "ok",
                })
            return rows
        except Exception as exc:
            raise exc if isinstance(exc, ApexBrokerError) else BrokerService._classify_ibkr_error(
                host=host,
                port=port,
                client_id=ib_client_id,
                error=exc,
                fatal_message=fatal_error.get("message"),
            ) from exc
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass
            if owned_loop is not None:
                asyncio.set_event_loop(None)
                owned_loop.close()

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
                    BrokerConnectionModel.is_active
                )
            )
            return [row[0] for row in result.all() if row[0]]

    async def get_tenant_equity_snapshot(self, tenant_id: str) -> Dict[str, Any]:
        """Aggregate tenant equity with per-source normalized snapshots."""
        cached = self._equity_snapshot_cache.get(tenant_id)
        if self._is_snapshot_fresh(cached, self._equity_refresh_ttl_seconds):
            payload = cached.get("snapshot", {})
            return dict(payload) if isinstance(payload, dict) else {
                "tenant_id": tenant_id,
                "total_equity": 0.0,
                "breakdown": [],
                "as_of": datetime.utcnow().isoformat() + "Z",
            }

        lock = self._equity_locks.setdefault(tenant_id, asyncio.Lock())
        async with lock:
            cached = self._equity_snapshot_cache.get(tenant_id)
            if self._is_snapshot_fresh(cached, self._equity_refresh_ttl_seconds):
                payload = cached.get("snapshot", {})
                return dict(payload) if isinstance(payload, dict) else {
                    "tenant_id": tenant_id,
                    "total_equity": 0.0,
                    "breakdown": [],
                    "as_of": datetime.utcnow().isoformat() + "Z",
                }

            snapshots = await self._collect_equity_snapshots_for_user(tenant_id)
            total_value = sum(snapshot.value for snapshot in snapshots)
            payload = {
                "tenant_id": tenant_id,
                "total_equity": float(total_value),
                "breakdown": [snapshot.__dict__ for snapshot in snapshots],
                "as_of": datetime.utcnow().isoformat() + "Z",
            }
            self._equity_snapshot_cache[tenant_id] = {
                "snapshot": payload,
                "cached_at": time.time(),
            }
            return payload

    @staticmethod
    def _is_snapshot_fresh(cached: Optional[Dict[str, Any]], ttl_seconds: int) -> bool:
        if not cached or ttl_seconds <= 0:
            return False
        cached_at = float(cached.get("cached_at", 0) or 0)
        if cached_at <= 0:
            return False
        return (time.time() - cached_at) <= float(ttl_seconds)

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
        except Exception as exc:  # SWALLOW: Use engine's state or last-known cache when broker is unavailable
            # 1. Try internal cache first
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
            
            # 2. Hard Fallback: Check engine's trading_state.json
            try:
                state_path = ApexConfig.DATA_DIR / "trading_state.json"
                if state_path.exists():
                    with open(state_path, "r") as f:
                        state_data = json.load(f)
                    
                    # Check if engine has fresh heartbeats for this broker
                    heartbeats = state_data.get("broker_heartbeats", {})
                    hb = heartbeats.get(connection.broker_type.value)
                    if hb and hb.get("healthy"):
                        # Extract equity from breakdown or daily_pnl_by_broker if available
                        # But simpler: use the aggregated 'capital' if it matches the broker context
                        val = state_data.get("capital") or state_data.get("equity")
                        if val is not None and val > 0:
                            logger.info(f"Using harness state-file fallback for {connection.broker_type.value} equity")
                            return BrokerEquitySnapshot(
                                value=float(val),
                                broker=connection.broker_type.value,
                                stale=True,
                                as_of=state_data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                                source=f"{connection.name} (Harness Fallback)",
                                source_id=connection.id,
                            )
            except Exception as fallback_exc:
                logger.debug(f"Harness state fallback failed: {fallback_exc}")

            raise ApexBrokerError(
                code="BROKER_EQUITY_FETCH_FAILED",
                message=f"Unable to fetch equity for connection {connection.id}",
                context={"connection_id": connection.id, "broker_type": connection.broker_type.value, "error": str(exc)},
            ) from exc

    _EQUITY_FETCH_TIMEOUT_SECONDS = 45

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
                as_of=datetime.utcnow().isoformat() + "Z",
                source=connection.name,
                source_id=connection.id,
            )

        if connection.broker_type == BrokerType.IBKR:
            last_error: Optional[Exception] = None
            for attempt in range(self._ibkr_connect_retry_attempts):
                allocated_id = await lease_manager.allocate(
                    preferred_id=connection.client_id,
                    ttl=30,
                )
                try:
                    async with _ibkr_connect_semaphore:
                        equity = await asyncio.wait_for(
                            asyncio.to_thread(
                                self._fetch_ibkr_equity_blocking,
                                credentials,
                                allocated_id,
                            ),
                            timeout=self._EQUITY_FETCH_TIMEOUT_SECONDS,
                        )
                    return BrokerEquitySnapshot(
                        value=float(equity),
                        broker="ibkr",
                        stale=False,
                        as_of=datetime.utcnow().isoformat() + "Z",
                        source=connection.name,
                        source_id=connection.id,
                    )
                except Exception as exc:
                    error = exc if isinstance(exc, ApexBrokerError) else self._classify_ibkr_error(
                        host=str(credentials.get("host", "127.0.0.1")),
                        port=int(credentials.get("port", 7497)),
                        client_id=allocated_id,
                        error=exc,
                    )
                    last_error = error
                    if attempt >= self._ibkr_connect_retry_attempts - 1 or not self._is_retryable_ibkr_error(error):
                        raise error from exc
                    logger.warning(
                        "Retrying IBKR equity fetch for %s after %s (attempt %d/%d)",
                        connection.id,
                        error.code,
                        attempt + 1,
                        self._ibkr_connect_retry_attempts,
                    )
                    await asyncio.sleep(1.0)
                finally:
                    await lease_manager.release(allocated_id)
            if last_error is not None:
                raise last_error

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
        import random
        ib_client_id = int(client_id or credentials.get("client_id") or random.randint(100, 199))
        owned_loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            # ib_insync resolves the current loop during connect(); worker threads
            # have no default loop in Python 3.11+, so create one explicitly.
            owned_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(owned_loop)
        ib = IB()
        fatal_error: Dict[str, Any] = {}

        def _on_error(req_id: int, error_code: int, error_string: str, contract: Any) -> None:
            if int(error_code) == 10141:
                fatal_error["message"] = error_string

        error_event = getattr(ib, "errorEvent", None)
        if error_event is not None:
            error_event += _on_error
        try:
            ib.connect(host, port, clientId=ib_client_id, timeout=30, readonly=True)
            if fatal_error.get("message"):
                raise BrokerService._classify_ibkr_error(
                    host=host,
                    port=port,
                    client_id=ib_client_id,
                    error=RuntimeError(fatal_error["message"]),
                    fatal_message=fatal_error["message"],
                )
            summary_rows = ib.accountSummary()
            for row in summary_rows:
                if getattr(row, "tag", "") == "NetLiquidation":
                    return float(row.value)
            raise ApexBrokerError(
                code="IBKR_EQUITY_MISSING",
                message="NetLiquidation tag not found in IBKR account summary",
                context={"host": host, "port": port, "client_id": ib_client_id},
            )
        except Exception as exc:
            raise exc if isinstance(exc, ApexBrokerError) else BrokerService._classify_ibkr_error(
                host=host,
                port=port,
                client_id=ib_client_id,
                error=exc,
                fatal_message=fatal_error.get("message"),
            ) from exc
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass
            if owned_loop is not None:
                asyncio.set_event_loop(None)
                owned_loop.close()


# Singleton instance
broker_service = BrokerService()
