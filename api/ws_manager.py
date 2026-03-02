import json
import logging
import math
from collections import defaultdict
from typing import DefaultDict, Dict, List
from fastapi import WebSocket

logger = logging.getLogger("api")

# Send a full state_update every N ticks to resync clients after any drift.
# At the default 1s poll interval this is every 30 seconds.
_FULL_BROADCAST_EVERY_N_TICKS = 30

# Critical fields always included in delta (never omitted even if unchanged).
# These drive the dashboard's live P&L / position count display.
_DELTA_ALWAYS_INCLUDE = frozenset({
    "capital", "equity", "pnl", "pnl_pct", "position_count", "timestamp",
    "starting_capital", "initial_capital",
})

# Maximum simultaneous WebSocket connections per user (DoS guard).
_MAX_WS_PER_USER = 5

def _sanitize_floats(obj):
    """Replace NaN/Inf float values with JSON-safe defaults."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _values_equal(a, b) -> bool:
    """Deep equality check that handles dicts/lists via JSON comparison."""
    if type(a) != type(b):
        return False
    if isinstance(a, (dict, list)):
        return json.dumps(a, default=str, sort_keys=True) == json.dumps(b, default=str, sort_keys=True)
    return a == b


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connections_by_tenant: DefaultDict[str, List[WebSocket]] = defaultdict(list)
        # Delta-encoding state per tenant
        self._last_state_by_tenant: Dict[str, Dict] = {}
        self._tick_count_by_tenant: Dict[str, int] = defaultdict(int)

    async def connect(self, websocket: WebSocket, tenant_id: str, is_admin: bool = False) -> bool:
        """Accept and register a WebSocket connection.

        Returns False (and closes the socket) if the per-user connection cap is hit.
        """
        current_count = len(self.connections_by_tenant.get(tenant_id, []))
        if current_count >= _MAX_WS_PER_USER:
            logger.warning(
                "WS connection rejected for tenant %s: already at cap (%d/%d)",
                tenant_id, current_count, _MAX_WS_PER_USER,
            )
            await websocket.accept()
            await websocket.close(code=4008, reason="Too many connections")
            return False

        await websocket.accept()
        self.active_connections.append(websocket)
        self.connections_by_tenant[tenant_id].append(websocket)
        setattr(websocket.state, "tenant_id", tenant_id)
        setattr(websocket.state, "is_admin", is_admin)
        return True

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        tenant_id = getattr(websocket.state, "tenant_id", None)
        if tenant_id:
            tenant_connections = self.connections_by_tenant.get(tenant_id, [])
            if websocket in tenant_connections:
                tenant_connections.remove(websocket)
            if not tenant_connections and tenant_id in self.connections_by_tenant:
                del self.connections_by_tenant[tenant_id]
                # Clean up delta-encoding state for this tenant
                self._last_state_by_tenant.pop(tenant_id, None)
                self._tick_count_by_tenant.pop(tenant_id, None)

    def compute_delta_payload(self, tenant_id: str, new_state: Dict) -> Dict:
        """
        Return a delta-encoded payload for a tenant.

        On the first broadcast or every _FULL_BROADCAST_EVERY_N_TICKS ticks,
        returns ``type: "state_update"`` with all fields.  Otherwise returns
        ``type: "state_delta"`` containing only the fields that changed since
        the last broadcast — typically just 2-5 scalars (~90% bandwidth saving).
        """
        self._tick_count_by_tenant[tenant_id] += 1
        tick = self._tick_count_by_tenant[tenant_id]
        prev = self._last_state_by_tenant.get(tenant_id)
        force_full = (prev is None) or (tick % _FULL_BROADCAST_EVERY_N_TICKS == 0)

        # Always store the latest full state for future diffs
        self._last_state_by_tenant[tenant_id] = new_state

        if force_full:
            return dict(new_state, type="state_update")

        # Compute changed fields only (shallow for scalars, deep for dicts/lists).
        # Critical fields (_DELTA_ALWAYS_INCLUDE) are always present even if unchanged.
        delta = {
            k: v for k, v in new_state.items()
            if k in _DELTA_ALWAYS_INCLUDE or not _values_equal(v, prev.get(k))
        }

        if not delta:
            # Nothing changed — skip broadcast by returning an empty marker
            return {}

        delta["type"] = "state_delta"
        # Always include tenant_id so the frontend can route correctly
        delta["tenant_id"] = new_state.get("tenant_id", tenant_id)
        return delta

    async def broadcast(self, message: Dict, increment_metrics_fn=None):
        if increment_metrics_fn:
            increment_metrics_fn()
            
        encoded_message = json.dumps(_sanitize_floats(message))
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(encoded_message)
            except RuntimeError as e:
                logger.error(f"Error sending message to {connection.client}: {e}")
                await self.disconnect(connection)

    async def broadcast_to_tenant(self, tenant_id: str, message: Dict, increment_metrics_fn=None):
        """Broadcast a message to websocket clients for exactly one tenant."""
        if increment_metrics_fn:
            increment_metrics_fn()

        encoded_message = json.dumps(_sanitize_floats(message))
        for connection in self.connections_by_tenant.get(tenant_id, []).copy():
            try:
                await connection.send_text(encoded_message)
            except RuntimeError as exc:
                logger.error("Error sending message to tenant %s websocket: %s", tenant_id, exc)
                await self.disconnect(connection)

    def connected_tenants(self) -> List[str]:
        """Return currently connected tenant IDs."""
        return list(self.connections_by_tenant.keys())

    def is_tenant_admin(self, tenant_id: str) -> bool:
        """Check if any active WS connection for this tenant has the admin role."""
        for ws in self.connections_by_tenant.get(tenant_id, []):
            if getattr(ws.state, "is_admin", False):
                return True
        return False

manager = ConnectionManager()
