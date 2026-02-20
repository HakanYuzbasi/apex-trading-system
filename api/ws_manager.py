import json
import logging
from collections import defaultdict
from typing import DefaultDict, Dict, List
from fastapi import WebSocket

logger = logging.getLogger("api")

def _sanitize_floats(obj):
    """Replace NaN/Inf float values with JSON-safe defaults."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connections_by_tenant: DefaultDict[str, List[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, tenant_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connections_by_tenant[tenant_id].append(websocket)
        setattr(websocket.state, "tenant_id", tenant_id)
        # We handle metrics externally now or via a broader callback

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

    async def broadcast(self, message: Dict, increment_metrics_fn=None):
        if increment_metrics_fn:
            increment_metrics_fn()
            
        encoded_message = json.dumps(message, default=_sanitize_floats)
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

        encoded_message = json.dumps(message, default=_sanitize_floats)
        for connection in self.connections_by_tenant.get(tenant_id, []).copy():
            try:
                await connection.send_text(encoded_message)
            except RuntimeError as exc:
                logger.error("Error sending message to tenant %s websocket: %s", tenant_id, exc)
                await self.disconnect(connection)

    def connected_tenants(self) -> List[str]:
        """Return currently connected tenant IDs."""
        return list(self.connections_by_tenant.keys())

manager = ConnectionManager()
