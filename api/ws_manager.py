import json
import logging
from typing import Dict, List
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

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # We handle metrics externally now or via a broader callback

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

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

manager = ConnectionManager()
