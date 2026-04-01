"""
quant_system/data/live_websocket.py
================================================================================
ASYNC TICK CONSUMER (ALPACA LIVE CRYPTO/STOCK STREAM)
================================================================================
"""

import asyncio
import json
import websockets
import os
import logging
from datetime import datetime

logger = logging.getLogger("AlpacaStream")

class AlpacaDataStream:
    def __init__(self, symbol: str = "BTC/USD"):
        self.symbol = symbol
        self.api_key = os.environ.get("APCA_API_KEY_ID", "MISSING_KEY")
        self.api_secret = os.environ.get("APCA_API_SECRET_KEY", "MISSING_SECRET")
        # Alpaca V1beta3 Crypto Data Stream
        self.uri = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        self.queue = asyncio.Queue()

    async def _listen(self):
        """Asynchronously controls network persistence wrapping Exponential Backoff sequences natively."""
        backoff = 1
        while True:
            try:
                async with websockets.connect(self.uri) as ws:
                    # Authenticate over stream
                    auth_msg = {"action": "auth", "key": self.api_key, "secret": self.api_secret}
                    await ws.send(json.dumps(auth_msg))
                    auth_resp = await ws.recv()
                    logger.info(f"Stream Authenticated -> {auth_resp}")
                    
                    # Subscribe strictly to target asset
                    sub_msg = {"action": "subscribe", "trades": [self.symbol]}
                    await ws.send(json.dumps(sub_msg))
                    sub_resp = await ws.recv()
                    logger.info(f"Stream Verified & Subscribed -> {sub_resp}")
                    
                    backoff = 1 # Network stabilized
                    
                    async for msg in ws:
                        data = json.loads(msg)
                        for event in data:
                            if event.get("T") == "t": # Physical trade event
                                tick = {
                                    "stream": "trade",
                                    "symbol": event.get("S"),
                                    "price": float(event.get("p")),
                                    "size": float(event.get("s")),
                                    "timestamp": event.get("t")
                                }
                                await self.queue.put(tick)
                                
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"WebSocket Drop Detected. Firing backoff: {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            except Exception as e:
                logger.error(f"WebSocket Fatal Trace: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def stream_ticks(self):
        """Yields exact dict schemas required by internal O(1) buffer logic indefinitely."""
        asyncio.create_task(self._listen()) # Attach process cleanly
        while True:
            tick = await self.queue.get()
            yield tick
