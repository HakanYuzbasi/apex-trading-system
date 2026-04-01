"""
data/websocket_streamer.py - Real-Time Market Data WebSocket
Connects to Alpaca's high-speed WSS endpoints to eliminate REST polling latency.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
import websockets
from typing import Dict, Set

from config import ApexConfig
from core.symbols import AssetClass, parse_symbol

logger = logging.getLogger(__name__)

class WebsocketStreamer:
    """
    Subscribes to Alpaca's WebSocket data streams for equities and crypto.
    Maintains a real-time memory buffer of Level-1 trade/quote prices.
    """
    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str):
        self.api_key = alpaca_api_key
        self.secret_key = alpaca_secret_key
        
        # Determine strict WSS endpoints
        self.equity_wss_url = "wss://stream.data.alpaca.markets/v2/iex"
        self.crypto_wss_url = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        
        # Memory buffer mimicking market_data._price_cache
        self._price_cache: Dict[str, tuple] = {}
        
        self.equity_symbols: Set[str] = set()
        self.crypto_symbols: Set[str] = set()
        
        self._shutdown_event = asyncio.Event()
        self._tasks = []

        # Option D: Ready-gate prevents the execution loop from fetching prices
        # before the WSS handshake (auth + subscribe) completes, eliminating
        # the race condition where REST fallbacks were always hit at open.
        self._equity_ready = asyncio.Event()
        self._crypto_ready = asyncio.Event()
        # Latency instrumentation: track WSS hit rate vs REST fallback rate
        self._wss_hits: int = 0
        self._wss_misses: int = 0

        # Reconnect metrics
        self._reconnect_count: Dict[str, int] = {"equity": 0, "crypto": 0}
        self._last_connect_ts: Dict[str, float] = {}
        self._last_disconnect_ts: Dict[str, float] = {}
        self._session_start_ts: float = datetime.now(timezone.utc).timestamp()

    @property
    def hit_rate(self) -> float:
        """Fraction of price fetches served from WSS cache (0.0–1.0)."""
        total = self._wss_hits + self._wss_misses
        return self._wss_hits / total if total > 0 else 0.0

    def get_metrics(self) -> dict:
        """Return current WebSocket connection health metrics."""
        now = datetime.now(timezone.utc).timestamp()
        session_uptime = now - self._session_start_ts
        return {
            "hit_rate":          round(self.hit_rate, 3),
            "wss_hits":          self._wss_hits,
            "wss_misses":        self._wss_misses,
            "equity_reconnects": self._reconnect_count.get("equity", 0),
            "crypto_reconnects": self._reconnect_count.get("crypto", 0),
            "equity_connected":  self._equity_ready.is_set(),
            "crypto_connected":  self._crypto_ready.is_set(),
            "equity_last_connect_ts":     self._last_connect_ts.get("equity"),
            "crypto_last_connect_ts":     self._last_connect_ts.get("crypto"),
            "equity_last_disconnect_ts":  self._last_disconnect_ts.get("equity"),
            "crypto_last_disconnect_ts":  self._last_disconnect_ts.get("crypto"),
            "session_uptime_seconds":     round(session_uptime, 0),
            "cached_symbols":    len(self._price_cache),
        }

    async def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """
        Block until at least one WSS channel confirms auth+subscribe OR timeout.
        Returns True when ready, False on timeout (caller falls back to REST).
        """
        if not self._tasks:
            return False  # nothing started
        try:
            ready_futures = []
            if self.equity_symbols:
                ready_futures.append(self._equity_ready.wait())
            if self.crypto_symbols:
                ready_futures.append(self._crypto_ready.wait())
            if ready_futures:
                done, _ = await asyncio.wait(
                    [asyncio.ensure_future(f) for f in ready_futures],
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                return bool(done)
        except Exception:
            pass
        return False

    def load_universe(self, symbols: Set[str]):
        """Sorts the massive universe into their distinct WebSocket feeds."""
        for sym in symbols:
            try:
                parsed = parse_symbol(sym)
                if parsed.asset_class == AssetClass.CRYPTO:
                    # e.g., 'BTC/USD'
                    self.crypto_symbols.add(parsed.base + "/USD")
                else:
                    self.equity_symbols.add(parsed.raw)
            except Exception as e:
                logger.debug(f"WSS parsing failed for {sym}: {e}")

    async def start(self):
        """Asynchronously triggers the WSS connection loops."""
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys missing — WebsocketStreamer bypassed.")
            return

        logger.info(f"🚀 Starting High-Frequency WSS Engines (Equities: {len(self.equity_symbols)}, Crypto: {len(self.crypto_symbols)})")
        
        if self.equity_symbols:
            self._tasks.append(asyncio.create_task(self._stream_alpaca_wss(
                self.equity_wss_url, list(self.equity_symbols), stream_type="equity"
            )))
            
        if self.crypto_symbols:
            self._tasks.append(asyncio.create_task(self._stream_alpaca_wss(
                self.crypto_wss_url, list(self.crypto_symbols), stream_type="crypto"
            )))

    async def stop(self):
        """Cleanly drains the WebSockets connections."""
        self._shutdown_event.set()
        for t in self._tasks:
            t.cancel()

    def get_current_price(self, symbol: str) -> float:
        """O(1) Memory fetch mimicking the legacy MarketDataFetcher for zero-latency bridging."""
        try:
            parsed = parse_symbol(symbol)
            lookup_key = parsed.normalized
            if lookup_key in self._price_cache:
                price, _ = self._price_cache[lookup_key]
                self._wss_hits += 1
                return price
        except Exception:
            pass
        self._wss_misses += 1
        return 0.0

    async def _stream_alpaca_wss(self, url: str, symbols: list, stream_type: str):
        """Core WebSocket listener algorithm with automatic dropout recovery."""
        retry_delay = 1
        
        while not self._shutdown_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    # 1. Authenticate Request
                    auth_message = {
                        "action": "auth",
                        "key": self.api_key,
                        "secret": self.secret_key
                    }
                    await ws.send(json.dumps(auth_message))
                    auth_response = await ws.recv()
                    logger.debug(f"[{stream_type.upper()}] WSS Auth Response: {auth_response}")

                    # 2. Subscribe to strict Trade ticks (t) and Quotes (q) for lowest latency
                    sub_message = {
                        "action": "subscribe",
                        "trades": symbols,
                        "bars": symbols  # As a secondary fallback structure
                    }
                    await ws.send(json.dumps(sub_message))
                    sub_response = await ws.recv()
                    logger.debug(f"[{stream_type.upper()}] WSS Subscription: {sub_response}")

                    # Option D: Signal that this channel is ready — unblocks wait_until_ready()
                    if stream_type == "equity":
                        self._equity_ready.set()
                    else:
                        self._crypto_ready.set()
                    logger.info(f"✅ [{stream_type.upper()}] WSS ready — {len(symbols)} symbols subscribed")

                    # Track (re)connect timestamp
                    self._last_connect_ts[stream_type] = datetime.now(timezone.utc).timestamp()

                    retry_delay = 1 # Reset health threshold
                    
                    # 3. Endless Consumption Pipeline
                    while not self._shutdown_event.is_set():
                        msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                        payloads = json.loads(msg)
                        for data in payloads:
                            msg_type = data.get("T")
                            # Extract Trade ('t') or Bar ('b')
                            if msg_type in ("t", "b"): 
                                sym = data.get("S")
                                price = float(data.get("p", data.get("c", 0.0))) # 'p' for trade price, 'c' for bar close
                                if sym and price > 0:
                                    # Normalize memory storage format
                                    if stream_type == "crypto" and "/" in sym:
                                        norm_sym = f"CRYPTO:{sym}"
                                    else:
                                        norm_sym = sym
                                    self._price_cache[norm_sym] = (price, datetime.now(timezone.utc))
                                    
            except asyncio.TimeoutError:
                continue # Heartbeat timeout, quietly drift and reconnect
            except asyncio.CancelledError:
                logger.info(f"[{stream_type.upper()}] WSS Gracefully Terminated.")
                break
            except Exception as e:
                self._reconnect_count[stream_type] = self._reconnect_count.get(stream_type, 0) + 1
                self._last_disconnect_ts[stream_type] = datetime.now(timezone.utc).timestamp()
                logger.warning(f"[{stream_type.upper()}] WSS stream crashed: {e}. Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
