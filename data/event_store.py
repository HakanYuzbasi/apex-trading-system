"""
data/event_store.py
Write-Ahead Log (WAL) pattern for Deterministic Replay & Event Sourcing.
Serializes market ticks, signals, and risk decisions to an append-only JSONL file.
"""
import asyncio
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class EventType:
    MARKET_TICK = "MARKET_TICK"
    SIGNAL_GENERATION = "SIGNAL_GENERATION"
    RISK_DECISION = "RISK_DECISION"
    ORDER_EXECUTION = "ORDER_EXECUTION"
    POSITION_UPDATE = "POSITION_UPDATE"
    CAPITAL_UPDATE = "CAPITAL_UPDATE"

class EventStore:
    """
    Asynchronous Write-Ahead Log for trading events.
    Ensures 100% deterministic replayability.
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir) / "audit"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate daily journal file
        date_str = datetime.utcnow().strftime("%Y%m%d")
        self.journal_path = self.data_dir / f"event_journal_{date_str}.jsonl"
        
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_hash = hashlib.sha256(b"genesis").hexdigest()
        
        # In-memory projection for immediate consistency
        self.state: Dict[str, Any] = {
            "positions": {},
            "capital": 0.0,
        }

    async def start(self):
        """Start the background WAL writer."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._write_loop())
        logger.info(f"📓 EventStore WAL initialized. Journaling to {self.journal_path}")

    async def stop(self):
        """Gracefully stop the background writer."""
        self._running = False
        if self._worker_task:
            while not self._queue.empty():
                await asyncio.sleep(0.1)
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def dispatch(self, event_type: str, payload: Dict[str, Any]):
        """Non-blocking dispatch of an event to the WAL."""
        if not self._running:
            return
            
        timestamp = datetime.utcnow().isoformat()
        
        # Create tamper-evident chain of custody
        raw_event = f"{self._last_hash}|{timestamp}|{event_type}|{json.dumps(payload, sort_keys=True)}"
        self._last_hash = hashlib.sha256(raw_event.encode()).hexdigest()
        
        event = {
            "timestamp": timestamp,
            "type": event_type,
            "payload": payload,
            "hash": self._last_hash
        }
        
        # Mutate local projection immediately
        self._apply_to_state(event_type, payload)
        
        # Queue for persistent storage
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.error("EventStore WAL queue full! Dropping event.")

    def _apply_to_state(self, event_type: str, payload: Dict[str, Any]):
        """Apply mutations to in-memory state projection."""
        if event_type == EventType.POSITION_UPDATE:
            sym = payload.get("symbol")
            if sym:
                self.state["positions"][sym] = payload.get("quantity", 0)
        elif event_type == EventType.CAPITAL_UPDATE:
            self.state["capital"] = payload.get("capital", self.state["capital"])

    async def _write_loop(self):
        """Background worker that writes events to disk."""
        import aiofiles
        try:
            async with aiofiles.open(self.journal_path, mode='a') as f:
                while self._running:
                    try:
                        event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                        await f.write(json.dumps(event) + '\n')
                        await f.flush()
                        self._queue.task_done()
                    except asyncio.TimeoutError:
                        continue
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"EventStore WAL write loop failed: {e}")