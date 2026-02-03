"""
api/server.py - APEX Trading API Server

Core backend for the SOTA UI.
Exposes REST endpoints and WebSockets for real-time data streaming.
Reads state from trading_state.json written by the ApexTrader.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from datetime import datetime

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

# Path to trading state file
STATE_FILE = Path(__file__).parent.parent / "data" / "trading_state.json"

app = FastAPI(title="APEX Trading API", version="2.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------
# WebSocket Manager
# --------------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Client connected via WebSocket")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("Client disconnected")

    async def broadcast(self, message: Dict):
        """Send message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")

manager = ConnectionManager()

# --------------------------------------------------------------------------------
# State Reader (From trading_state.json)
# --------------------------------------------------------------------------------

def read_trading_state() -> Dict:
    """Read current trading state from file."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading state file: {e}")

    # Return default state if file doesn't exist
    return {
        "timestamp": datetime.now().isoformat(),
        "capital": 0,
        "positions": {},
        "signals": {},
        "daily_pnl": 0,
        "total_pnl": 0,
        "sector_exposure": {}
    }

# --------------------------------------------------------------------------------
# Real-time State Streaming
# --------------------------------------------------------------------------------

async def stream_trading_state():
    """Stream real trading state to connected clients."""
    last_state = None
    while True:
        try:
            current_state = read_trading_state()

            # Only broadcast if state changed
            if current_state != last_state:
                update = {
                    "type": "state_update",
                    "timestamp": current_state.get("timestamp", datetime.now().isoformat()),
                    "capital": current_state.get("capital", 0),
                    "starting_capital": current_state.get("starting_capital", 0),
                    "positions": current_state.get("positions", {}),
                    "daily_pnl": current_state.get("daily_pnl", 0),
                    "total_pnl": current_state.get("total_pnl", 0),
                    "max_drawdown": current_state.get("max_drawdown", 0),
                    "sharpe_ratio": current_state.get("sharpe_ratio", 0),
                    "win_rate": current_state.get("win_rate", 0),
                    "sector_exposure": current_state.get("sector_exposure", {}),
                    "open_positions": current_state.get("open_positions", 0),
                    "total_trades": current_state.get("total_trades", 0)
                }
                await manager.broadcast(update)
                last_state = current_state

        except Exception as e:
            logger.error(f"Error streaming state: {e}")

        await asyncio.sleep(2)  # Poll every 2 seconds

@app.on_event("startup")
async def startup_event():
    logger.info("APEX API Server Starting...")
    logger.info(f"Reading state from: {STATE_FILE}")
    asyncio.create_task(stream_trading_state())

# --------------------------------------------------------------------------------
# REST Endpoints
# --------------------------------------------------------------------------------

@app.get("/status")
async def get_status():
    state = read_trading_state()
    return {
        "status": "online" if state.get("timestamp") else "offline",
        "timestamp": state.get("timestamp"),
        "capital": state.get("capital", 0),
        "starting_capital": state.get("starting_capital", 0),
        "daily_pnl": state.get("daily_pnl", 0),
        "total_pnl": state.get("total_pnl", 0),
        "max_drawdown": state.get("max_drawdown", 0),
        "sharpe_ratio": state.get("sharpe_ratio", 0),
        "win_rate": state.get("win_rate", 0),
        "open_positions": state.get("open_positions", 0),
        "total_trades": state.get("total_trades", 0)
    }

@app.get("/positions")
async def get_positions():
    state = read_trading_state()
    positions = state.get("positions", {})

    result = []
    for symbol, data in positions.items():
        result.append({
            "symbol": symbol,
            "qty": data.get("qty", 0),
            "side": data.get("side", "LONG"),
            "entry": data.get("avg_price", 0),
            "current": data.get("current_price", 0),
            "pnl": data.get("pnl", 0),
            "pnl_pct": data.get("pnl_pct", 0),
            "signal": data.get("current_signal", 0),
            "signal_direction": data.get("signal_direction", "UNKNOWN")
        })

    return result

@app.get("/state")
async def get_full_state():
    """Get complete trading state."""
    return read_trading_state()

@app.get("/sectors")
async def get_sector_exposure():
    """Get sector exposure breakdown."""
    state = read_trading_state()
    return state.get("sector_exposure", {})

# --------------------------------------------------------------------------------
# WebSocket Endpoint
# --------------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current state immediately on connect
        current_state = read_trading_state()
        await websocket.send_json({
            "type": "state_update",
            "timestamp": current_state.get("timestamp", datetime.now().isoformat()),
            "capital": current_state.get("capital", 0),
            "starting_capital": current_state.get("starting_capital", 0),
            "positions": current_state.get("positions", {}),
            "daily_pnl": current_state.get("daily_pnl", 0),
            "total_pnl": current_state.get("total_pnl", 0),
            "max_drawdown": current_state.get("max_drawdown", 0),
            "sharpe_ratio": current_state.get("sharpe_ratio", 0),
            "win_rate": current_state.get("win_rate", 0),
            "sector_exposure": current_state.get("sector_exposure", {}),
            "open_positions": current_state.get("open_positions", 0),
            "total_trades": current_state.get("total_trades", 0)
        })
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received command: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
