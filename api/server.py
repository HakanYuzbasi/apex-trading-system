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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import math
from datetime import datetime

from config import ApexConfig
from core.logging_config import setup_logging

logger = logging.getLogger("api")
from api.auth import authenticate_websocket, require_user

setup_logging(
    level=ApexConfig.LOG_LEVEL,
    log_file=ApexConfig.LOG_FILE,
    json_format=False,
    console_output=True,
    max_bytes=ApexConfig.LOG_MAX_BYTES,
    backup_count=ApexConfig.LOG_BACKUP_COUNT,
)

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

# Path to trading state file
STATE_FILE = ApexConfig.DATA_DIR / "trading_state.json"
PRICE_CACHE_FILE = ApexConfig.DATA_DIR / "price_cache.json"

def read_price_cache() -> Dict[str, float]:
    """Read current price cache from file."""
    try:
        if PRICE_CACHE_FILE.exists():
            with open(PRICE_CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Error reading price cache: {e}")
    return {}

app = FastAPI(title="APEX Trading API", version="2.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SaaS auth: resolve user from PostgreSQL (JWT/API key) or legacy in-memory; set request.state.user
try:
    from services.auth.middleware import SaaSAuthMiddleware
    app.add_middleware(SaaSAuthMiddleware)
except Exception as e:
    logging.getLogger("api").warning("SaaS auth middleware not loaded: %s", e)

# Mount SaaS auth and feature routers
try:
    from services.auth.router import router as auth_router
    app.include_router(auth_router)
except Exception as e:
    logging.getLogger("api").warning("SaaS auth router not loaded: %s", e)

try:
    from services.backtest_validator.router import router as backtest_validator_router
    app.include_router(backtest_validator_router)
except Exception as e:
    logging.getLogger("api").warning("Backtest validator router not loaded: %s", e)

try:
    from services.execution_simulator.router import router as execution_sim_router
    app.include_router(execution_sim_router)
except Exception as e:
    logging.getLogger("api").warning("Execution simulator router not loaded: %s", e)

try:
    from services.drift_monitor.router import router as drift_monitor_router
    app.include_router(drift_monitor_router)
except Exception as e:
    logging.getLogger("api").warning("Drift monitor router not loaded: %s", e)

try:
    from services.compliance_copilot.router import router as compliance_copilot_router
    app.include_router(compliance_copilot_router)
except Exception as e:
    logging.getLogger("api").warning("Compliance copilot router not loaded: %s", e)

try:
    from services.portfolio_allocator.router import router as portfolio_allocator_router
    app.include_router(portfolio_allocator_router)
except Exception as e:
    logging.getLogger("api").warning("Portfolio allocator router not loaded: %s", e)

# Health check endpoints
try:
    from api.health import router as health_router
    app.include_router(health_router)
except Exception as e:
    logging.getLogger("api").warning("Health router not loaded: %s", e)

try:
    from services.tca.router import router as tca_router
    app.include_router(tca_router)
except Exception as e:
    logging.getLogger("api").warning("Could not load TCA router: %s", e)

# --------------------------------------------------------------------------------
# WebSocket Manager
# --------------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info("Client connected via WebSocket")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                logger.info("Client disconnected")
            else:
                logger.warning("Attempted to disconnect unknown websocket")

    async def broadcast(self, message: Dict):
        """Send message to all connected clients."""
        async with self._lock:
            dead: List[WebSocket] = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")
                    dead.append(connection)
            for ws in dead:
                self.active_connections.remove(ws)

manager = ConnectionManager()

# --------------------------------------------------------------------------------
# State Reader (From trading_state.json)
# --------------------------------------------------------------------------------

def read_trading_state() -> Dict:
    """Read current trading state from file, enriched with live prices from cache."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)

            # Enrich positions with live prices from price cache
            price_cache = read_price_cache()
            positions = data.get("positions", {})
            total_position_pnl = 0.0

            for symbol, pos in positions.items():
                live_price = price_cache.get(symbol, 0)
                avg_price = pos.get("avg_price", 0)
                qty = pos.get("qty", 0)

                if live_price > 0 and avg_price > 0:
                    pos["current_price"] = live_price
                    if qty > 0:  # Long
                        pnl = (live_price - avg_price) * qty
                        pnl_pct = (live_price / avg_price - 1) * 100
                    else:  # Short
                        pnl = (avg_price - live_price) * abs(qty)
                        pnl_pct = (avg_price / live_price - 1) * 100 if live_price > 0 else 0
                    pos["pnl"] = round(pnl, 2)
                    pos["pnl_pct"] = round(pnl_pct, 2)
                    total_position_pnl += pnl

            # Update total_pnl if we have position P&L data
            if total_position_pnl != 0:
                starting_capital = data.get("starting_capital", data.get("capital", 0))
                if starting_capital > 0:
                    data["total_pnl"] = round(total_position_pnl, 2)
                    data["daily_pnl"] = round(total_position_pnl, 2)  # Approximate as daily

            return _sanitize_floats(data)
    except Exception as e:
        logger.error(f"Error reading state file: {e}")

    # Return default state if file doesn't exist
    return {
        "timestamp": None,
        "capital": 0,
        "positions": {},
        "signals": {},
        "daily_pnl": 0,
        "total_pnl": 0,
        "sector_exposure": {}
    }

def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def _state_is_fresh(state: Dict, threshold_seconds: int) -> bool:
    ts = _parse_timestamp(state.get("timestamp"))
    if not ts:
        return False
    now = datetime.now(ts.tzinfo) if ts.tzinfo else datetime.utcnow()
    age = (now - ts).total_seconds()
    return age <= threshold_seconds

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
                    "initial_capital": current_state.get("initial_capital", 0),
                    "starting_capital": current_state.get("starting_capital", 0),
                    "positions": current_state.get("positions", {}),
                    "daily_pnl": current_state.get("daily_pnl", 0),
                    "total_pnl": current_state.get("total_pnl", 0),
                    "max_drawdown": current_state.get("max_drawdown", 0),
                    "sharpe_ratio": current_state.get("sharpe_ratio", 0),
                    "win_rate": current_state.get("win_rate", 0),
                    "sector_exposure": current_state.get("sector_exposure", {}),
                    "open_positions": current_state.get("open_positions", 0),
                    "max_positions": current_state.get("max_positions", ApexConfig.MAX_POSITIONS),
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
async def get_status(user=Depends(require_user)):
    state = read_trading_state()
    return {
        "status": "online" if _state_is_fresh(state, ApexConfig.HEALTH_STALENESS_SECONDS) else "offline",
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
async def get_positions(user=Depends(require_user)):
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
async def get_full_state(user=Depends(require_user)):
    """Get complete trading state."""
    return read_trading_state()

@app.get("/health")
async def get_health(user=Depends(require_user)):
    """Lightweight health check (auth required)."""
    state = read_trading_state()
    return {
        "status": "online" if _state_is_fresh(state, ApexConfig.HEALTH_STALENESS_SECONDS) else "offline",
        "timestamp": state.get("timestamp"),
        "stale_after_seconds": ApexConfig.HEALTH_STALENESS_SECONDS,
        "api": "ok",
    }

@app.get("/sectors")
async def get_sector_exposure(user=Depends(require_user)):
    """Get sector exposure breakdown."""
    state = read_trading_state()
    return state.get("sector_exposure", {})

# --------------------------------------------------------------------------------
# WebSocket Endpoint
# --------------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    user = await authenticate_websocket(websocket)
    if not user:
        await websocket.close(code=1008)
        return
    await manager.connect(websocket)
    try:
        # Send current state immediately on connect
        current_state = read_trading_state()
        await websocket.send_json({
            "type": "state_update",
            "timestamp": current_state.get("timestamp", datetime.now().isoformat()),
            "capital": current_state.get("capital", 0),
            "initial_capital": current_state.get("initial_capital", 0),
            "starting_capital": current_state.get("starting_capital", 0),
            "positions": current_state.get("positions", {}),
            "daily_pnl": current_state.get("daily_pnl", 0),
            "total_pnl": current_state.get("total_pnl", 0),
            "max_drawdown": current_state.get("max_drawdown", 0),
            "sharpe_ratio": current_state.get("sharpe_ratio", 0),
            "win_rate": current_state.get("win_rate", 0),
            "sector_exposure": current_state.get("sector_exposure", {}),
            "open_positions": current_state.get("open_positions", 0),
            "max_positions": current_state.get("max_positions", ApexConfig.MAX_POSITIONS),
            "total_trades": current_state.get("total_trades", 0)
        })
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received command: {data}")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
