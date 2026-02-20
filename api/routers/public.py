import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import ApexConfig
from api.ws_manager import manager
from api.dependencies import read_trading_state

router = APIRouter(prefix="/public", tags=["Public"])

@router.get("/metrics")
async def get_public_metrics():
    """
    Returns public-facing top-level KPI metrics.
    No authentication required.
    """
    state = read_trading_state()
    return {
        "status": "online" if (state.get("timestamp") and datetime.fromisoformat(state["timestamp"]).timestamp() > datetime.now().timestamp() - ApexConfig.HEALTH_STALENESS_SECONDS) else "offline",
        "timestamp": state.get("timestamp"),
        "capital": state.get("capital", 0),
        "starting_capital": state.get("starting_capital", 0),
        "daily_pnl": state.get("daily_pnl", 0),
        "total_pnl": state.get("total_pnl", 0),
        "max_drawdown": state.get("max_drawdown", 0),
        "sharpe_ratio": state.get("sharpe_ratio", 0),
        "win_rate": state.get("win_rate", 0),
        "open_positions": state.get("open_positions", 0),
        "option_positions": state.get("option_positions", 0),
        "open_positions_total": state.get("open_positions_total", state.get("open_positions", 0)),
        "total_trades": state.get("total_trades", 0)
    }

@router.get("/cockpit")
async def get_public_cockpit():
    """
    Returns a read-only snapshot of the trading state for the public dashboard.
    No authentication required. Sensitive keys (if any) are omitted.
    """
    state = read_trading_state()
    return {
        "status": {
            "api_reachable": True,
            "state_fresh": True,
            "timestamp": state.get("timestamp", datetime.now().isoformat()),
            "capital": state.get("capital", 0),
            "daily_pnl": state.get("daily_pnl", 0),
            "total_pnl": state.get("total_pnl", 0),
            "max_drawdown": state.get("max_drawdown", 0),
            "sharpe_ratio": state.get("sharpe_ratio", 0),
            "win_rate": state.get("win_rate", 0),
            "open_positions": state.get("open_positions", 0),
            "max_positions": state.get("max_positions", 5),
            "total_trades": state.get("total_trades", 0),
        },
        "positions": [
            {
                "symbol": sym,
                "qty": pos_data.get("qty", 0),
                "side": pos_data.get("side", "LONG"),
                "entry": pos_data.get("avg_price", 0),
                "current": pos_data.get("current_price", 0),
                "pnl": pos_data.get("pnl", 0),
                "pnl_pct": pos_data.get("pnl_pct", 0),
                "signal": pos_data.get("current_signal", 0),
                "signal_direction": pos_data.get("signal_direction", "UNKNOWN"),
                "source_id": pos_data.get("source_id", "public_view")
            }
            for sym, pos_data in state.get("positions", {}).items()
        ]
    }

@router.websocket("/ws")
async def public_websocket_endpoint(websocket: WebSocket):
    """
    Public WebSocket endpoint. Streams state updates without authentication.
    Does NOT accept incoming commands from the client.
    """
    await manager.connect(websocket, tenant_id="public")
    try:
        # Send current state immediately on connect
        current_state = read_trading_state()
        
        # Formulate initial push
        update = {
            "type": "state_update",
            "timestamp": current_state.get("timestamp", datetime.now().isoformat()),
            "capital": current_state.get("capital", 0),
            "positions": current_state.get("positions", {}),
            "daily_pnl": current_state.get("daily_pnl", 0),
            "total_pnl": current_state.get("total_pnl", 0),
            "max_drawdown": current_state.get("max_drawdown", 0),
            "sharpe_ratio": current_state.get("sharpe_ratio", 0),
            "win_rate": current_state.get("win_rate", 0),
            "open_positions": current_state.get("open_positions", 0),
            "total_trades": current_state.get("total_trades", 0)
        }
        await websocket.send_json(update)
        
        while True:
            # We must wait for messages to know when client disconnects
            # but we ignore any commands sent by unauthenticated users.
            data = await websocket.receive_text()
            pass
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
