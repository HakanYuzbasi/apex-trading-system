import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

from config import ApexConfig
from api.dependencies import _state_is_fresh, read_trading_state, sanitize_execution_metrics
from api.auth import rate_limit

router = APIRouter(prefix="/public", tags=["Public"])

@router.get("/metrics")
@rate_limit(requests=60, window=60)
async def get_public_metrics(request: Request = None):
    """
    Returns public-facing top-level KPI metrics.
    No authentication required.
    """
    state = read_trading_state()
    safe_metrics = sanitize_execution_metrics(state)
    return {
        "status": "online" if _state_is_fresh(state, ApexConfig.HEALTH_STALENESS_SECONDS) else "offline",
        "timestamp": state.get("timestamp"),
        **safe_metrics,
    }

@router.get("/cockpit")
@rate_limit(requests=60, window=60)
async def get_public_cockpit(request: Request = None):
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
            "starting_capital": state.get("starting_capital", 0),
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
    Runs a dedicated projection loop to ensure sensitive data is stripped.
    """
    await websocket.accept()
    
    # We do NOT register with `manager.connect` to avoid receiving 
    # the un-sanitized global `broadcast_to_tenant` payloads.
    
    async def _send_sanitized_state():
        last_timestamp = None
        while True:
            try:
                current_state = read_trading_state()
                current_timestamp = current_state.get("timestamp")
                
                # Only push if the state timestamp changed
                if current_timestamp != last_timestamp:
                    update = {
                        "type": "state_update",
                        "timestamp": current_timestamp or datetime.now().isoformat(),
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
                    last_timestamp = current_timestamp
            except Exception:
                pass # Swallow errors to prevent crashing the loop
            
            await asyncio.sleep(1.0)
            
    # Run the sender loop as a task
    sender_task = asyncio.create_task(_send_sanitized_state())
    
    try:
        while True:
            # We must wait for messages to know when client disconnects
            # but we ignore any commands sent by unauthenticated users.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
