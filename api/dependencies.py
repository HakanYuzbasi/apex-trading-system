import json
import logging
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import ApexConfig

logger = logging.getLogger("api")

# Path to trading state file
STATE_FILE = ApexConfig.DATA_DIR / "trading_state.json"
PRICE_CACHE_FILE = ApexConfig.DATA_DIR / "price_cache.json"
CONTROL_COMMAND_FILE = ApexConfig.DATA_DIR / "trading_control_commands.json"
GOVERNOR_POLICY_DIR = ApexConfig.DATA_DIR / "governor_policies"
PREFLIGHT_STATUS_FILE = ApexConfig.DATA_DIR / "preflight_status.json"
SOCIAL_DECISION_AUDIT_FILE = Path(
    os.getenv(
        "APEX_SOCIAL_DECISION_AUDIT_FILE",
        str(ApexConfig.DATA_DIR / "runtime" / "social_governor_decisions.jsonl"),
    )
)
SOCIAL_DECISION_AUDIT_LEGACY_FILE = Path(
    os.getenv(
        "APEX_SOCIAL_DECISION_AUDIT_LEGACY_FILE",
        str(ApexConfig.DATA_DIR / "audit" / "social_governor_decisions.jsonl"),
    )
)

DEFAULT_STATE = {
    "timestamp": None,
    "capital": 0,
    "positions": {},
    "signals": {},
    "daily_pnl": 0,
    "total_pnl": 0,
    "sector_exposure": {},
}

_price_cache_lock = threading.Lock()
_price_cache_data: Dict[str, float] = {}
_price_cache_mtime_ns: Optional[int] = None

_state_cache_lock = threading.Lock()
_state_cache_data: Dict = DEFAULT_STATE
_state_cache_mtime_ns: Optional[int] = None
_state_cache_price_mtime_ns: Optional[int] = None

def _mtime_ns(path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None

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

def read_price_cache() -> Tuple[Dict[str, float], Optional[int]]:
    """Read current price cache from file with mtime-based caching."""
    global _price_cache_data, _price_cache_mtime_ns

    mtime_ns = _mtime_ns(PRICE_CACHE_FILE)
    with _price_cache_lock:
        if _price_cache_mtime_ns == mtime_ns:
            return _price_cache_data, _price_cache_mtime_ns

        if mtime_ns is None:
            _price_cache_data = {}
            _price_cache_mtime_ns = None
            return _price_cache_data, _price_cache_mtime_ns

        try:
            with open(PRICE_CACHE_FILE, "r") as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    loaded = {}
        except Exception as e:
            logger.debug(f"Error reading price cache: {e}")
            loaded = {}

        _price_cache_data = loaded
        _price_cache_mtime_ns = mtime_ns
        return _price_cache_data, _price_cache_mtime_ns

def read_trading_state() -> Dict:
    """Read current trading state with mtime-aware caching."""
    global _state_cache_data, _state_cache_mtime_ns, _state_cache_price_mtime_ns

    state_mtime_ns = _mtime_ns(STATE_FILE)
    price_cache, price_mtime_ns = read_price_cache()

    with _state_cache_lock:
        if (
            _state_cache_data is not None
            and _state_cache_mtime_ns == state_mtime_ns
            and _state_cache_price_mtime_ns == price_mtime_ns
        ):
            return _state_cache_data

        if state_mtime_ns is None:
            _state_cache_data = DEFAULT_STATE
            _state_cache_mtime_ns = None
            _state_cache_price_mtime_ns = price_mtime_ns
            return _state_cache_data

        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading state file: {e}")
            _state_cache_data = DEFAULT_STATE
            _state_cache_mtime_ns = None
            _state_cache_price_mtime_ns = price_mtime_ns
            return _state_cache_data

        # Enrich positions with live prices from cache
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

        sanitized = _sanitize_floats(data)
        _state_cache_data = sanitized
        _state_cache_mtime_ns = state_mtime_ns
        _state_cache_price_mtime_ns = price_mtime_ns
        return _state_cache_data

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
