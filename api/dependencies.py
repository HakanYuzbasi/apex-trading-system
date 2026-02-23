import json
import logging
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import ApexConfig

logger = logging.getLogger("api")

MONEY_ABS_MAX = 1_000_000_000_000.0
SHARPE_ABS_MAX = 20.0
COUNT_MAX = 1_000_000
DRAWDOWN_PERCENT_ABS_MAX = 100.0

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
    "daily_pnl_realized": 0,
    "daily_pnl_unrealized_fallback": 0,
    "daily_pnl_source": "equity_delta",
    "daily_pnl_by_broker": {"ibkr": 0, "alpaca": 0},
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


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _bounded_float(value: Any, minimum: float, maximum: float, fallback: float = 0.0) -> float:
    parsed = _as_finite_float(value)
    if parsed is None or parsed < minimum or parsed > maximum:
        return fallback
    return parsed


def _bounded_int(value: Any, minimum: int, maximum: int, fallback: int = 0) -> int:
    parsed = _as_finite_float(value)
    if parsed is None:
        return fallback
    rounded = int(parsed)
    if rounded < minimum or rounded > maximum:
        return fallback
    return rounded


def sanitize_execution_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize top-level execution KPIs for API responses.
    Rejects non-finite / implausible values and derives stable fallbacks.
    """
    capital_raw = _bounded_float(raw.get("capital"), -MONEY_ABS_MAX, MONEY_ABS_MAX, fallback=float("nan"))
    starting_capital_raw = _bounded_float(
        raw.get("starting_capital", raw.get("initial_capital")),
        -MONEY_ABS_MAX,
        MONEY_ABS_MAX,
        fallback=float("nan"),
    )
    total_pnl_raw = _bounded_float(raw.get("total_pnl"), -MONEY_ABS_MAX, MONEY_ABS_MAX, fallback=float("nan"))

    # Keep these three fields coherent when one is missing/invalid.
    if math.isnan(capital_raw) and (not math.isnan(starting_capital_raw)) and (not math.isnan(total_pnl_raw)):
        capital_raw = starting_capital_raw + total_pnl_raw
    if math.isnan(starting_capital_raw) and (not math.isnan(capital_raw)) and (not math.isnan(total_pnl_raw)):
        starting_capital_raw = capital_raw - total_pnl_raw
    if math.isnan(total_pnl_raw) and (not math.isnan(capital_raw)) and (not math.isnan(starting_capital_raw)):
        total_pnl_raw = capital_raw - starting_capital_raw

    capital = _bounded_float(capital_raw, -MONEY_ABS_MAX, MONEY_ABS_MAX, fallback=0.0)
    total_pnl = _bounded_float(total_pnl_raw, -MONEY_ABS_MAX, MONEY_ABS_MAX, fallback=0.0)
    starting_capital = _bounded_float(
        starting_capital_raw,
        -MONEY_ABS_MAX,
        MONEY_ABS_MAX,
        fallback=max(0.0, capital - total_pnl),
    )

    daily_pnl = _bounded_float(raw.get("daily_pnl"), -MONEY_ABS_MAX, MONEY_ABS_MAX, fallback=0.0)
    daily_pnl_realized = _bounded_float(
        raw.get("daily_pnl_realized", daily_pnl),
        -MONEY_ABS_MAX,
        MONEY_ABS_MAX,
        fallback=daily_pnl,
    )
    daily_pnl_unrealized_fallback = _bounded_float(
        raw.get("daily_pnl_unrealized_fallback"),
        -MONEY_ABS_MAX,
        MONEY_ABS_MAX,
        fallback=0.0,
    )
    daily_pnl_source_raw = str(raw.get("daily_pnl_source", "")).strip().lower()
    daily_pnl_source = daily_pnl_source_raw if daily_pnl_source_raw in {"broker_fills", "equity_delta"} else "equity_delta"
    daily_pnl_by_broker_raw = raw.get("daily_pnl_by_broker")
    daily_pnl_by_broker = {"ibkr": 0.0, "alpaca": 0.0}
    if isinstance(daily_pnl_by_broker_raw, dict):
        daily_pnl_by_broker["ibkr"] = _bounded_float(
            daily_pnl_by_broker_raw.get("ibkr"),
            -MONEY_ABS_MAX,
            MONEY_ABS_MAX,
            fallback=0.0,
        )
        daily_pnl_by_broker["alpaca"] = _bounded_float(
            daily_pnl_by_broker_raw.get("alpaca"),
            -MONEY_ABS_MAX,
            MONEY_ABS_MAX,
            fallback=0.0,
        )

    drawdown = _as_finite_float(raw.get("max_drawdown"))
    if drawdown is None:
        drawdown = 0.0
    elif abs(drawdown) > 1.0 and abs(drawdown) > DRAWDOWN_PERCENT_ABS_MAX:
        drawdown = 0.0

    sharpe_ratio = _bounded_float(raw.get("sharpe_ratio"), -SHARPE_ABS_MAX, SHARPE_ABS_MAX, fallback=0.0)

    win_rate_raw = _as_finite_float(raw.get("win_rate"))
    if win_rate_raw is None:
        win_rate = 0.0
    elif 0.0 <= win_rate_raw <= 1.0:
        win_rate = win_rate_raw
    elif 1.0 < win_rate_raw <= 100.0:
        win_rate = win_rate_raw / 100.0
    else:
        win_rate = 0.0

    open_positions = _bounded_int(raw.get("open_positions"), 0, COUNT_MAX, fallback=0)
    option_positions = _bounded_int(raw.get("option_positions"), 0, COUNT_MAX, fallback=0)
    open_positions_total = _bounded_int(
        raw.get("open_positions_total"),
        0,
        COUNT_MAX,
        fallback=open_positions + option_positions,
    )
    open_positions_total = max(open_positions_total, open_positions + option_positions)

    total_trades = _bounded_int(
        raw.get("total_trades", raw.get("trades_count")),
        0,
        COUNT_MAX,
        fallback=0,
    )

    broker_heartbeats_raw = raw.get("broker_heartbeats")
    broker_heartbeats: Dict[str, Dict[str, Any]] = {}
    if isinstance(broker_heartbeats_raw, dict):
        for broker in ("ibkr", "alpaca"):
            rec = broker_heartbeats_raw.get(broker)
            if not isinstance(rec, dict):
                continue
            broker_heartbeats[broker] = {
                "last_success_ts": str(rec.get("last_success_ts")) if rec.get("last_success_ts") else None,
                "last_error_ts": str(rec.get("last_error_ts")) if rec.get("last_error_ts") else None,
                "last_error": str(rec.get("last_error")) if rec.get("last_error") else None,
                "healthy": bool(rec.get("healthy")),
            }

    payload = {
        "capital": capital,
        "starting_capital": starting_capital,
        "daily_pnl": daily_pnl,
        "daily_pnl_realized": daily_pnl_realized,
        "daily_pnl_unrealized_fallback": daily_pnl_unrealized_fallback,
        "daily_pnl_source": daily_pnl_source,
        "daily_pnl_by_broker": daily_pnl_by_broker,
        "total_pnl": total_pnl,
        "max_drawdown": drawdown,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "open_positions": open_positions,
        "option_positions": option_positions,
        "open_positions_total": open_positions_total,
        "total_trades": total_trades,
    }
    if broker_heartbeats:
        payload["broker_heartbeats"] = broker_heartbeats
    return payload

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

        daily_source = str(data.get("daily_pnl_source", "")).strip().lower()
        has_broker_truth_daily = daily_source == "broker_fills" or ("daily_pnl_realized" in data)

        # Update total_pnl if we have position P&L data
        if total_position_pnl != 0:
            starting_capital = data.get("starting_capital", data.get("capital", 0))
            if starting_capital > 0:
                data["total_pnl"] = round(total_position_pnl, 2)
                if not has_broker_truth_daily:
                    data["daily_pnl"] = round(total_position_pnl, 2)  # Approximate as daily only when broker-truth is absent

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
    # Ensure both are offset-aware if possible, or both naive
    if ts.tzinfo and not now.tzinfo:
        now = now.replace(tzinfo=ts.tzinfo)
    elif not ts.tzinfo and now.tzinfo:
        now = now.astimezone(None).replace(tzinfo=None)
    
    age = (now - ts).total_seconds()
    return age <= threshold_seconds
