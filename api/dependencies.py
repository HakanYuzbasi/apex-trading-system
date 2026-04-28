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

# Plain-symbol crypto tickers (Alpaca/IBKR format without CRYPTO: prefix or /)
# Used to correctly classify positions in _filter_state_by_session.
_KNOWN_CRYPTO_BASE = frozenset({
    "BTC", "ETH", "SOL", "AVAX", "LTC", "XRP", "DOT", "UNI", "AAVE",
    "BCH", "LINK", "DOGE", "ADA", "MATIC", "ATOM", "ALGO", "FIL",
    "NEAR", "APT", "TRX", "SUI", "INJ", "ARB", "OP", "UNI", "XLM",
    "PAXG", "RENDER", "BAT", "USDC",
})


def _is_crypto_symbol(symbol: str) -> bool:
    """Return True if the symbol represents a crypto asset."""
    if not symbol:
        return False
    # Standardize to uppercase for robust matching
    s = str(symbol).strip().upper()
    if s.startswith("CRYPTO:"):
        return True
    if "/" in s and "USD" in s:
        base = s.split("/")[0].replace("CRYPTO:", "").upper()
        return base in _KNOWN_CRYPTO_BASE
    if s.endswith("USD"):
        base = s[:-3]
        if base in _KNOWN_CRYPTO_BASE:
            return True
    # Final check: is the raw symbol one of our known crypto bases?
    return s in _KNOWN_CRYPTO_BASE


def _normalize_symbol(symbol: str) -> str:
    """Return a canonical representation of a symbol.

    Standardizes crypto symbols like 'BTCUSD' or 'CRYPTO:BTC/USD' to 'CRYPTO:BTC/USD'.
    This ensures that positions from different sources are merged correctly.
    """
    if not symbol:
        return symbol
    s = symbol.upper().strip()

    # Handle concatenated crypto (e.g., BTCUSD)
    if not (":" in s or "/" in s) and s.endswith("USD"):
        base = s[:-3]
        if base in _KNOWN_CRYPTO_BASE:
            return f"CRYPTO:{base}/USD"

    # Handle other variants via core.symbols
    try:
        from core.symbols import normalize_symbol as core_normalize
        return core_normalize(s)
    except Exception:
        return s

# Path to trading state file
STATE_FILE = ApexConfig.DATA_DIR / "trading_state.json"
CORE_STATE_FILE = ApexConfig.DATA_DIR / "core_trading_state.json"
CRYPTO_STATE_FILE = ApexConfig.DATA_DIR / "crypto_trading_state.json"

def get_engine():
    """Return the live trading engine singleton if running in-process, else None.

    The API server runs as a separate container from the trading engine, so this
    always returns None here. All call-sites handle None with graceful fallbacks
    (file-based or Redis state reads).
    """
    return None


def _session_state_file(session_type: str) -> Path:
    """Return the state file path for a given session type."""
    if session_type == "core":
        return CORE_STATE_FILE
    elif session_type == "crypto":
        return CRYPTO_STATE_FILE
    return STATE_FILE
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


def _calculate_sharpe_ratio(raw: Dict[str, Any]) -> float:
    """Calculate a proxy Sharpe ratio if not provided by the engine.

    Uses (Total P&L / Capital) / Max Drawdown as a risk-adjusted return estimate.
    While not a perfect Sharpe, it provides a functional 'Return/Risk' score for the UI.
    """
    try:
        # Check if engine already provided a valid non-zero sharpe.
        # Values outside [-10, 10] are almost certainly calculation errors.
        engine_sharpe = _as_finite_float(raw.get("sharpe_ratio"))
        if engine_sharpe is not None and 0 < engine_sharpe <= 10.0:
            return engine_sharpe

        total_pnl = _as_finite_float(raw.get("total_pnl")) or 0.0
        capital = _as_finite_float(raw.get("capital")) or 1.0
        drawdown = abs(_as_finite_float(raw.get("max_drawdown")) or 0.01)

        if capital <= 0 or total_pnl == 0:
            return 0.0

        # Proxy for risk-adjusted return: % return divided by % drawdown.
        # This acts as a 'Calmar-like' proxy for the Sharpe ratio when higher resolution
        # return history is unavailable (e.g., during session initialization).
        # Scaled to align with the return-volatility Sharpe observed in Pitch Metrics.
        return round((total_pnl / capital) / max(drawdown, 0.02) * 0.8, 2)
    except Exception:
        return 0.0


def _win_rate_from_audit(lookback_days: int = 7) -> Tuple[Optional[float], int]:
    """Compute win_rate and closed_trade count from trade audit JSONL files.

    Returns (win_rate, closed_trades) or (None, 0) if no exit data found.
    Reads the last `lookback_days` of audit files so intraday metrics are
    always fresh even when the engine's performance_tracker has symbol-key issues.
    """
    try:
        from datetime import timedelta
        exits = []
        today = datetime.utcnow()
        for days_back in range(lookback_days):
            date = today - timedelta(days=days_back)
            # Check both admin and admin-1 audit dirs (depends on auth config)
            for user_id in ("admin", "admin-1"):
                audit_path = (
                    ApexConfig.DATA_DIR
                    / "users"
                    / user_id
                    / "audit"
                    / f"trade_audit_{date.strftime('%Y%m%d')}.jsonl"
                )
                if not audit_path.exists():
                    continue
                for line in audit_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    # Accept both 'event' and 'event_type' field names
                    evt = str(row.get("event", row.get("event_type", ""))).upper()
                    if evt == "EXIT":
                        pnl = row.get("pnl_pct")
                        if pnl is not None:
                            try:
                                exits.append(float(pnl))
                            except (TypeError, ValueError):
                                pass
        if not exits:
            return None, 0
        wins = sum(1 for p in exits if p > 0)
        return wins / len(exits), len(exits)
    except Exception as e:
        logger.debug("_win_rate_from_audit error: %s", e)
        return None, 0


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

    drawdown_raw = _as_finite_float(raw.get("max_drawdown"))
    if drawdown_raw is None:
        drawdown = 0.0
    elif 1.0 < abs(drawdown_raw) <= DRAWDOWN_PERCENT_ABS_MAX:
        drawdown = drawdown_raw / 100.0
    elif abs(drawdown_raw) > DRAWDOWN_PERCENT_ABS_MAX:
        drawdown = 0.0
    else:
        drawdown = drawdown_raw

    sharpe_ratio = _calculate_sharpe_ratio(raw)

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

    # Override win_rate and closed-trade count from audit JSONL files.
    # The performance_tracker has symbol-key mismatches (ETH/USD vs CRYPTO:ETH/USD)
    # and missing BUY records for startup-restored positions, making its win_rate
    # unreliable. The audit files are the single source of truth.
    _audit_win_rate, _audit_closed = _win_rate_from_audit()
    if _audit_closed > 0:
        win_rate = _audit_win_rate  # type: ignore[assignment]
        # Use audit closed-trade count as total_trades floor (more accurate than tracker)
        total_trades = max(total_trades, _audit_closed)

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
        "meta_confidence_score": _as_finite_float(raw.get("meta_confidence_score")) or 1.0,
        "bayesian_vol_prob": _as_finite_float(raw.get("bayesian_vol_prob")) or 0.0,
        "correlation_matrix": raw.get("correlation_matrix", {}),
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

        # Build a live price map from broker_positions (market_value / qty)
        # as a fallback when price_cache doesn't have coverage.
        broker_pos_list = data.get("broker_positions", [])
        broker_price_map: Dict[str, float] = {}
        if isinstance(broker_pos_list, list):
            for bp in broker_pos_list:
                sym = str(bp.get("symbol", "") or bp.get("normalized_symbol", ""))
                mv = float(bp.get("market_value", 0) or 0)
                qty = float(bp.get("qty", 0) or 0)
                if sym and mv > 0 and abs(qty) > 1e-10:
                    broker_price_map[sym] = round(mv / abs(qty), 6)

        # Enrich and aggregate positions (merging by normalized symbol).
        # Deduplication pass first: the engine sometimes writes BOTH the compact form
        # (e.g. XRPUSD) and the canonical form (CRYPTO:XRP/USD) for the same position.
        # Prefer the canonical CRYPTO: form; if both exist with nearly identical qty,
        # skip the compact alias entirely to avoid double-counting.
        raw_positions = data.get("positions") or {}

        # Build a canonical→sources map to detect duplicates
        canonical_to_sources: Dict[str, list] = {}
        for sym in raw_positions:
            norm = _normalize_symbol(sym)
            canonical_to_sources.setdefault(norm, []).append(sym)

        # For each canonical key with >1 source, keep whichever source is already in
        # canonical form (CRYPTO:X/USD), discard compact aliases (XUSD) that have
        # a qty within 5% of the canonical source — they're the same position.
        deduped_positions: Dict[str, dict] = {}
        for norm_sym, sources in canonical_to_sources.items():
            if len(sources) == 1:
                deduped_positions[norm_sym] = raw_positions[sources[0]]
            else:
                # Multiple raw symbols map to the same canonical — pick the best one
                # Priority: canonical CRYPTO: prefix > compact XUSD > other
                canonical_src = next((s for s in sources if s.startswith("CRYPTO:")), None)
                compact_src = next((s for s in sources if not s.startswith("CRYPTO:")), None)
                chosen = raw_positions.get(canonical_src or sources[0])
                if canonical_src and compact_src:
                    qty_c = abs(float(raw_positions[canonical_src].get("qty", 0)))
                    qty_k = abs(float(raw_positions[compact_src].get("qty", 0)))
                    # If qtys are within 5% treat as duplicate — use canonical
                    if qty_c > 0 and abs(qty_c - qty_k) / qty_c < 0.05:
                        chosen = raw_positions[canonical_src]
                    else:
                        # Genuinely different qtys → sum (e.g. separate broker lots)
                        chosen = dict(raw_positions[canonical_src])
                        chosen["qty"] = round(qty_c + qty_k, 8)
                deduped_positions[norm_sym] = chosen

        merged_positions = {}
        total_position_pnl = 0.0

        for norm_sym, pos in deduped_positions.items():
            qty = pos.get("qty", 0)
            if abs(qty) < 1e-6:
                continue

            # Prefer price from canonical symbol form, fall back to compact
            raw_sym = norm_sym
            live_price = (price_cache.get(raw_sym, 0) or price_cache.get(raw_sym.replace("CRYPTO:", "").replace("/", ""), 0)
                          or broker_price_map.get(raw_sym, 0))
            avg_price = pos.get("avg_price", 0)

            pnl = 0.0
            pnl_pct = 0.0
            if live_price > 0 and avg_price > 0:
                if qty > 0:  # Long
                    pnl = (live_price - avg_price) * qty
                    pnl_pct = (live_price / avg_price - 1) * 100
                else:  # Short
                    pnl = (avg_price - live_price) * abs(qty)
                    pnl_pct = (avg_price / live_price - 1) * 100 if live_price > 0 else 0
                total_position_pnl += pnl

            merged_positions[norm_sym] = {
                "symbol": norm_sym,
                "qty": qty,
                "side": pos.get("side", "LONG"),
                "avg_price": avg_price,
                "entry": avg_price,
                "current": live_price,
                "current_price": live_price,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "signal": pos.get("signal", 0),
                "signal_direction": pos.get("signal_direction", "UNKNOWN"),
            }

        data["positions"] = merged_positions
        data["open_positions"] = len(merged_positions)

        daily_source = str(data.get("daily_pnl_source", "")).strip().lower()
        has_broker_truth_daily = daily_source == "broker_fills" or ("daily_pnl_realized" in data)

        # Update total_pnl / daily_pnl to include live unrealized P&L
        if total_position_pnl != 0:
            data["total_pnl"] = round(total_position_pnl, 2)
            if has_broker_truth_daily:
                # Add live unrealized on top of realized fills
                realized = float(data.get("daily_pnl_realized", data.get("daily_pnl", 0)) or 0)
                data["daily_pnl"] = round(realized + total_position_pnl, 2)
            else:
                data["daily_pnl"] = round(total_position_pnl, 2)

        # Normalize engine field names before sanitizing
        data = _alias_engine_state_fields(data)

        sanitized = _sanitize_floats(data)
        _state_cache_data = sanitized
        _state_cache_mtime_ns = state_mtime_ns
        _state_cache_price_mtime_ns = price_mtime_ns
        return _state_cache_data

async def async_read_trading_state() -> Dict:
    """Redis-first read of trading state; falls back to file-based mtime cache."""
    try:
        from services.common.redis_client import cache_get
        cached = await cache_get("apex:state:trading")
        if cached is not None and isinstance(cached, dict):
            return cached
    except Exception:
        pass
    return read_trading_state()


def read_session_state(session_type: str) -> Dict:
    """Read trading state for a specific session (core or crypto).

    Falls back to the unified state file if session-specific file doesn't exist.
    """
    state_file = _session_state_file(session_type)
    mtime = _mtime_ns(state_file)
    if mtime is None:
        # Fallback: filter unified state by session
        unified = read_trading_state()
        if session_type == "unified":
            return unified
        return _filter_state_by_session(unified, session_type)

    try:
        with open(state_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {session_type} state file: {e}")
        return dict(DEFAULT_STATE)

    # Build broker price map for positions not covered by the price cache
    broker_pos_raw = data.get("broker_positions", [])
    broker_price_map: Dict[str, float] = {}
    if isinstance(broker_pos_raw, list):
        for bp in broker_pos_raw:
            sym = str(bp.get("symbol", "") or bp.get("normalized_symbol", ""))
            mv = float(bp.get("market_value", 0) or 0)
            qty_bp = float(bp.get("qty", 0) or 0)
            if sym and mv > 0 and abs(qty_bp) > 1e-10:
                broker_price_map[sym] = round(mv / abs(qty_bp), 6)

    # Enrich and aggregate positions (merging by normalized symbol)
    price_cache, _ = read_price_cache()
    raw_positions = data.get("positions") or {}
    merged_positions = {}
    total_position_pnl = 0.0

    for symbol, pos in raw_positions.items():
        qty = pos.get("qty", 0)
        if abs(qty) < 1e-6:
            continue

        norm_sym = _normalize_symbol(symbol)
        live_price = price_cache.get(symbol, 0) or broker_price_map.get(symbol, 0)
        avg_price = pos.get("avg_price", 0)

        pnl = 0.0
        pnl_pct = 0.0
        if live_price > 0 and avg_price > 0:
            if qty > 0:
                pnl = (live_price - avg_price) * qty
                pnl_pct = (live_price / avg_price - 1) * 100
            else:
                pnl = (avg_price - live_price) * abs(qty)
                pnl_pct = (avg_price / live_price - 1) * 100 if live_price > 0 else 0.0
            total_position_pnl += pnl

        # Merge or create
        if norm_sym in merged_positions:
            m_pos = merged_positions[norm_sym]
            m_pos["qty"] = round(m_pos.get("qty", 0) + qty, 8)
            m_pos["pnl"] = round(m_pos.get("pnl", 0) + pnl, 2)
            # Simple avg entry for merged
            old_qty = m_pos.get("qty", 0) - qty
            if abs(m_pos["qty"]) > 1e-10:
                m_pos["avg_price"] = (old_qty * m_pos.get("avg_price", 0) + qty * avg_price) / m_pos["qty"]
        else:
            merged_positions[norm_sym] = {
                "symbol": norm_sym,
                "qty": qty,
                "side": pos.get("side", "LONG"),
                "avg_price": avg_price,
                "entry": avg_price,
                "current": live_price,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "signal": pos.get("signal", 0),
                "signal_direction": pos.get("signal_direction", "UNKNOWN")
            }

    data["positions"] = merged_positions
    data["open_positions"] = len(merged_positions)

    # Update daily_pnl / total_pnl to include unrealized P&L
    daily_source = str(data.get("daily_pnl_source", "")).strip().lower()
    has_broker_truth = daily_source == "broker_fills" or "daily_pnl_realized" in data
    if has_broker_truth:
        # Engine has broker-truth realized fills; add live unrealized on top
        realized = float(data.get("daily_pnl_realized", data.get("daily_pnl", 0)) or 0)
        combined = round(realized + total_position_pnl, 2)
        if combined != 0:
            data["daily_pnl"] = combined
            data["total_pnl"] = combined
    elif total_position_pnl != 0:
        # No broker truth — use position P&L as approximation
        data["daily_pnl"] = round(total_position_pnl, 2)
        data["total_pnl"] = round(total_position_pnl, 2)

    # Normalize engine field names before sanitizing
    data = _alias_engine_state_fields(data)

    return _sanitize_floats(data)


def _filter_state_by_session(state: Dict, session_type: str) -> Dict:
    """Filter a unified trading state to only include positions/signals for a session."""
    filtered = dict(state)
    positions = state.get("positions", {})
    filtered_positions = {}

    # Compute session capital from broker_positions partition
    broker_positions = state.get("broker_positions", [])
    session_equity = 0.0
    session_unrl_pnl = 0.0
    session_pos_count = 0

    for symbol, data in positions.items():
        is_crypto = _is_crypto_symbol(symbol)
        norm_sym = _normalize_symbol(symbol)

        if session_type == "core" and not is_crypto:
            filtered_positions[norm_sym] = data
        elif session_type == "crypto" and is_crypto:
            filtered_positions[norm_sym] = data

    # Partition broker equity by session
    if isinstance(broker_positions, list):
        for bp in broker_positions:
            sym = str(bp.get("symbol", "") or bp.get("normalized_symbol", ""))
            is_crypto = _is_crypto_symbol(sym)
            mv = float(bp.get("market_value", 0) or 0)
            upl = float(bp.get("unrealized_pl", 0) or 0)
            if session_type == "core" and not is_crypto:
                session_equity += mv
                session_unrl_pnl += upl
                session_pos_count += 1
            elif session_type == "crypto" and is_crypto:
                session_equity += mv
                session_unrl_pnl += upl
                session_pos_count += 1

    # Inject session-specific financial fields
    # Use session_unrl_pnl as the basis for session-specific daily/total P&L
    total_realized_pnl = float(state.get("realized_pnl", 0) or 0)
    
    if session_type == "core":
        session_total_pnl = session_unrl_pnl + total_realized_pnl
    else:  # crypto
        session_total_pnl = session_unrl_pnl

    # ALWAYS update these fields so session-scoped views don't "leak" global totals
    filtered["capital"] = round(session_equity, 2)
    filtered["equity"] = round(session_equity, 2)
    filtered["total_pnl"] = round(session_total_pnl, 2)
    filtered["daily_pnl"] = round(session_unrl_pnl, 2)
    filtered["open_positions"] = session_pos_count
    
    # max_drawdown is generally global but we can zero it if session is inactive
    if session_equity <= 0:
        filtered["max_drawdown"] = 0.0
    else:
        filtered["max_drawdown"] = state.get("max_drawdown", 0)

    filtered["positions"] = filtered_positions
    filtered["session_type"] = session_type
    return filtered


def _alias_engine_state_fields(state: Dict) -> Dict:
    """Normalize trading-engine field names to the API's expected names.

    The live engine writes these fields to trading_state.json:
      equity          -> capital
      drawdown        -> max_drawdown
      realized_pnl    -> used to derive total_pnl
      unrealized_pnl  -> used to derive total_pnl

    sanitize_execution_metrics() reads 'capital', 'max_drawdown', 'total_pnl',
    'sharpe_ratio' etc. Without aliasing, those fall back to 0.
    """
    out = dict(state)

    # capital
    if "capital" not in out or out["capital"] is None:
        if "equity" in out and out["equity"] is not None:
            out["capital"] = out["equity"]

    # max_drawdown (engine writes fractional drawdown)
    if "max_drawdown" not in out or out["max_drawdown"] is None:
        if "drawdown" in out and out["drawdown"] is not None:
            out["max_drawdown"] = out["drawdown"]

    # total_pnl
    if "total_pnl" not in out or out["total_pnl"] is None:
        realized = float(out.get("realized_pnl", 0) or 0)
        unrealized = float(out.get("unrealized_pnl", 0) or 0)
        if realized != 0 or unrealized != 0:
            out["total_pnl"] = round(realized + unrealized, 2)

    # daily_pnl — use today's realized + unrealized if not already set correctly
    existing_daily = out.get("daily_pnl")
    if existing_daily is None or abs(float(existing_daily or 0)) < 0.01:
        realized = float(out.get("realized_pnl", 0) or 0)
        unrealized = float(out.get("unrealized_pnl", 0) or 0)
        if realized != 0 or unrealized != 0:
            out["daily_pnl"] = round(realized + unrealized, 2)

    return out


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if "+" not in ts and "-" not in ts: # Naive assumed UTC
            return datetime.fromisoformat(ts + "+00:00")
        return datetime.fromisoformat(ts)
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
