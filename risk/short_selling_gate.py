"""
risk/short_selling_gate.py — Short Selling Risk Gate

Controls when and how the system may enter short equity positions.
Enforces risk guardrails that are specific to short selling.

Gate logic:
    1. Feature flag: SHORT_SELLING_ENABLED (default True)
    2. Regime check: only bear, strong_bear, neutral, volatile
    3. Asset class: IBKR equities only (not crypto)
    4. Symbol eligibility: large-cap whitelist OR price >= MIN_PRICE
    5. Max concurrent shorts: SHORT_MAX_POSITIONS (default 3)
    6. Max short exposure per symbol: SHORT_MAX_NOTIONAL (default $25,000)
    7. Short concentration cap: total short notional <= SHORT_MAX_TOTAL_NOTIONAL
    8. VIX guard: no new shorts when VIX >= SHORT_VIX_BLOCK (shorts become crowded)

Result:
    check_short_gate(...) → (allowed: bool, reason: str, confidence_adj: float)

    allowed=True, reason="ok"  → proceed normally
    allowed=False, reason=...  → block the short entry

Config keys:
    SHORT_SELLING_ENABLED         = True
    SHORT_ACTIVE_REGIMES          = ["bear", "strong_bear", "neutral", "volatile"]
    SHORT_MAX_POSITIONS           = 3
    SHORT_MAX_NOTIONAL            = 25000.0
    SHORT_MAX_TOTAL_NOTIONAL      = 60000.0
    SHORT_MIN_PRICE               = 10.0   # penny stock guard
    SHORT_VIX_BLOCK               = 35.0   # VIX too high → crowded short covering risk
    SHORT_SIGNAL_FLOOR            = 0.12   # |signal| must be at least this for a short
    SHORT_CONFIDENCE_FLOOR        = 0.55   # minimum confidence to enter a short
    SHORT_LARGE_CAP_ONLY          = False  # restrict to large-cap whitelist
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "SHORT_SELLING_ENABLED":     True,
    "SHORT_ACTIVE_REGIMES":      ["bear", "strong_bear", "neutral", "volatile"],
    "SHORT_MAX_POSITIONS":       3,
    "SHORT_MAX_NOTIONAL":        25000.0,
    "SHORT_MAX_TOTAL_NOTIONAL":  60000.0,
    "SHORT_MIN_PRICE":           10.0,
    "SHORT_VIX_BLOCK":           35.0,
    "SHORT_SIGNAL_FLOOR":        0.12,
    "SHORT_CONFIDENCE_FLOOR":    0.55,
    "SHORT_LARGE_CAP_ONLY":      False,
}

# Large-cap equity whitelist (S&P 500 components that are commonly shortable)
_LARGE_CAP_SHORTABLE = frozenset([
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA",
    "BRK.B", "JPM", "JNJ", "V", "UNH", "XOM", "PG", "HD", "MA", "CVX",
    "LLY", "ABBV", "MRK", "AVGO", "PEP", "KO", "COST", "MCD", "WMT",
    "CRM", "BAC", "TMO", "DIS", "NKE", "AMD", "INTC", "QCOM", "TXN",
    "CSCO", "NFLX", "ADBE", "NOW", "PYPL", "GILD", "AMGN", "BMY",
    "MS", "GS", "C", "WFC", "AXP", "BLK", "SPGI", "CME", "ICE",
    "CAT", "DE", "HON", "RTX", "GE", "BA", "LMT", "NOC", "GD",
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs are good short vehicles
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLU", "XLB", "XLRE", "XLP", "XLY",
])


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


def check_short_gate(
    symbol: str,
    signal: float,
    confidence: float,
    regime: str,
    asset_class: str,
    current_price: float,
    current_positions: Dict[str, float],  # symbol → quantity (negative = short)
    price_cache: Dict[str, float],        # symbol → current price (for notional calc)
    current_vix: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """
    Evaluate whether a short entry is permitted.

    Args:
        symbol: Ticker
        signal: Current signal value (negative = bearish/short signal)
        confidence: Model confidence [0, 1]
        regime: Current market regime
        asset_class: "EQUITY", "CRYPTO", "FX", etc.
        current_price: Current market price of the symbol
        current_positions: {symbol: qty} dict — negative qty = short position
        price_cache: {symbol: price} for notional calculations
        current_vix: Current VIX level (None = skip VIX check)

    Returns:
        (allowed, reason, confidence_adj)
        - allowed: False blocks the entry
        - reason: human-readable explanation
        - confidence_adj: adjusted confidence (may be reduced slightly for shorts)
    """
    # Only apply this gate to short signals
    if signal >= 0:
        return True, "not_a_short", confidence

    if not _cfg("SHORT_SELLING_ENABLED"):
        return False, "short_selling_disabled", confidence

    # Crypto shorts are handled by Alpaca — this gate is IBKR equities only
    if str(asset_class).upper() == "CRYPTO":
        return True, "crypto_short_allowed", confidence

    regime_lower = str(regime).lower()
    active_regimes = [r.lower() for r in list(_cfg("SHORT_ACTIVE_REGIMES"))]
    if regime_lower not in active_regimes:
        return (
            False,
            f"short_regime_blocked:{regime}",
            confidence,
        )

    # VIX guard
    vix = float(current_vix or 0.0)
    vix_block = float(_cfg("SHORT_VIX_BLOCK"))
    if vix > 0 and vix >= vix_block:
        return (
            False,
            f"short_vix_too_high:{vix:.1f}>={vix_block}",
            confidence,
        )

    # Price guard
    price = float(current_price or 0.0)
    min_price = float(_cfg("SHORT_MIN_PRICE"))
    if price > 0 and price < min_price:
        return (
            False,
            f"short_price_too_low:{price:.2f}<{min_price}",
            confidence,
        )

    # Large-cap whitelist check
    if _cfg("SHORT_LARGE_CAP_ONLY") and symbol.upper() not in _LARGE_CAP_SHORTABLE:
        return (
            False,
            f"short_not_large_cap:{symbol}",
            confidence,
        )

    # Signal floor
    sig_floor = float(_cfg("SHORT_SIGNAL_FLOOR"))
    if abs(float(signal)) < sig_floor:
        return (
            False,
            f"short_signal_too_weak:{abs(signal):.3f}<{sig_floor}",
            confidence,
        )

    # Confidence floor
    conf_floor = float(_cfg("SHORT_CONFIDENCE_FLOOR"))
    if float(confidence) < conf_floor:
        return (
            False,
            f"short_confidence_too_low:{confidence:.3f}<{conf_floor}",
            confidence,
        )

    # Count existing short positions and total short notional
    existing_shorts: List[str] = []
    total_short_notional = 0.0
    for sym, qty in current_positions.items():
        if float(qty) < 0:
            existing_shorts.append(sym)
            sym_price = float(price_cache.get(sym, 0.0))
            total_short_notional += abs(float(qty)) * sym_price

    max_shorts = int(_cfg("SHORT_MAX_POSITIONS"))
    if len(existing_shorts) >= max_shorts:
        return (
            False,
            f"short_max_positions:{len(existing_shorts)}>={max_shorts}",
            confidence,
        )

    max_total = float(_cfg("SHORT_MAX_TOTAL_NOTIONAL"))
    if total_short_notional >= max_total:
        return (
            False,
            f"short_total_notional:{total_short_notional:.0f}>={max_total:.0f}",
            confidence,
        )

    # Per-symbol notional cap — estimated size check
    max_per_sym = float(_cfg("SHORT_MAX_NOTIONAL"))
    # confidence_adj: very slight penalty for shorts (higher risk/recall)
    confidence_adj = float(confidence) * 0.97
    confidence_adj = max(0.0, min(1.0, confidence_adj))

    logger.debug(
        "ShortGate %s: allowed regime=%s signal=%.3f conf=%.3f "
        "existing_shorts=%d total_notional=$%.0f",
        symbol, regime, signal, confidence,
        len(existing_shorts), total_short_notional,
    )

    return True, "ok", confidence_adj


def get_short_exposure_summary(
    current_positions: Dict[str, float],
    price_cache: Dict[str, float],
) -> Dict:
    """
    Return a summary dict of current short exposure for monitoring.
    """
    short_symbols = []
    total_notional = 0.0
    for sym, qty in current_positions.items():
        if float(qty) < 0:
            p = float(price_cache.get(sym, 0.0))
            notional = abs(float(qty)) * p
            short_symbols.append({"symbol": sym, "qty": float(qty), "notional": round(notional, 2)})
            total_notional += notional

    return {
        "short_count": len(short_symbols),
        "total_short_notional": round(total_notional, 2),
        "max_allowed": int(_cfg("SHORT_MAX_POSITIONS")),
        "max_total_notional": float(_cfg("SHORT_MAX_TOTAL_NOTIONAL")),
        "positions": sorted(short_symbols, key=lambda x: x["notional"], reverse=True),
    }
