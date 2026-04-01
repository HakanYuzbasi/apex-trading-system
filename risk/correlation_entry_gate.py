"""
risk/correlation_entry_gate.py — Correlation-Aware Entry Gate

Prevents entering a new position whose recent returns are highly correlated
with the current open book, thereby avoiding hidden concentration risk.

Algorithm:
1. Compute a short rolling-return series for the candidate symbol.
2. For each open position, compute Pearson correlation with the candidate.
3. If the MAX correlation across all open positions exceeds the configurable
   threshold, block the entry and return a reason string.
4. Also tracks a portfolio average correlation metric for monitoring.

Config keys (all on ApexConfig, with defaults below):
    CORR_ENTRY_GATE_ENABLED            = True
    CORR_ENTRY_MAX_CORR                = 0.65   # block if any open pos corr > this
    CORR_ENTRY_SOFT_CORR               = 0.50   # warn + penalise confidence
    CORR_ENTRY_CONF_PENALTY            = 0.05   # subtract from confidence when soft
    CORR_ENTRY_LOOKBACK                = 30     # bars of returns to compare
    CORR_ENTRY_MIN_BARS                = 10     # min bars required to gate
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEF = {
    "CORR_ENTRY_GATE_ENABLED":   True,
    "CORR_ENTRY_MAX_CORR":       0.65,
    "CORR_ENTRY_SOFT_CORR":      0.50,
    "CORR_ENTRY_CONF_PENALTY":   0.05,
    "CORR_ENTRY_LOOKBACK":       30,
    "CORR_ENTRY_MIN_BARS":       10,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


def _returns(prices: pd.Series, n: int) -> Optional[np.ndarray]:
    """Last-n log returns from a Close price series. Returns None if insufficient."""
    closes = prices.dropna()
    if len(closes) < n + 1:
        return None
    tail = closes.iloc[-(n + 1):]
    return np.diff(np.log(tail.values.astype(float)))


def check_correlation_gate(
    candidate_symbol: str,
    candidate_prices: pd.DataFrame,
    open_positions: Dict[str, int],          # symbol → qty
    historical_data: Dict[str, pd.DataFrame], # symbol → OHLCV dataframe
    confidence: float = 1.0,
) -> Tuple[bool, str, float]:
    """
    Check whether entering candidate_symbol would add excessive correlation risk.

    Returns:
        (blocked: bool, reason: str, adjusted_confidence: float)
        - blocked=True  → don't enter
        - blocked=False, reason non-empty → soft warning, confidence penalised
        - blocked=False, reason=''       → all clear
    """
    if not _cfg("CORR_ENTRY_GATE_ENABLED"):
        return False, "", confidence

    # Crypto assets have structural intra-class correlation >0.65 — this gate
    # is calibrated for equities only. Applying it to crypto blocks ALL entries.
    if candidate_symbol.startswith("CRYPTO:") or "/" in candidate_symbol:
        return False, "", confidence

    if not open_positions:
        return False, "", confidence

    lookback  = int(_cfg("CORR_ENTRY_LOOKBACK"))
    min_bars  = int(_cfg("CORR_ENTRY_MIN_BARS"))
    hard_thresh = float(_cfg("CORR_ENTRY_MAX_CORR"))
    soft_thresh = float(_cfg("CORR_ENTRY_SOFT_CORR"))
    penalty     = float(_cfg("CORR_ENTRY_CONF_PENALTY"))

    # Candidate returns
    if candidate_prices is None or "Close" not in candidate_prices.columns:
        return False, "", confidence
    cand_ret = _returns(candidate_prices["Close"], lookback)
    if cand_ret is None or len(cand_ret) < min_bars:
        return False, "", confidence

    max_corr = 0.0
    max_corr_sym = ""

    for sym, qty in open_positions.items():
        if qty == 0 or sym == candidate_symbol:
            continue
        hist = historical_data.get(sym)
        if hist is None or "Close" not in hist.columns:
            continue
        pos_ret = _returns(hist["Close"], lookback)
        if pos_ret is None or len(pos_ret) < min_bars:
            continue

        # Align lengths
        n = min(len(cand_ret), len(pos_ret))
        if n < min_bars:
            continue

        c = cand_ret[-n:]
        p = pos_ret[-n:]
        if np.std(c) < 1e-9 or np.std(p) < 1e-9:
            continue  # constant price → skip

        corr = float(np.corrcoef(c, p)[0, 1])
        if abs(corr) > max_corr:
            max_corr = abs(corr)
            max_corr_sym = sym

    if max_corr >= hard_thresh:
        reason = (
            f"CorrGate: {candidate_symbol} corr={max_corr:.2f} "
            f"with {max_corr_sym} >= {hard_thresh:.2f} threshold — blocked"
        )
        logger.warning(
            "🚫 CorrGate BLOCK %s: corr=%.2f with %s (threshold=%.2f)",
            candidate_symbol, max_corr, max_corr_sym, hard_thresh,
        )
        return True, reason, confidence

    if max_corr >= soft_thresh:
        adj_conf = max(0.0, confidence - penalty)
        reason = (
            f"CorrGate: {candidate_symbol} corr={max_corr:.2f} "
            f"with {max_corr_sym} (soft warn, conf -{penalty:.2f})"
        )
        logger.info(
            "⚠️  CorrGate SOFT %s: corr=%.2f with %s — conf penalised %.2f→%.2f",
            candidate_symbol, max_corr, max_corr_sym, confidence, adj_conf,
        )
        return False, reason, adj_conf

    return False, "", confidence
