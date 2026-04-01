"""
models/bear_regime_strategies.py — Bear & Neutral Regime Alpha Strategies

Provides two complementary signals that generate edge in non-bull regimes:

1. MeanReversionSignal
   - VWAP z-score + RSI extremes → fade overextended moves
   - Only active in: bear, strong_bear, neutral, volatile
   - Signal range: [-1.0, +1.0]

2. SectorRotationSignal
   - Measures relative strength across sector ETFs vs SPY
   - Goes long the strongest sector, short (or avoids) the weakest
   - Active in all regimes but highest weight in bear/neutral
   - Signal range: [-1.0, +1.0] where +1 = top-ranked sector

Both are designed as add-on blends, not standalone strategies.

Config keys (ApexConfig, with defaults):
    BEAR_MR_ENABLED              = True
    BEAR_MR_BLEND_WEIGHT         = 0.10   # % weight in final signal blend
    BEAR_MR_RSI_OVERSOLD         = 32
    BEAR_MR_RSI_OVERBOUGHT       = 68
    BEAR_MR_VWAP_ZSCORE_THRESH   = 1.5    # |z| above this = overextended
    BEAR_MR_ACTIVE_REGIMES       = ["bear","strong_bear","neutral","volatile"]

    SECTOR_ROTATION_ENABLED      = True
    SECTOR_ROTATION_BLEND_WEIGHT = 0.08
    SECTOR_ROTATION_LOOKBACK     = 20     # bars for relative momentum
    SECTOR_ROTATION_TOP_N        = 3      # top-N sectors to go long
    SECTOR_ETFS                  = [XLK, XLF, XLE, XLV, XLI, XLC, XLU, XLB, XLRE, XLP, XLY]
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple  # noqa: F401

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default config ────────────────────────────────────────────────────────────

_DEF: Dict = {
    "BEAR_MR_ENABLED":              True,
    "BEAR_MR_BLEND_WEIGHT":         0.10,
    "BEAR_MR_RSI_OVERSOLD":         32,
    "BEAR_MR_RSI_OVERBOUGHT":       68,
    "BEAR_MR_VWAP_ZSCORE_THRESH":   1.5,
    "BEAR_MR_ACTIVE_REGIMES":       ["bear", "strong_bear", "neutral", "volatile"],
    "SECTOR_ROTATION_ENABLED":      True,
    "SECTOR_ROTATION_BLEND_WEIGHT": 0.08,
    "SECTOR_ROTATION_LOOKBACK":     20,
    "SECTOR_ROTATION_TOP_N":        3,
    "SECTOR_ETFS": [
        "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLC", "XLU", "XLB", "XLRE", "XLP", "XLY",
    ],
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


# ── Helper functions ──────────────────────────────────────────────────────────

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute most-recent RSI value. Returns 50.0 on insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss < 1e-10:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _vwap_zscore(df: pd.DataFrame, period: int = 20) -> float:
    """
    Compute z-score of current price relative to VWAP over last `period` bars.
    Falls back to 0.0 if Volume is missing or insufficient data.
    """
    if len(df) < period:
        return 0.0
    tail = df.tail(period)
    if "Volume" not in tail.columns or tail["Volume"].sum() < 1:
        # No volume data — fall back to price z-score
        closes = tail["Close"].values.astype(float)
        mean = closes.mean()
        std = closes.std()
        if std < 1e-9:
            return 0.0
        return float((closes[-1] - mean) / std)
    typical = (tail["High"] + tail["Low"] + tail["Close"]) / 3.0
    vwap = (typical * tail["Volume"]).sum() / tail["Volume"].sum()
    vwap_prices = typical.values.astype(float)
    std = vwap_prices.std()
    if std < 1e-9:
        return 0.0
    return float((float(tail["Close"].iloc[-1]) - float(vwap)) / std)


def _momentum(closes: np.ndarray, lookback: int) -> float:
    """Percentage return over lookback bars. Returns 0.0 on insufficient data."""
    if len(closes) < lookback + 1:
        return 0.0
    start = float(closes[-(lookback + 1)])
    end   = float(closes[-1])
    if start < 1e-9:
        return 0.0
    return (end - start) / start


# ── Mean Reversion Signal ─────────────────────────────────────────────────────

class MeanReversionSignal:
    """
    Generates a contrarian mean-reversion signal for bear/neutral regimes.

    Logic:
    - If RSI < oversold AND VWAP z-score < -threshold → buy signal (+)
    - If RSI > overbought AND VWAP z-score > +threshold → sell signal (-)
    - Signal magnitude scales with how extreme the readings are
    - Returns 0.0 outside active regimes (no interference in bull markets)
    """

    def get_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime: str = "neutral",
    ) -> float:
        """
        Compute mean-reversion signal for `symbol`.

        Args:
            symbol: ticker symbol (for logging)
            df: OHLCV dataframe with Close (and optionally High/Low/Volume)
            regime: current market regime string

        Returns:
            float in [-1.0, 1.0]: positive = buy the dip, negative = short the rip
        """
        if not _cfg("BEAR_MR_ENABLED"):
            return 0.0

        active_regimes = _cfg("BEAR_MR_ACTIVE_REGIMES")
        if regime.lower() not in [r.lower() for r in active_regimes]:
            return 0.0

        if df is None or "Close" not in df.columns or len(df) < 16:
            return 0.0

        closes = df["Close"].values.astype(float)
        rsi = _rsi(closes, period=14)
        vz  = _vwap_zscore(df, period=20)

        rsi_os  = float(_cfg("BEAR_MR_RSI_OVERSOLD"))
        rsi_ob  = float(_cfg("BEAR_MR_RSI_OVERBOUGHT"))
        vz_thresh = float(_cfg("BEAR_MR_VWAP_ZSCORE_THRESH"))

        signal = 0.0

        # Buy the dip: oversold RSI + price below VWAP
        if rsi < rsi_os and vz < -vz_thresh:
            rsi_score = (rsi_os - rsi) / rsi_os          # 0→1
            vz_score  = min(abs(vz) / (vz_thresh * 2), 1.0)
            signal = +(rsi_score + vz_score) / 2.0

        # Sell the rip: overbought RSI + price above VWAP
        elif rsi > rsi_ob and vz > vz_thresh:
            rsi_score = (rsi - rsi_ob) / (100.0 - rsi_ob)
            vz_score  = min(abs(vz) / (vz_thresh * 2), 1.0)
            signal = -(rsi_score + vz_score) / 2.0

        signal = float(np.clip(signal, -1.0, 1.0))

        if abs(signal) > 0.01:
            logger.debug(
                "MeanReversion %s [%s]: rsi=%.1f vz=%.2f → signal=%+.3f",
                symbol, regime, rsi, vz, signal,
            )

        return signal


# ── Sector Rotation Signal ────────────────────────────────────────────────────

class SectorRotationSignal:
    """
    Ranks sector ETFs by recent momentum and scores a candidate symbol based
    on whether its sector is among the leaders or laggards.

    The signal is used as a blend weight modifier (not standalone):
    +1.0 = top-ranked sector (boost long bias)
    -1.0 = bottom-ranked sector (boost short / avoid bias)
     0.0 = mid-ranked or no sector data available
    """

    # Simplified ticker→sector mapping (covers SPDR ETF universe)
    _SECTOR_MAP: Dict[str, str] = {
        "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "GOOGL": "XLK",
        "META": "XLC", "NFLX": "XLC", "AMZN": "XLC",
        "JPM": "XLF", "GS": "XLF", "BAC": "XLF", "BRK.B": "XLF",
        "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
        "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV",
        "CAT": "XLI", "BA": "XLI", "RTX": "XLI", "HON": "XLI",
        "NEE": "XLU", "DUK": "XLU", "SO": "XLU",
        "WMT": "XLP", "KO": "XLP", "PG": "XLP", "COST": "XLP",
        "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
        "LIN": "XLB", "APD": "XLB",
        "AMT": "XLRE", "PLD": "XLRE",
        # Sector ETFs map to themselves
        "XLK": "XLK", "XLF": "XLF", "XLE": "XLE", "XLV": "XLV",
        "XLI": "XLI", "XLC": "XLC", "XLU": "XLU", "XLB": "XLB",
        "XLRE": "XLRE", "XLP": "XLP", "XLY": "XLY",
        # SPY/QQQ = no specific sector
        "SPY": None, "QQQ": None,
    }

    def get_signal(
        self,
        symbol: str,
        historical_data: Dict[str, pd.DataFrame],
        regime: str = "neutral",
    ) -> float:
        """
        Score a symbol based on its sector's relative momentum rank.

        Returns float in [-1.0, 1.0].
        """
        if not _cfg("SECTOR_ROTATION_ENABLED"):
            return 0.0

        sector_etfs: List[str] = list(_cfg("SECTOR_ETFS"))
        lookback = int(_cfg("SECTOR_ROTATION_LOOKBACK"))
        top_n    = int(_cfg("SECTOR_ROTATION_TOP_N"))

        # Find candidate's sector
        candidate_sector = self._SECTOR_MAP.get(symbol.upper())
        if candidate_sector is None:
            return 0.0  # SPY / unknown → no sector bias

        # Compute momentum for all available sector ETFs
        sector_scores: Dict[str, float] = {}
        for etf in sector_etfs:
            df = historical_data.get(etf)
            if df is None or "Close" not in df.columns:
                continue
            closes = df["Close"].values.astype(float)
            mom = _momentum(closes, lookback)
            sector_scores[etf] = mom

        if len(sector_scores) < 2:
            return 0.0  # not enough sectors to rank

        # Rank sectors by momentum (highest = rank 0)
        ranked = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        n_sectors = len(ranked)
        ranks = {etf: i for i, (etf, _) in enumerate(ranked)}

        rank = ranks.get(candidate_sector)
        if rank is None:
            return 0.0  # candidate's sector not in ranked list

        # Top-N sectors → positive signal; bottom-N → negative; middle → 0
        bottom_n = max(1, n_sectors - top_n)
        if rank < top_n:
            # Top sector: scale from +1.0 (rank=0) to +0.2 (rank=top_n-1)
            signal = 1.0 - (rank / top_n) * 0.8
        elif rank >= bottom_n:
            # Bottom sector: scale from -0.2 to -1.0
            pos_from_bottom = n_sectors - 1 - rank
            signal = -(1.0 - (pos_from_bottom / top_n) * 0.8)
        else:
            signal = 0.0

        signal = float(np.clip(signal, -1.0, 1.0))

        logger.debug(
            "SectorRotation %s [%s]: sector=%s rank=%d/%d mom=%.2f%% → signal=%+.3f",
            symbol, regime, candidate_sector, rank, n_sectors,
            sector_scores.get(candidate_sector, 0) * 100, signal,
        )

        return signal


# ── Module-level singletons ───────────────────────────────────────────────────

_mean_reversion = MeanReversionSignal()
_sector_rotation = SectorRotationSignal()


def get_bear_regime_blend(
    symbol: str,
    df: pd.DataFrame,
    historical_data: Dict[str, pd.DataFrame],
    regime: str,
    base_signal: float,
    mr_weight_override: Optional[float] = None,
    sr_weight_override: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute blended bear/neutral regime alpha and return adjusted signal.

    Args:
        symbol: ticker
        df: symbol OHLCV dataframe
        historical_data: full historical_data dict (for sector ETFs)
        regime: current regime string
        base_signal: existing signal before blend
        mr_weight_override: optional adaptive weight for mean_reversion
        sr_weight_override: optional adaptive weight for sector_rotation

    Returns:
        (blended_signal, component_dict)
    """
    mr_weight = float(mr_weight_override if mr_weight_override is not None else _cfg("BEAR_MR_BLEND_WEIGHT"))
    sr_weight = float(sr_weight_override if sr_weight_override is not None else _cfg("SECTOR_ROTATION_BLEND_WEIGHT"))

    mr_signal = _mean_reversion.get_signal(symbol, df, regime)
    sr_signal = _sector_rotation.get_signal(symbol, historical_data, regime)

    # Weighted blend on top of base signal
    total_weight = 1.0 + mr_weight + sr_weight
    blended = (base_signal + mr_weight * mr_signal + sr_weight * sr_signal) / total_weight
    blended = float(np.clip(blended, -1.0, 1.0))

    components = {
        "mean_reversion": round(mr_signal, 4),
        "sector_rotation": round(sr_signal, 4),
        "blended": round(blended, 4),
    }

    return blended, components
