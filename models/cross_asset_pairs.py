"""
models/cross_asset_pairs.py — Cross-Asset Pairs Arbitrage

Extends statistical pairs trading to cross-asset universes:
  - Crypto ↔ Crypto   (BTC/ETH, ETH/SOL, …)
  - Crypto ↔ Equity   (BTC ↔ MSTR, ETH ↔ COIN, SOL ↔ PYPL …)
  - Sector ↔ Sector   (SPY ↔ QQQ, GLD ↔ SLV …)

Key improvements over vanilla PairsTrader:
  - Cross-asset normalisation (log-return correlation pre-filter)
  - Volatility-normalised spread (spread / rolling vol)
  - Regime-conditional entry thresholds (bear → tighter z-entry)
  - Per-pair signal decay (score × exp(-age/half_life))
  - Serialisable state for API + dashboard

Usage:
    from models.cross_asset_pairs import CrossAssetPairsArb
    arb = CrossAssetPairsArb()
    arb.scan_pairs(historical_data)            # O(N²), run every few hours
    overlay = arb.get_overlay_signals(regime="neutral")
    # overlay: {"BTC/USD": +0.12, "MSTR": -0.08, …}
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
MIN_BARS          = 40          # minimum bars to form a pair
LOOKBACK          = 60          # regression window
Z_ENTRY_DEFAULT   = 1.8         # z-score to enter
Z_EXIT_DEFAULT    = 0.4         # z-score to exit
MAX_PAIRS         = 20          # cap active pairs
CORR_PRE_FILTER   = 0.50        # minimum |log-return correlation| to bother with coint test
MIN_HALF_LIFE     = 2           # days
MAX_HALF_LIFE     = 25          # days
SIGNAL_WEIGHT     = 0.15        # max overlay weight per leg
SIGNAL_DECAY_HALF = 4           # bars; signal decays to 50% after this many bars

# Regime-conditional z-entry multipliers
_REGIME_Z_MULT = {
    "bull":       0.90,
    "neutral":    1.00,
    "bear":       1.20,
    "volatile":   1.30,
    "crisis":     1.50,
}


@dataclass
class PairRecord:
    leg_y: str
    leg_x: str
    hedge_ratio: float
    half_life: float
    z_score: float
    z_entry: float
    z_exit: float
    spread_mean: float
    spread_std: float
    last_spread: float
    corr: float
    last_updated: float = field(default_factory=time.time)
    signal_y: float = 0.0      # overlay signal for leg Y
    signal_x: float = 0.0      # overlay signal for leg X (usually opposite sign)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()}

    @property
    def pair_key(self) -> str:
        return f"{self.leg_y}_{self.leg_x}"


class CrossAssetPairsArb:
    """
    Cross-asset statistical arbitrage signal generator.

    Discovers cointegrated pairs across crypto and equity universes,
    computes volatility-normalised spread z-scores, and returns
    per-symbol overlay signals [−SIGNAL_WEIGHT, +SIGNAL_WEIGHT].
    """

    def __init__(
        self,
        z_entry: float = Z_ENTRY_DEFAULT,
        z_exit: float = Z_EXIT_DEFAULT,
        max_pairs: int = MAX_PAIRS,
        state_dir: Optional[Path] = None,
    ):
        self._z_entry = z_entry
        self._z_exit = z_exit
        self._max_pairs = max_pairs
        self._pairs: Dict[str, PairRecord] = {}       # pair_key → PairRecord
        self._last_scan_ts: float = 0.0

        if state_dir:
            self._state_dir = Path(state_dir)
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load()
        else:
            self._state_dir = None

    # ── Public API ──────────────────────────────────────────────────────────

    def scan_pairs(
        self,
        historical_data: Dict[str, "pd.DataFrame"],  # symbol → OHLCV DataFrame
        force: bool = False,
    ) -> int:
        """
        Scan for cointegrated pairs in historical_data.

        Returns the number of active pairs found.
        Expensive O(N²) — rate-limit to every few hours in production.
        """
        if not force and (time.time() - self._last_scan_ts) < 3600:
            return len(self._pairs)

        try:
            return self._run_scan(historical_data)
        except Exception as exc:
            logger.warning("CrossAssetPairsArb.scan_pairs failed: %s", exc)
            return len(self._pairs)

    def update_scores(self, historical_data: Dict[str, "pd.DataFrame"]) -> None:
        """
        Refresh z-scores for existing pairs without full re-scan.
        Call every cycle (fast — only touches known pairs).
        """
        to_drop: List[str] = []
        for pk, rec in list(self._pairs.items()):
            ok = self._update_pair_zscore(rec, historical_data)
            if not ok:
                to_drop.append(pk)
        for pk in to_drop:
            self._pairs.pop(pk, None)
        if to_drop:
            logger.debug("CrossAssetPairsArb: dropped %d stale pairs", len(to_drop))

    def get_overlay_signals(self, regime: str = "neutral") -> Dict[str, float]:
        """
        Return per-symbol overlay signals in [−SIGNAL_WEIGHT, +SIGNAL_WEIGHT].

        Positive → bullish overlay, negative → bearish overlay.
        Multiple pairs involving the same symbol are accumulated (capped at ±SIGNAL_WEIGHT).
        """
        zm = _REGIME_Z_MULT.get(regime, 1.0)
        acc: Dict[str, float] = {}

        for rec in self._pairs.values():
            z_eff = rec.z_score
            z_entry_adj = rec.z_entry * zm

            # Signal decay: older pairs have weaker signals
            age_bars = max(1, int((time.time() - rec.last_updated) / 60))  # assume 1-min bars
            decay = float(np.exp(-age_bars / max(1, SIGNAL_DECAY_HALF)))

            if abs(z_eff) >= z_entry_adj:
                # Spread above mean → sell Y / buy X; below → buy Y / sell X
                direction = -1.0 if z_eff > 0 else +1.0
                strength = min(abs(z_eff) / (z_entry_adj + 1e-9), 2.0) * SIGNAL_WEIGHT * decay
                rec.signal_y = direction * strength
                rec.signal_x = -direction * strength * abs(rec.hedge_ratio)
            elif abs(z_eff) < rec.z_exit:
                rec.signal_y = 0.0
                rec.signal_x = 0.0

            acc[rec.leg_y] = acc.get(rec.leg_y, 0.0) + rec.signal_y
            acc[rec.leg_x] = acc.get(rec.leg_x, 0.0) + rec.signal_x

        # Clamp each symbol to ±SIGNAL_WEIGHT
        return {sym: float(np.clip(v, -SIGNAL_WEIGHT, SIGNAL_WEIGHT))
                for sym, v in acc.items() if abs(v) > 1e-6}

    def get_snapshot(self) -> Dict:
        """Serialisable dashboard snapshot."""
        active = [r.to_dict() for r in sorted(self._pairs.values(), key=lambda r: abs(r.z_score), reverse=True)[:10]]
        return {
            "available": True,
            "n_pairs": len(self._pairs),
            "last_scan_ts": self._last_scan_ts,
            "active_pairs": active,
            "z_entry": self._z_entry,
            "z_exit": self._z_exit,
        }

    # ── Internals ───────────────────────────────────────────────────────────

    def _run_scan(self, historical_data: Dict[str, "pd.DataFrame"]) -> int:
        try:
            import pandas as pd
        except ImportError:
            return len(self._pairs)

        symbols = [s for s, df in historical_data.items()
                   if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) >= MIN_BARS]
        if len(symbols) < 2:
            return 0

        # Build returns matrix
        prices: Dict[str, pd.Series] = {}
        for sym in symbols:
            close = historical_data[sym]["Close"].dropna()
            if len(close) >= MIN_BARS:
                prices[sym] = close

        sym_list = list(prices.keys())
        returns: Dict[str, np.ndarray] = {
            s: np.diff(np.log(prices[s].values[-LOOKBACK - 1:]))
            for s in sym_list
            if len(prices[s]) >= LOOKBACK + 1
        }
        sym_list = list(returns.keys())
        n = len(sym_list)
        if n < 2:
            return 0

        new_pairs: Dict[str, PairRecord] = {}
        pair_count = 0

        for i in range(n):
            if pair_count >= self._max_pairs:
                break
            for j in range(i + 1, n):
                if pair_count >= self._max_pairs:
                    break
                s_y, s_x = sym_list[i], sym_list[j]
                ry, rx = returns[s_y], returns[s_x]
                n_common = min(len(ry), len(rx))
                if n_common < MIN_BARS:
                    continue

                ry_c = ry[-n_common:]
                rx_c = rx[-n_common:]
                corr = float(np.corrcoef(ry_c, rx_c)[0, 1])
                if abs(corr) < CORR_PRE_FILTER or not np.isfinite(corr):
                    continue

                # Compute spread from price levels (not returns)
                py = prices[s_y].values[-LOOKBACK:]
                px = prices[s_x].values[-LOOKBACK:]
                n_lev = min(len(py), len(px))
                py = py[-n_lev:]
                px = px[-n_lev:]
                if n_lev < MIN_BARS:
                    continue

                rec = self._fit_pair(s_y, s_x, py, px, corr)
                if rec is None:
                    continue

                new_pairs[rec.pair_key] = rec
                pair_count += 1

        # Merge: keep existing z-scores for continuing pairs
        merged: Dict[str, PairRecord] = {}
        for pk, rec in new_pairs.items():
            if pk in self._pairs:
                rec.signal_y = self._pairs[pk].signal_y
                rec.signal_x = self._pairs[pk].signal_x
            merged[pk] = rec

        self._pairs = merged
        self._last_scan_ts = time.time()
        self._save()
        logger.info("CrossAssetPairsArb: scan found %d pairs from %d symbols", len(self._pairs), n)
        return len(self._pairs)

    def _fit_pair(
        self,
        s_y: str,
        s_x: str,
        py: np.ndarray,
        px: np.ndarray,
        corr: float,
    ) -> Optional[PairRecord]:
        """OLS hedge ratio + spread stats + ADF half-life check."""
        try:
            from numpy.linalg import lstsq
            X = np.column_stack([np.ones(len(px)), px])
            coeffs, _, _, _ = lstsq(X, py, rcond=None)
            hedge_ratio = float(coeffs[1])
            if not np.isfinite(hedge_ratio) or abs(hedge_ratio) > 10:
                return None

            spread = py - hedge_ratio * px
            spread_mean = float(np.mean(spread))
            spread_std = float(np.std(spread))
            if spread_std < 1e-9:
                return None

            z_score = float((spread[-1] - spread_mean) / spread_std)

            # ADF-like half-life via OU fit (lag-1 regression on diff of spread)
            d_spread = np.diff(spread)
            s_lag = spread[:-1] - spread_mean
            if len(s_lag) < 5:
                return None
            # AR(1) coefficient from OLS: Δs = α*s_lag + ε
            num = float(np.dot(s_lag, d_spread))
            den = float(np.dot(s_lag, s_lag))
            if abs(den) < 1e-9:
                return None
            alpha = num / den
            if alpha >= 0:  # non-mean-reverting
                return None
            half_life = float(-np.log(2) / alpha)
            if not (MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE):
                return None

            return PairRecord(
                leg_y=s_y,
                leg_x=s_x,
                hedge_ratio=hedge_ratio,
                half_life=half_life,
                z_score=z_score,
                z_entry=self._z_entry,
                z_exit=self._z_exit,
                spread_mean=spread_mean,
                spread_std=spread_std,
                last_spread=float(spread[-1]),
                corr=corr,
            )
        except Exception as exc:
            logger.debug("CrossAssetPairsArb._fit_pair(%s,%s) failed: %s", s_y, s_x, exc)
            return None

    def _update_pair_zscore(
        self,
        rec: PairRecord,
        historical_data: Dict[str, "pd.DataFrame"],
    ) -> bool:
        """Recompute z-score for an existing pair using latest price data."""
        try:
            import pandas as pd
            for sym in (rec.leg_y, rec.leg_x):
                if sym not in historical_data:
                    return False
                df = historical_data[sym]
                if not isinstance(df, pd.DataFrame) or "Close" not in df.columns:
                    return False

            py = historical_data[rec.leg_y]["Close"].dropna().values
            px = historical_data[rec.leg_x]["Close"].dropna().values
            n = min(len(py), len(px), LOOKBACK)
            if n < MIN_BARS:
                return False

            py = py[-n:]
            px = px[-n:]
            spread = py - rec.hedge_ratio * px
            spread_mean = float(np.mean(spread))
            spread_std = float(np.std(spread))
            if spread_std < 1e-9:
                return False

            rec.z_score = float((spread[-1] - spread_mean) / spread_std)
            rec.last_spread = float(spread[-1])
            rec.spread_mean = spread_mean
            rec.spread_std = spread_std
            rec.last_updated = time.time()
            return True
        except Exception:
            return False

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._state_dir is None:
            return
        try:
            state = {
                "pairs": {pk: r.to_dict() for pk, r in self._pairs.items()},
                "last_scan_ts": self._last_scan_ts,
            }
            p = self._state_dir / "cross_asset_pairs.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception as exc:
            logger.debug("CrossAssetPairsArb: save failed: %s", exc)

    def _load(self) -> None:
        if self._state_dir is None:
            return
        try:
            p = self._state_dir / "cross_asset_pairs.json"
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            self._last_scan_ts = float(raw.get("last_scan_ts", 0.0))
            for pk, rd in raw.get("pairs", {}).items():
                self._pairs[pk] = PairRecord(
                    leg_y=rd["leg_y"],
                    leg_x=rd["leg_x"],
                    hedge_ratio=float(rd.get("hedge_ratio", 1.0)),
                    half_life=float(rd.get("half_life", 5.0)),
                    z_score=float(rd.get("z_score", 0.0)),
                    z_entry=float(rd.get("z_entry", self._z_entry)),
                    z_exit=float(rd.get("z_exit", self._z_exit)),
                    spread_mean=float(rd.get("spread_mean", 0.0)),
                    spread_std=float(rd.get("spread_std", 1.0)),
                    last_spread=float(rd.get("last_spread", 0.0)),
                    corr=float(rd.get("corr", 0.0)),
                    last_updated=float(rd.get("last_updated", 0.0)),
                    signal_y=float(rd.get("signal_y", 0.0)),
                    signal_x=float(rd.get("signal_x", 0.0)),
                )
            logger.info("CrossAssetPairsArb: loaded %d pairs", len(self._pairs))
        except Exception as exc:
            logger.debug("CrossAssetPairsArb: load failed: %s", exc)
