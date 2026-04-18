"""
risk/black_litterman.py — Black-Litterman Portfolio Allocation

Implements the Black-Litterman model to combine:
  1. Market equilibrium (reverse-optimisation from market-cap-proxy weights)
  2. Factor IC views: signals with reliable IC form "investor views" on returns
  3. Output: posterior expected returns → MVO optimal weights

This produces position-size multipliers that over/under-weight assets
relative to equal-weight baseline, informed by:
  - Spearman IC of each signal per asset
  - Recent realized P&L as view strength
  - Regime-conditional confidence scaling

Usage:
    bl = BlackLittermanAllocator()
    weights = bl.get_weights(symbols, price_history, factor_ic_tracker)

Config keys:
    BL_ENABLED             = True
    BL_TAU                 = 0.05   # uncertainty scaling on prior
    BL_RISK_AVERSION       = 2.5    # market risk aversion coefficient δ
    BL_VIEW_CONFIDENCE     = 0.75   # default confidence in IC-derived views
    BL_MIN_IC_FOR_VIEW     = 0.10   # IC below this → no view for that factor
    BL_MIN_OBS_FOR_VIEW    = 10     # minimum IC observations for a view
    BL_WEIGHT_FLOOR        = 0.0    # minimum portfolio weight
    BL_WEIGHT_CAP          = 0.25   # maximum portfolio weight per asset
    BL_REFRESH_INTERVAL    = 50     # recompute every N cycles
    BL_PERSIST_PATH        = "data/bl_weights.json"
    BL_LOOKBACK_BARS       = 60     # bars of returns for covariance estimation
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from config import ApexConfig
except Exception:  # pragma: no cover — defensive for isolated imports
    ApexConfig = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "BL_ENABLED":           True,
    "BL_TAU":               0.05,
    "BL_RISK_AVERSION":     2.5,
    "BL_VIEW_CONFIDENCE":   0.75,
    "BL_MIN_IC_FOR_VIEW":   0.10,
    "BL_MIN_OBS_FOR_VIEW":  10,
    "BL_WEIGHT_FLOOR":      0.0,
    "BL_WEIGHT_CAP":        0.25,
    "BL_REFRESH_INTERVAL":  50,
    "BL_PERSIST_PATH":      "data/bl_weights.json",
    "BL_LOOKBACK_BARS":     60,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


# ── Core BL maths ─────────────────────────────────────────────────────────────

def _reverse_optimize(
    sigma: np.ndarray,
    weights: np.ndarray,
    delta: float,
) -> np.ndarray:
    """
    Reverse optimisation: π = δ × Σ × w
    Returns implied equilibrium excess returns π.
    """
    return delta * sigma @ weights


def _bl_posterior(
    pi: np.ndarray,
    sigma: np.ndarray,
    tau: float,
    P: np.ndarray,   # k×n view matrix
    Q: np.ndarray,   # k×1 view expected returns
    omega: np.ndarray,  # k×k view uncertainty diagonal
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Black-Litterman posterior expected returns and covariance.

    Returns (mu_bl, sigma_bl) — posterior mean and covariance.
    """
    n = len(pi)
    tau_sigma = tau * sigma
    # BL posterior mean: (τΣ)⁻¹ + Pᵀ Ω⁻¹ P)⁻¹ ((τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q)
    try:
        inv_tau_sigma = np.linalg.inv(tau_sigma + 1e-8 * np.eye(n))
        inv_omega = np.linalg.inv(omega + 1e-10 * np.eye(len(Q)))
        M = inv_tau_sigma + P.T @ inv_omega @ P
        inv_M = np.linalg.inv(M + 1e-8 * np.eye(n))
        mu_bl = inv_M @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)
        sigma_bl = sigma + inv_M
    except np.linalg.LinAlgError:
        mu_bl = pi.copy()
        sigma_bl = sigma.copy()
    return mu_bl, sigma_bl


def _mvo_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    delta: float,
    floor: float,
    cap: float,
) -> np.ndarray:
    """
    Simple MVO: w* = (δ Σ)⁻¹ μ, then normalise and apply floor/cap.
    Falls back to equal-weight on degenerate covariance.
    """
    n = len(mu)
    try:
        inv_sigma = np.linalg.inv(delta * sigma + 1e-8 * np.eye(n))
        w = inv_sigma @ mu
    except np.linalg.LinAlgError:
        w = np.ones(n)

    # Keep only non-negative weights (long-only)
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total < 1e-10:
        w = np.ones(n) / n
    else:
        w = w / total

    # Apply floor/cap iteratively (renorm can violate cap, so repeat until stable)
    floor_f = float(floor)
    cap_f = float(cap)
    for _ in range(20):
        w = np.clip(w, floor_f, cap_f)
        total = w.sum()
        if total < 1e-10:
            return np.ones(n) / n
        w = w / total
        if w.max() <= cap_f + 1e-8:
            break

    return w


# ── Allocator class ───────────────────────────────────────────────────────────

@dataclass
class BLResult:
    symbols: List[str]
    weights: Dict[str, float]           # symbol → portfolio weight
    multipliers: Dict[str, float]       # symbol → weight / equal_weight
    posterior_returns: Dict[str, float] # symbol → BL expected return
    n_views: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BlackLittermanAllocator:
    """
    Black-Litterman portfolio allocator.
    Uses FactorICTracker views when available; falls back to equal-weight.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._persist_path = Path(persist_path or str(_cfg("BL_PERSIST_PATH")))
        self._last_result: Optional[BLResult] = None
        self._last_cycle: int = -999
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_weights(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        ic_tracker=None,
        live_signals: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> BLResult:
        """
        Compute BL weights for ``symbols``.

        Args:
            symbols: List of tradeable symbols.
            price_data: ``{symbol: OHLCV DataFrame}``.
            ic_tracker: FactorICTracker instance (or ``None`` → equal-weight
                prior for macro views).
            live_signals: Optional per-symbol live ML signal views:
                ``{symbol: {"signal": float in [-1,1], "confidence": float in [0,1]}}``.
                Each ``(signal × BL_VIEW_SCALE)`` is injected as an absolute
                view on that symbol's return, with uncertainty
                ``BL_UNCERTAINTY_BASE / max(confidence², eps)`` so low-
                confidence signals pull the posterior less. When ``None`` or
                empty, the allocator falls back to IC-tracker views only
                (legacy behaviour).

        Returns:
            BLResult with weights and multipliers.
        """
        if not _cfg("BL_ENABLED") or len(symbols) < 2:
            return self._equal_weight(symbols)

        n = len(symbols)
        lookback = int(_cfg("BL_LOOKBACK_BARS"))
        delta = float(_cfg("BL_RISK_AVERSION"))
        tau = float(_cfg("BL_TAU"))
        floor = float(_cfg("BL_WEIGHT_FLOOR"))
        cap = float(_cfg("BL_WEIGHT_CAP"))

        # ── 1. Build returns matrix ───────────────────────────────────────────
        returns_matrix = self._build_returns(symbols, price_data, lookback)
        if returns_matrix is None or returns_matrix.shape[1] < 3:
            return self._equal_weight(symbols)

        actual_syms = list(returns_matrix.index) if isinstance(returns_matrix, pd.DataFrame) else symbols
        if len(actual_syms) < 2:
            return self._equal_weight(symbols)

        ret_arr = returns_matrix.values if isinstance(returns_matrix, pd.DataFrame) else returns_matrix
        sigma = np.cov(ret_arr) + 1e-8 * np.eye(len(actual_syms))
        eq_w = np.ones(len(actual_syms)) / len(actual_syms)

        # ── 2. Reverse optimise equilibrium returns ───────────────────────────
        pi = _reverse_optimize(sigma, eq_w, delta)

        # ── 3. Build IC views + live-signal views ─────────────────────────────
        P, Q, omega = self._build_views(actual_syms, ic_tracker, live_signals)
        n_views = len(Q) if Q is not None else 0

        # ── 4. BL posterior ───────────────────────────────────────────────────
        if P is not None and len(Q) > 0:
            mu_bl, _ = _bl_posterior(pi, sigma, tau, P, Q, omega)
        else:
            mu_bl = pi

        # ── 5. MVO ────────────────────────────────────────────────────────────
        raw_weights = _mvo_weights(mu_bl, sigma, delta, floor, cap)

        # ── 6. Build result ───────────────────────────────────────────────────
        ew = 1.0 / len(actual_syms)
        weights = {s: round(float(raw_weights[i]), 4) for i, s in enumerate(actual_syms)}
        multipliers = {s: round(float(raw_weights[i]) / ew, 4) for i, s in enumerate(actual_syms)}
        posterior_returns = {s: round(float(mu_bl[i]), 6) for i, s in enumerate(actual_syms)}

        # Fill in missing symbols with equal weight
        for sym in symbols:
            if sym not in weights:
                weights[sym] = round(ew, 4)
                multipliers[sym] = 1.0
                posterior_returns[sym] = 0.0

        result = BLResult(
            symbols=actual_syms,
            weights=weights,
            multipliers=multipliers,
            posterior_returns=posterior_returns,
            n_views=n_views,
        )
        self._last_result = result
        self._save(result)

        logger.info(
            "BlackLitterman: %d symbols, %d views — top: %s",
            len(actual_syms), n_views,
            sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3],
        )
        return result

    def maybe_update(
        self,
        cycle: int,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        ic_tracker=None,
        live_signals: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[BLResult]:
        """
        Call in main loop; only recomputes every ``BL_REFRESH_INTERVAL``
        cycles. ``live_signals`` is forwarded to :meth:`get_weights` so each
        refresh picks up the most recent model outputs.
        """
        interval = int(_cfg("BL_REFRESH_INTERVAL"))
        if cycle % interval != 0:
            return self._last_result
        return self.get_weights(symbols, price_data, ic_tracker, live_signals)

    def get_multiplier(self, symbol: str, default: float = 1.0) -> float:
        """Return the BL sizing multiplier for a symbol (1.0 = equal weight)."""
        if not _cfg("BL_ENABLED") or self._last_result is None:
            return default
        return float(self._last_result.multipliers.get(symbol, default))

    def get_report(self) -> dict:
        if self._last_result is None:
            return {"enabled": _cfg("BL_ENABLED"), "weights": {}, "multipliers": {}}
        return {
            "enabled": _cfg("BL_ENABLED"),
            "weights": self._last_result.weights,
            "multipliers": self._last_result.multipliers,
            "posterior_returns": self._last_result.posterior_returns,
            "n_views": self._last_result.n_views,
            "timestamp": self._last_result.timestamp,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_returns(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        lookback: int,
    ) -> Optional[pd.DataFrame]:
        """Build a (symbols × lookback) log-return matrix, dropping symbols with missing data."""
        series_dict = {}
        for sym in symbols:
            df = price_data.get(sym)
            if df is None or "Close" not in df.columns or len(df) < lookback + 1:
                continue
            closes = df["Close"].tail(lookback + 1).values.astype(float)
            log_rets = np.log(closes[1:] / np.maximum(closes[:-1], 1e-10))
            series_dict[sym] = log_rets

        if len(series_dict) < 2:
            return None

        # Align to common length
        min_len = min(len(v) for v in series_dict.values())
        df_ret = pd.DataFrame(
            {sym: v[-min_len:] for sym, v in series_dict.items()}
        ).T
        return df_ret

    def _build_views(
        self,
        symbols: List[str],
        ic_tracker,
        live_signals: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build the BL view matrix ``P``, view returns ``Q`` and uncertainty
        ``Ω`` from two sources:

        1. **Macro IC views** — one uniform-weight row per reliable factor
           in ``ic_tracker.get_report().signals``. View return is
           ``factor.ic × BL_VIEW_SCALE``; uncertainty scales with IC magnitude.
        2. **Live per-symbol views** — one identity-style row per symbol in
           ``live_signals`` for which ``|signal| > 0``. View return is
           ``signal × BL_VIEW_SCALE``; uncertainty is
           ``BL_UNCERTAINTY_BASE / max(confidence², eps)``.

        Either branch may be empty; only when *both* are empty does the
        method return ``(None, None, None)``.

        Args:
            symbols: Ordered list of symbols defining the column space of
                ``P`` (row width ``n = len(symbols)``).
            ic_tracker: FactorICTracker instance (may be ``None``).
            live_signals: Per-symbol live-signal dict (see
                :meth:`get_weights` docstring).

        Returns:
            Tuple ``(P, Q, omega)`` or ``(None, None, None)`` when no views
            could be constructed.
        """
        n = len(symbols)
        sym_index = {s: i for i, s in enumerate(symbols)}
        views_P: List[np.ndarray] = []
        views_Q: List[float] = []
        views_var: List[float] = []

        view_scale = float(_cfg("BL_VIEW_SCALE"))
        view_conf = float(_cfg("BL_VIEW_CONFIDENCE"))

        # ── 1. Macro IC views ────────────────────────────────────────────
        if ic_tracker is not None:
            try:
                report = ic_tracker.get_report()
            except Exception:
                report = None

            if report is not None:
                min_ic = float(_cfg("BL_MIN_IC_FOR_VIEW"))
                min_obs = int(_cfg("BL_MIN_OBS_FOR_VIEW"))
                active = [
                    r for r in getattr(report, "signals", [])
                    if getattr(r, "is_reliable", False)
                    and getattr(r, "obs", 0) >= min_obs
                    and abs(getattr(r, "ic", 0.0)) >= min_ic
                    and getattr(r, "status", "") == "active"
                ]
                for factor in active:
                    # A positive IC factor → positive view across all symbols.
                    view_return = float(factor.ic) * view_scale
                    ic_conf = min(abs(factor.ic) / 0.30, 1.0) * view_conf
                    view_var = (1.0 - ic_conf) ** 2 * 0.01
                    views_P.append(np.ones(n) / n)
                    views_Q.append(view_return)
                    views_var.append(view_var)

        # ── 2. Live per-symbol views ─────────────────────────────────────
        if live_signals:
            if ApexConfig is not None and hasattr(ApexConfig, "BL_UNCERTAINTY_BASE"):
                uncertainty_base = float(ApexConfig.BL_UNCERTAINTY_BASE)
            else:
                # Fallback for stripped-down test shims without the attribute.
                uncertainty_base = 0.0025
            eps = 1e-4
            for sym, payload in live_signals.items():
                if sym not in sym_index or not isinstance(payload, dict):
                    continue
                try:
                    signal = float(payload.get("signal", 0.0))
                    conf = float(payload.get("confidence", 0.0))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(signal) or not np.isfinite(conf):
                    continue
                if abs(signal) <= 0.0:
                    continue
                signal = max(-1.0, min(1.0, signal))
                conf = max(0.0, min(1.0, conf))

                row = np.zeros(n, dtype=float)
                row[sym_index[sym]] = 1.0
                views_P.append(row)
                views_Q.append(signal * view_scale)
                views_var.append(uncertainty_base / max(conf * conf, eps))

        if not views_P:
            return None, None, None

        P = np.array(views_P)
        Q = np.array(views_Q)
        omega = np.diag(views_var)
        return P, Q, omega

    def _equal_weight(self, symbols: List[str]) -> BLResult:
        n = max(len(symbols), 1)
        ew = round(1.0 / n, 4)
        return BLResult(
            symbols=symbols,
            weights={s: ew for s in symbols},
            multipliers={s: 1.0 for s in symbols},
            posterior_returns={s: 0.0 for s in symbols},
            n_views=0,
        )

    def _save(self, result: BLResult) -> None:
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "weights": result.weights,
                "multipliers": result.multipliers,
                "posterior_returns": result.posterior_returns,
                "n_views": result.n_views,
                "timestamp": result.timestamp,
                "symbols": result.symbols,
            }
            tmp = self._persist_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._persist_path)
        except Exception as e:
            logger.debug("BlackLitterman: save failed — %s", e)

    def _load(self) -> None:
        try:
            if not self._persist_path.exists():
                return
            with open(self._persist_path) as f:
                state = json.load(f)
            self._last_result = BLResult(
                symbols=state.get("symbols", []),
                weights=state.get("weights", {}),
                multipliers=state.get("multipliers", {}),
                posterior_returns=state.get("posterior_returns", {}),
                n_views=state.get("n_views", 0),
                timestamp=state.get("timestamp", ""),
            )
        except Exception as e:
            logger.debug("BlackLitterman: load failed — %s", e)
