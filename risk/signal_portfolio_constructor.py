"""
risk/signal_portfolio_constructor.py — Signal-Aware Portfolio Construction

Bridges ML signal strengths (expected returns) with Hierarchical Risk Parity
to produce target allocation weights per symbol.

What this fixes:
  - Current system: sizes each trade independently (Kelly × VIX × HRP dampener)
  - This module: portfolio-level target weights → per-trade sizing cap

Algorithm (simplified Black-Litterman meets HRP):
  1. Collect ML signal values as Ω-scaled views on expected returns
  2. Compute rolling covariance matrix from price returns
  3. Blend equilibrium (equal-weight) + signal views using confidence weight
  4. Run HRP on the blended expected returns + cov matrix
  5. Per-symbol target weight guides position sizing cap

Usage:
    constructor = SignalPortfolioConstructor(state_dir=Path("data/portfolio_constructor"))

    # Update views (called each cycle with fresh ML signals)
    constructor.update_signals({"AAPL": 0.18, "MSFT": 0.12, "BTC/USD": 0.22})

    # Update covariance from rolling returns
    constructor.update_returns({"AAPL": [0.01, -0.005, ...], "MSFT": [...]})

    # Get target weight for a symbol [0, max_weight]
    w = constructor.get_target_weight("AAPL")

    # Check if proposed sizing is within the weight budget
    scale = constructor.get_sizing_scale("AAPL", proposed_notional=10000, portfolio_value=500000)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_MAX_WEIGHT = 0.15           # No single symbol > 15% of portfolio
_DEFAULT_MIN_HISTORY = 10            # Min return observations before HRP is active
_DEFAULT_SIGNAL_BLEND = 0.40         # Weight of signal views vs equal-weight prior
_DEFAULT_RECOMPUTE_INTERVAL = 60     # Seconds between full recomputes


@dataclass
class PortfolioWeights:
    """Output of the constructor: target allocations per symbol."""
    weights: Dict[str, float]           # symbol → target fraction [0, max_weight]
    computed_at: float                  # epoch seconds
    method: str                         # "hrp_signal" | "equal" | "fallback"
    n_symbols: int
    cov_condition: float = 0.0          # condition number of cov matrix (proxy for quality)
    top_signals: Dict[str, float] = field(default_factory=dict)

    def get(self, symbol: str, default: float = 0.0) -> float:
        return self.weights.get(symbol, default)


class SignalPortfolioConstructor:
    """
    Converts per-symbol ML signals into covariance-aware target weights.

    The weights are SOFT constraints on sizing — they don't block trades,
    they scale down positions that would over-concentrate the portfolio.
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        max_single_weight: float = _DEFAULT_MAX_WEIGHT,
        min_history: int = _DEFAULT_MIN_HISTORY,
        signal_blend: float = _DEFAULT_SIGNAL_BLEND,
        recompute_interval_s: float = _DEFAULT_RECOMPUTE_INTERVAL,
    ):
        self._max_w = max_single_weight
        self._min_history = min_history
        self._signal_blend = signal_blend
        self._recompute_interval = recompute_interval_s

        self._signals: Dict[str, float] = {}          # symbol → latest ML signal
        self._returns: Dict[str, List[float]] = {}    # symbol → rolling returns
        self._current: Optional[PortfolioWeights] = None
        self._last_compute_ts: float = 0.0

        if state_dir:
            self._state_dir = Path(state_dir)
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load()
        else:
            self._state_dir = None

    # ── Public write API ──────────────────────────────────────────────────────

    def update_signals(self, signals: Dict[str, float]) -> None:
        """Update ML signal strength estimates (expected return proxies)."""
        for sym, val in signals.items():
            if val is not None and np.isfinite(float(val)):
                self._signals[sym] = float(val)

    def update_returns(self, returns: Dict[str, List[float]]) -> None:
        """Update rolling return history used to build the covariance matrix."""
        for sym, rets in returns.items():
            self._returns[sym] = [float(r) for r in rets if np.isfinite(r)]

    def maybe_recompute(self) -> bool:
        """Recompute target weights if interval has elapsed. Returns True if recomputed."""
        if time.time() - self._last_compute_ts < self._recompute_interval:
            return False
        self._current = self._compute()
        self._last_compute_ts = time.time()
        self._save()
        return True

    # ── Public query API ──────────────────────────────────────────────────────

    def get_target_weight(self, symbol: str) -> float:
        """Return current target portfolio weight for symbol in [0, max_weight]."""
        if self._current is None:
            return self._max_w  # unconstrained until first compute
        return self._current.get(symbol, 0.0)

    def get_sizing_scale(
        self,
        symbol: str,
        proposed_notional: float,
        portfolio_value: float,
    ) -> float:
        """
        Return a scaling factor [0, 1] to apply to proposed_notional so that
        the resulting position doesn't breach the target weight budget.

        Returns 1.0 (no change) when:
          - portfolio_value == 0
          - no weights computed yet
          - proposed weight is within budget

        Returns < 1.0 when proposed notional would overshoot target weight.
        """
        if portfolio_value <= 0 or proposed_notional <= 0:
            return 1.0
        target_w = self.get_target_weight(symbol)
        if target_w <= 0:
            return 1.0
        budget = target_w * portfolio_value
        if proposed_notional <= budget:
            return 1.0
        return float(budget / proposed_notional)

    def get_portfolio_snapshot(self) -> Dict:
        """Return current weights snapshot for API/dashboard."""
        if self._current is None:
            return {
                "available": False,
                "note": "Weights not yet computed",
                "weights": {},
            }
        return {
            "available": True,
            "method": self._current.method,
            "n_symbols": self._current.n_symbols,
            "computed_at": self._current.computed_at,
            "cov_condition": round(self._current.cov_condition, 2),
            "top_signals": {
                k: round(v, 4)
                for k, v in sorted(
                    self._current.top_signals.items(),
                    key=lambda x: -abs(x[1]),
                )[:20]
            },
            "weights": {
                k: round(v, 4)
                for k, v in sorted(
                    self._current.weights.items(),
                    key=lambda x: -x[1],
                )
            },
        }

    # ── Core computation ──────────────────────────────────────────────────────

    def _compute(self) -> PortfolioWeights:
        """Compute signal-aware HRP target weights."""
        symbols = [s for s in self._signals if s in self._returns]
        symbols = [s for s in symbols if len(self._returns[s]) >= self._min_history]

        if len(symbols) < 2:
            return self._fallback_weights()

        # Build return matrix: shape (n_symbols, n_obs)
        min_len = min(len(self._returns[s]) for s in symbols)
        matrix = np.array([self._returns[s][-min_len:] for s in symbols])

        try:
            cov = np.cov(matrix)
        except Exception:
            return self._fallback_weights()

        if cov.ndim < 2 or np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            return self._fallback_weights()

        # Condition number as quality metric
        try:
            cond = float(np.linalg.cond(cov))
        except Exception:
            cond = 0.0

        # Signal views: normalise signals to [0, 1] range → expected return proxy
        raw_sigs = np.array([self._signals[s] for s in symbols])
        sig_min, sig_max = raw_sigs.min(), raw_sigs.max()
        if sig_max > sig_min:
            norm_sigs = (raw_sigs - sig_min) / (sig_max - sig_min)
        else:
            norm_sigs = np.ones(len(symbols)) / len(symbols)

        # Equal-weight prior blended with signal views
        eq_w = np.ones(len(symbols)) / len(symbols)
        blended = (1 - self._signal_blend) * eq_w + self._signal_blend * norm_sigs
        blended = np.maximum(blended, 1e-6)
        blended /= blended.sum()

        # Run HRP using the blended views as the return proxy for cluster weighting
        hrp_w = self._hrp_weights(cov, blended)

        # Apply max_weight cap and renormalize
        hrp_w = np.minimum(hrp_w, self._max_w)
        total = hrp_w.sum()
        if total > 0:
            hrp_w /= total
        hrp_w = np.minimum(hrp_w, self._max_w)

        weights_dict = {s: float(w) for s, w in zip(symbols, hrp_w)}
        top_sigs = {s: self._signals[s] for s in symbols}

        return PortfolioWeights(
            weights=weights_dict,
            computed_at=time.time(),
            method="hrp_signal",
            n_symbols=len(symbols),
            cov_condition=cond,
            top_signals=top_sigs,
        )

    def _fallback_weights(self) -> PortfolioWeights:
        """Equal-weight fallback when insufficient data."""
        symbols = list(self._signals.keys())
        n = max(len(symbols), 1)
        w = min(1.0 / n, self._max_w)
        return PortfolioWeights(
            weights={s: w for s in symbols},
            computed_at=time.time(),
            method="equal",
            n_symbols=n,
        )

    def _hrp_weights(self, cov: np.ndarray, signal_views: np.ndarray) -> np.ndarray:
        """
        Signal-weighted HRP.

        Standard HRP gives equal risk to each cluster; here we tilt cluster
        allocations by the average signal view in that cluster.
        """
        n = cov.shape[0]
        if n == 1:
            return np.array([1.0])

        # Correlation matrix → distance matrix
        std = np.sqrt(np.diag(cov))
        std = np.where(std < 1e-12, 1e-12, std)
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        dist = np.sqrt(np.maximum((1 - corr) / 2.0, 0))

        # Quasi-diagonalisation via single-linkage clustering
        order = self._quasi_diag(dist)

        # Recursive bisection with signal tilt
        weights = np.ones(n)
        clusters = [order]
        while clusters:
            cluster = clusters.pop()
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            # Cluster variances (inverse-variance weighting)
            var_l = self._cluster_var(cov, left)
            var_r = self._cluster_var(cov, right)
            total_var = var_l + var_r
            if total_var < 1e-14:
                alpha = 0.5
            else:
                alpha = 1.0 - var_l / total_var  # right gets more weight if left is riskier

            # Signal tilt: boost the cluster with stronger average signal
            avg_sig_l = float(signal_views[left].mean())
            avg_sig_r = float(signal_views[right].mean())
            sig_sum = avg_sig_l + avg_sig_r
            if sig_sum > 1e-10:
                sig_tilt_l = avg_sig_l / sig_sum
                sig_tilt_r = avg_sig_r / sig_sum
                # Blend 70% HRP + 30% signal tilt
                alloc_r = 0.70 * alpha + 0.30 * sig_tilt_r
                alloc_r = float(np.clip(alloc_r, 0.05, 0.95))
            else:
                alloc_r = float(alpha)

            alloc_l = 1.0 - alloc_r
            weights[left] *= alloc_l
            weights[right] *= alloc_r
            clusters.extend([left, right])

        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    @staticmethod
    def _quasi_diag(dist: np.ndarray) -> List[int]:
        """Return quasi-diagonalised ordering via single-linkage."""
        from scipy.cluster.hierarchy import linkage, leaves_list  # type: ignore
        try:
            n = dist.shape[0]
            if n == 1:
                return [0]
            condensed = dist[np.triu_indices(n, k=1)]
            link = linkage(condensed, method="single")
            return [int(i) for i in leaves_list(link)]
        except Exception:
            return list(range(dist.shape[0]))

    @staticmethod
    def _cluster_var(cov: np.ndarray, indices: List[int]) -> float:
        """Cluster variance under inverse-variance weighting."""
        sub = cov[np.ix_(indices, indices)]
        diag_v = np.diag(sub)
        inv_v = 1.0 / np.maximum(diag_v, 1e-14)
        w = inv_v / inv_v.sum()
        return float(w @ sub @ w)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._state_dir is None or self._current is None:
            return
        try:
            state = {
                "weights": self._current.weights,
                "computed_at": self._current.computed_at,
                "method": self._current.method,
                "n_symbols": self._current.n_symbols,
                "cov_condition": self._current.cov_condition,
            }
            p = self._state_dir / "portfolio_weights.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception as exc:
            logger.debug("SignalPortfolioConstructor: save failed: %s", exc)

    def _load(self) -> None:
        if self._state_dir is None:
            return
        try:
            p = self._state_dir / "portfolio_weights.json"
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            self._current = PortfolioWeights(
                weights=raw.get("weights", {}),
                computed_at=float(raw.get("computed_at", 0)),
                method=raw.get("method", "loaded"),
                n_symbols=int(raw.get("n_symbols", 0)),
                cov_condition=float(raw.get("cov_condition", 0)),
            )
        except Exception as exc:
            logger.debug("SignalPortfolioConstructor: load failed: %s", exc)
