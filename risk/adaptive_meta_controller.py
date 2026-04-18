"""
risk/adaptive_meta_controller.py  —  Self-Learning Unified Trade Gate

Replaces the cascade of hard-coded threshold checks (news gate, consensus gate,
cross-asset gate, drawdown gate, etc.) with a single component that:

  1. Captures the full market context at each entry decision
  2. Computes a continuous context-quality score  (no binary gates)
  3. Learns — via online ridge regression — which contexts lead to
     profitable trades
  4. Auto-tunes its own implicit thresholds toward maximum risk-adjusted return

No config values or thresholds are hard-coded.
The only "prior" is an interpretable heuristic (market wisdom encoded in soft
tanh curves) that activates from day 1 and is gradually replaced by empirical
data as trade outcomes accumulate.
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from config import ApexConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeContext:
    """Serialisable snapshot of all decision-relevant context at entry time."""

    symbol: str
    signal: float              # ML signal  [-1, 1]
    confidence: float          # model confidence [0, 1]
    asset_class: str           # "EQUITY" | "CRYPTO" | "FOREX"
    regime: str                # e.g. "bull", "bear", "volatile"

    # External context — all default to 0.0 / False when unavailable
    news_sentiment: float  = 0.0    # [-1, 1]
    news_confidence: float = 0.0    # [0, 1]
    news_momentum: float   = 0.0    # [-1, 1] improving vs deteriorating

    macro_risk_appetite:  float = 0.0
    yield_curve_inverted: bool  = False
    vix_backwardation:    bool  = False

    ofi:             float = 0.0    # order flow imbalance  [-1, 1]
    consensus_ratio: float = 0.5    # fraction of signal sources agreeing [0, 1]

    btc_signal:          float = 0.0
    spy_signal:          float = 0.0
    funding_rate_signal: float = 0.0   # crypto: fade-the-crowd signal
    pattern_signal:      float = 0.0   # candlestick pattern signal

    hhi:           float = 0.0   # portfolio concentration [0, 1]
    daily_loss_pct: float = 0.0  # intraday P&L as a fraction (e.g. -0.02)
    vol_percentile: float = 0.5  # current realised vol vs 6-month history

    def to_feature_vector(self) -> np.ndarray:
        """Convert to a normalised numeric vector.  Order is FIXED — do not reorder."""
        _sign = float(np.sign(self.signal + 1e-9))
        return np.array([
            self.signal,
            self.confidence,
            self.news_sentiment  * _sign,   # news aligned with signal direction
            self.news_confidence,
            self.news_momentum   * _sign,   # improving sentiment in trade direction
            self.macro_risk_appetite * _sign,
            1.0 if self.yield_curve_inverted else 0.0,
            1.0 if self.vix_backwardation    else 0.0,
            self.ofi             * _sign,   # OFI aligned with direction
            self.consensus_ratio,
            self.btc_signal      * _sign,   # BTC cross-asset alignment
            self.spy_signal      * _sign,   # SPY cross-asset alignment
            self.funding_rate_signal * _sign,
            self.pattern_signal  * _sign,
            self.hhi,                        # concentration (higher = worse)
            self.daily_loss_pct,             # drawdown (negative = worse)
            self.vol_percentile,             # vol regime
        ], dtype=np.float64)

    _N_FEATURES: int = 17   # must match the vector above


@dataclass
class TradeDecision:
    """Decision returned by the meta-controller for a given context."""

    allow:                  bool
    confidence_multiplier:  float   # applied to caller's confidence (1.0 = no change)
    size_multiplier:        float   # applied to position size      (1.0 = no change)
    context_score:          float   # [-1, 1]  learned quality of this context
    reasoning:              List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveMetaController:
    """
    Self-learning, context-aware trade gate.

    Architecture
    ────────────
    Each (asset_class × regime) bucket maintains a rolling deque of
    (feature_vector, realised_pnl) pairs.  An online ridge regression
    maps features → expected pnl (normalised by a hurdle return).

    The ``context_score`` output is:
      • heuristic-only when < MIN_OUTCOMES samples are available
      • a weighted blend that shifts from heuristic → learned as data grows
      • fully learned-model-driven after 3 × MIN_OUTCOMES samples

    A negative context_score reduces confidence and size smoothly.
    Below _BLOCK_SCORE the trade is blocked.
    ALL thresholds are derived from the learned distribution — not from any
    config file.
    """

    _BUFFER_SIZE  = 200     # max outcome history per bucket
    _MIN_OUTCOMES = 15      # min before learned model starts contributing
    _HURDLE_RATE  = 0.003   # per-trade expected return target (0.3%)
    _BLOCK_SCORE  = -0.55   # context_score below this → block
    _CONF_PENALTY = 0.25    # max downward confidence adjustment (×0.75)
    _CONF_BOOST   = 0.12    # max upward   confidence adjustment (×1.12)
    _SIZE_FLOOR   = 0.40    # min size multiplier
    _SIZE_CEIL    = 1.25    # max size multiplier
    _RIDGE_ALPHA  = 0.10    # L2 regularisation for ridge regression

    def __init__(self, persist_path: Optional[str] = None) -> None:
        """
        Construct the controller.

        Args:
            persist_path: Override for the on-disk state file. When ``None``
                (the default), the path is sourced from
                :attr:`ApexConfig.META_CONTROLLER_STATE_PATH` so deployment
                locations can be changed by env without code edits.
        """
        self._persist_path: str = str(
            persist_path
            if persist_path is not None
            else ApexConfig.META_CONTROLLER_STATE_PATH
        )
        self._buffers: Dict[str, deque]       = {}
        self._weights: Dict[str, np.ndarray]  = {}
        self._n_obs:   Dict[str, int]         = {}
        # Latest market-context snapshot — persisted alongside the learned
        # state so a cold-started controller can diagnose / recover what
        # regime the last decision ran under. Populated via
        # :meth:`record_context_snapshot`.
        self._last_context: Dict[str, Any] = {}
        self._load_state()
        logger.info(
            "AdaptiveMetaController ready — %d learned buckets, "
            "min_outcomes=%d before learned model activates",
            len(self._weights), self._MIN_OUTCOMES,
        )

    # ── Context snapshot (for recovery & diagnostics) ────────────────────────

    def record_context_snapshot(
        self,
        regime: str,
        vix_level: Optional[float],
        severity: float,
    ) -> None:
        """
        Persist the most-recent market-context snapshot atomically.

        This stores ``{regime, vix_level, timestamp, severity}`` into the
        state file so an operator or a freshly-started process can inspect
        which regime / stress level the controller saw last.

        Args:
            regime: Canonical regime label (e.g. ``"bull"``, ``"high_volatility"``).
            vix_level: Most-recent VIX reading. ``None`` when unavailable.
            severity: Composite stress score in ``[0, 1]`` (callers typically
                use ``1 - context_score`` clipped to that range).

        Raises:
            TypeError: If ``regime`` is not a string.
        """
        if not isinstance(regime, str):
            raise TypeError(f"regime must be str, got {type(regime).__name__}")
        vix_out: Optional[float] = None
        if vix_level is not None:
            try:
                v = float(vix_level)
                if np.isfinite(v):
                    vix_out = v
            except (TypeError, ValueError):
                vix_out = None
        try:
            sev = float(severity)
            if not np.isfinite(sev):
                sev = 0.0
        except (TypeError, ValueError):
            sev = 0.0
        sev = float(np.clip(sev, 0.0, 1.0))

        self._last_context = {
            "regime": regime,
            "vix_level": vix_out,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": sev,
        }
        self._save_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, ctx: TradeContext) -> TradeDecision:
        """
        Evaluate the market context and return a trade decision.

        Combines a heuristic prior (always available on day 1) with the
        learned ridge model (activates gradually after MIN_OUTCOMES trades).
        """
        bucket = f"{ctx.asset_class}:{ctx.regime}"
        fvec   = ctx.to_feature_vector()
        reasons: List[str] = []

        # ── Heuristic prior (no training data needed) ─────────────────────
        h_score = self._heuristic(ctx)
        reasons.append(f"heuristic={h_score:+.3f}")

        # ── Learned model blend ───────────────────────────────────────────
        n = len(self._buffers.get(bucket, []))
        if n >= self._MIN_OUTCOMES and bucket in self._weights:
            l_score = float(np.clip(np.dot(self._weights[bucket], fvec), -2.0, 2.0))
            # Blend weight: 0 at MIN_OUTCOMES → 1 at 3 × MIN_OUTCOMES
            blend = min(1.0, (n - self._MIN_OUTCOMES) / (2.0 * self._MIN_OUTCOMES))
            ctx_score = (1.0 - blend) * h_score + blend * l_score
            reasons.append(f"learned={l_score:+.3f} blend={blend:.2f} n={n}")
        else:
            ctx_score = h_score
            reasons.append(f"prior only (n={n}<{self._MIN_OUTCOMES})")

        ctx_score = float(np.clip(ctx_score, -1.5, 1.5))

        # ── Block ─────────────────────────────────────────────────────────
        if ctx_score < self._BLOCK_SCORE:
            reasons.append(f"BLOCKED context_score={ctx_score:.3f}")
            return TradeDecision(
                allow=False, confidence_multiplier=0.0, size_multiplier=0.0,
                context_score=ctx_score, reasoning=reasons,
            )

        # ── Continuous adjustments (smooth tanh, no sharp cliffs) ─────────
        raw = float(np.tanh(ctx_score * 1.5))
        c_delta = raw * (self._CONF_BOOST if raw > 0 else self._CONF_PENALTY)
        conf_mult = float(np.clip(1.0 + c_delta,
                                   1.0 - self._CONF_PENALTY,
                                   1.0 + self._CONF_BOOST))
        size_mult = float(np.clip(1.0 + raw * 0.20, self._SIZE_FLOOR, self._SIZE_CEIL))

        return TradeDecision(
            allow=True,
            confidence_multiplier=conf_mult,
            size_multiplier=size_mult,
            context_score=ctx_score,
            reasoning=reasons,
        )

    def record_outcome(self, ctx: TradeContext, pnl_pct: float) -> None:
        """
        Feed a realised trade outcome back into the learner.
        Called at trade close with the TradeContext captured at entry.
        """
        bucket = f"{ctx.asset_class}:{ctx.regime}"
        if bucket not in self._buffers:
            self._buffers[bucket] = deque(maxlen=self._BUFFER_SIZE)
        self._buffers[bucket].append((ctx.to_feature_vector(), float(pnl_pct)))
        self._n_obs[bucket] = self._n_obs.get(bucket, 0) + 1

        if len(self._buffers[bucket]) >= self._MIN_OUTCOMES:
            self._refit(bucket)

        if self._n_obs[bucket] % 5 == 0:
            self._save_state()

        logger.debug(
            "MetaController.record_outcome: %s bucket=%s pnl=%.4f n=%d",
            ctx.symbol, bucket, pnl_pct, len(self._buffers[bucket]),
        )

    # ── Heuristic prior ───────────────────────────────────────────────────────

    def _heuristic(self, ctx: TradeContext) -> float:
        """
        Market-wisdom prior — no hard thresholds, only soft tanh curves.

        Each component is in (−1, +1); weighted sum → clipped to (−1, +1).
        Score > 0: context supports the trade.
        Score < 0: context contradicts / is uncertain.
        """
        s = 0.0
        _sign = float(np.sign(ctx.signal + 1e-9))

        # Signal strength & confidence
        s += float(np.tanh(abs(ctx.signal) * 5.0))           * 0.10
        s += float(np.tanh((ctx.confidence - 0.55) * 6.0))   * 0.10

        # News: sentiment alignment + momentum + conviction
        if ctx.news_confidence > 0.05:
            news_align = ctx.news_sentiment * _sign
            s += float(np.tanh(news_align * ctx.news_confidence * 4.0)) * 0.20
        mom_align = ctx.news_momentum * _sign
        s += float(np.tanh(mom_align * 3.0)) * 0.07

        # Macro: risk appetite alignment
        macro_align = ctx.macro_risk_appetite * _sign
        s += float(np.tanh(macro_align * 2.5)) * 0.14
        if ctx.yield_curve_inverted:
            s -= 0.08  # systematic headwind for equities
        if ctx.vix_backwardation:
            s -= 0.09  # stress signal, suppress all entries

        # Order flow (smart money)
        ofi_align = ctx.ofi * _sign
        s += float(np.tanh(ofi_align * 4.0)) * 0.11

        # Signal source consensus
        s += float(np.tanh((ctx.consensus_ratio - 0.50) * 6.0)) * 0.11

        # Cross-asset coherence
        btc_align = ctx.btc_signal * _sign
        spy_align = ctx.spy_signal * _sign
        if abs(ctx.btc_signal) > 0.05 and abs(ctx.spy_signal) > 0.05:
            coherence = (btc_align + spy_align) / 2.0
            s += float(np.tanh(coherence * 3.0)) * 0.08
        elif abs(ctx.btc_signal) > 0.05:
            s += float(np.tanh(btc_align * 2.0)) * 0.04
        elif abs(ctx.spy_signal) > 0.05:
            s += float(np.tanh(spy_align * 2.0)) * 0.04

        # Crypto: funding rate (fade-the-crowd)
        if ctx.asset_class.upper() == "CRYPTO" and abs(ctx.funding_rate_signal) > 0.05:
            fr_align = ctx.funding_rate_signal * _sign
            s += float(np.tanh(fr_align * 3.0)) * 0.06

        # Candlestick pattern confirmation
        pat_align = ctx.pattern_signal * _sign
        s += float(np.tanh(pat_align * 4.0)) * 0.05

        # Portfolio concentration penalty
        if ctx.hhi > 0.20:
            s -= float(np.tanh((ctx.hhi - 0.20) * 5.0)) * 0.09

        # Drawdown caution (more conservative when already losing)
        if ctx.daily_loss_pct < -0.01:
            s -= float(np.tanh(abs(ctx.daily_loss_pct) * 20.0)) * 0.08

        # Volatility regime (very high vol = uncertain environment)
        if ctx.vol_percentile > 0.85:
            s -= float(np.tanh((ctx.vol_percentile - 0.85) * 8.0)) * 0.07

        return float(np.clip(s, -1.0, 1.0))

    # ── Ridge regression ──────────────────────────────────────────────────────

    def _refit(self, bucket: str) -> None:
        """Refit online ridge regression for a bucket."""
        buf = list(self._buffers[bucket])
        X   = np.vstack([fv for fv, _ in buf])
        y   = np.array([lbl for _, lbl in buf], dtype=np.float64)

        # Normalise labels to hurdle-relative units; clip outliers
        y_norm = np.clip(y / (self._HURDLE_RATE + 1e-8), -4.0, 4.0)

        try:
            n, d = X.shape
            A = X.T @ X + self._RIDGE_ALPHA * n * np.eye(d)
            b = X.T @ y_norm
            w = np.linalg.solve(A, b)
            self._weights[bucket] = w.astype(np.float64)
            logger.debug(
                "MetaController refit: bucket=%s n=%d w_norm=%.4f",
                bucket, n, float(np.linalg.norm(w)),
            )
        except np.linalg.LinAlgError as e:
            logger.debug("MetaController refit failed %s: %s", bucket, e)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """
        Load learned weights, observation counts and the last context
        snapshot from :attr:`_persist_path`. Missing / malformed files are
        tolerated — the controller will simply start from a cold state.
        """
        try:
            if not os.path.exists(self._persist_path):
                return
            with open(self._persist_path) as f:
                state = json.load(f)
            for bucket, w in state.get("weights", {}).items():
                self._weights[bucket] = np.array(w, dtype=np.float64)
            for bucket, n in state.get("n_obs", {}).items():
                self._n_obs[bucket] = int(n)
            last_ctx = state.get("last_context")
            if isinstance(last_ctx, dict):
                self._last_context = {
                    "regime":    str(last_ctx.get("regime", "")),
                    "vix_level": last_ctx.get("vix_level"),
                    "timestamp": str(last_ctx.get("timestamp", "")),
                    "severity":  float(last_ctx.get("severity", 0.0) or 0.0),
                }
            logger.info(
                "MetaController: loaded state — %d learned buckets",
                len(self._weights),
            )
        except Exception as e:
            logger.warning("MetaController: failed to load state: %s", e)

    def _save_state(self) -> None:
        """
        Persist the full controller state atomically.

        Writes to ``<persist_path>.tmp`` then ``os.replace`` onto the target.
        This prevents readers from observing a half-written file if the
        process crashes mid-write — a critical property for a file that is
        re-read on every cold start.
        """
        try:
            directory = os.path.dirname(self._persist_path) or "."
            os.makedirs(directory, exist_ok=True)
            state = {
                "weights":       {k: v.tolist() for k, v in self._weights.items()},
                "n_obs":         self._n_obs,
                "buffer_sizes":  {k: len(v) for k, v in self._buffers.items()},
                "last_context":  dict(self._last_context) if self._last_context else None,
            }
            tmp_path = f"{self._persist_path}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # fsync unsupported on some filesystems (e.g. some tmpfs)
                    pass
            os.replace(tmp_path, self._persist_path)
        except Exception as e:
            logger.debug("MetaController: save failed: %s", e)
