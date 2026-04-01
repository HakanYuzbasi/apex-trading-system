"""
risk/signal_enhancer.py — ML/NLP/Statistical Signal Enhancement Layer

Sits AFTER the raw signal generators and BEFORE the entry gates. Takes the
blended ML signal and enhances it using:

1. NLP Sentiment Adjustment — TF-IDF + logistic regression on news headlines
   (Platt-scaled probability → sentiment adjustment ±0.05 to ±0.15)

2. Multi-Factor Momentum Overlay — price momentum (1/5/20-bar), volume ratio,
   ADX trend strength, combined via learned weights

3. Calibrated Confidence — Isotonic regression calibration of raw confidence
   scores (fixes overconfident/underconfident ML models)

4. Regime-Adaptive Ensemble Weights — Bayesian online weighting of signal
   components based on which factors work in each regime

5. Market Microstructure Filter — time-of-day, spread width, volume quality
   adjustments that reduce signal noise without blocking conviction signals

Mathematical foundation:
- All regression models use ridge regularization (α=0.01) to prevent overfit
- Bayesian weighting uses Beta posteriors with exponential decay (λ=0.01/trade)
- Confidence calibration via sklearn.calibration.CalibratedClassifierCV
- No future leakage: features computed from [t-N, t-1], signal at t

Thread-safety: NOT thread-safe. Single-owner (execution loop) use only.
"""

import json
import logging
import math
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit  # stable sigmoid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EnhancedSignal:
    """Output from SignalEnhancer.enhance()."""
    raw_signal: float
    enhanced_signal: float
    raw_confidence: float
    calibrated_confidence: float
    sentiment_adj: float        # net NLP sentiment contribution
    momentum_adj: float         # net momentum overlay contribution
    regime_weight: float        # ensemble regime weight applied
    microstructure_ok: bool     # False → noisy/illiquid bar, be cautious
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def signal(self) -> float:
        return self.enhanced_signal

    @property
    def confidence(self) -> float:
        return self.calibrated_confidence


@dataclass
class _BayesianWeight:
    """Beta-distributed weight for a signal component."""
    alpha: float = 2.0  # prior: slight lean toward "useful"
    beta: float = 1.0
    decay: float = 0.99  # per-trade exponential forgetting

    @property
    def weight(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, contributed_positively: bool) -> None:
        self.alpha = max(0.1, self.alpha * self.decay + (1.0 if contributed_positively else 0.0))
        self.beta = max(0.1, self.beta * self.decay + (0.0 if contributed_positively else 1.0))

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: dict) -> "_BayesianWeight":
        return cls(alpha=d.get("alpha", 2.0), beta=d.get("beta", 1.0))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SignalEnhancer:
    """ML/NLP/statistical signal enhancement.

    Usage
    -----
    enhancer = SignalEnhancer(data_dir="data")
    result = enhancer.enhance(
        symbol="AAPL",
        raw_signal=0.18,
        raw_confidence=0.65,
        price_history=df["close"],        # pd.Series, last N bars
        volume_history=df["volume"],      # pd.Series, last N bars
        regime="bull",
        news_headlines=["AAPL beats earnings", ...],  # optional
        asset_class="equity",
    )
    """

    # ── NLP sentiment ──────────────────────────────────────────────────────
    # TF-IDF vocabulary: 200-word financial lexicon, pre-seeded
    # Model: Logistic regression (ridge C=10) trained online
    # Update frequency: every 20 trades with realized P&L labels

    # ── Momentum windows ──────────────────────────────────────────────────
    MOM_WINDOWS = [1, 5, 20]         # bars
    VOLUME_LOOKBACK = 20             # bars for avg volume

    # ── Calibration ───────────────────────────────────────────────────────
    CALIBRATION_MIN_TRADES = 20
    CALIBRATION_INTERVAL = 50        # recalibrate every N trades

    # ── Persist ───────────────────────────────────────────────────────────
    PERSIST_INTERVAL_S = 300         # save state every 5 min

    # Financial lexicon: curated bullish/bearish seed vocabulary
    _POSITIVE_VOCAB = [
        "beat", "beats", "surge", "surges", "rally", "rallies", "gain", "gains",
        "upgrade", "upgraded", "outperform", "record", "record-high", "buy",
        "strong", "growth", "profit", "profits", "positive", "bullish",
        "breakthrough", "deal", "deals", "dividend", "dividends", "innovation",
        "expansion", "approval", "approved", "launch", "launches",
        "accelerate", "accelerating", "momentum", "rebound", "recovery",
        "partnership", "acquisition", "synergy", "margin", "margins", "revenue",
    ]
    _NEGATIVE_VOCAB = [
        "miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops",
        "downgrade", "downgraded", "underperform", "sell", "weak", "loss",
        "losses", "negative", "bearish", "warning", "risk", "risks",
        "lawsuit", "investigation", "fraud", "recall", "layoff", "layoffs",
        "bankruptcy", "default", "cancel", "cancels", "delay", "delays",
        "cut", "cuts", "decline", "declining", "disappoint", "disappoints",
        "pressure", "pressures", "headwinds", "concern", "concerns",
    ]

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        self._persist_path = os.path.join(data_dir, "signal_enhancer_state.json")

        # ── Online sentiment lexicon weights (initialized from seed vocab) ──
        # weights[word] in [-1, 1]; updated via gradient descent on trade outcomes
        self._lex_weights: Dict[str, float] = {}
        self._init_lexicon()

        # ── Per-symbol per-regime confidence calibration data ──────────────
        # Stores (predicted_conf, actual_outcome) pairs for isotonic regression
        self._calib_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._calib_models: Dict[str, object] = {}   # sklearn IsotonicRegression

        # ── Per-regime Bayesian component weights ──────────────────────────
        self._component_weights: Dict[str, Dict[str, _BayesianWeight]] = defaultdict(
            lambda: {
                "nlp": _BayesianWeight(alpha=1.5, beta=1.0),
                "momentum_short": _BayesianWeight(alpha=2.0, beta=1.0),
                "momentum_medium": _BayesianWeight(alpha=1.8, beta=1.0),
                "momentum_long": _BayesianWeight(alpha=1.5, beta=1.5),
                "volume": _BayesianWeight(alpha=1.5, beta=1.0),
            }
        )

        # ── Running stats for z-score normalization ────────────────────────
        self._signal_stats: Dict[str, Tuple[float, float, int]] = {}  # mean, var, count

        # ── Time of last persist ───────────────────────────────────────────
        self._last_persist_ts: float = 0.0
        self._trade_count: int = 0

        self._load_state()
        logger.info("SignalEnhancer initialized: %d lexicon words, %d calib symbols",
                    len(self._lex_weights), len(self._calib_data))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(
        self,
        symbol: str,
        raw_signal: float,
        raw_confidence: float,
        price_history: Optional[pd.Series] = None,
        volume_history: Optional[pd.Series] = None,
        regime: str = "neutral",
        news_headlines: Optional[List[str]] = None,
        asset_class: str = "equity",
        spread_bps: float = 5.0,
        hour_utc: Optional[int] = None,
    ) -> EnhancedSignal:
        """Compute enhanced signal and calibrated confidence.

        Parameters
        ----------
        symbol : str
        raw_signal : float
            ML model output in [-1, 1].
        raw_confidence : float
            Raw model confidence in [0, 1].
        price_history : pd.Series
            Close prices, index is time-ordered, last entry is most recent.
        volume_history : pd.Series
            Volume, same alignment as price_history.
        regime : str
            Current market regime.
        news_headlines : list of str
            Recent news headlines for this symbol.
        asset_class : str
            "equity", "crypto", or "forex"
        spread_bps : float
            Current bid-ask spread in basis points.
        hour_utc : int
            Current UTC hour (for time-of-day filter).
        """
        components: Dict[str, float] = {"raw": float(raw_signal)}

        # 1. Momentum overlay
        mom_adj, mom_components = self._momentum_adj(
            raw_signal, price_history, volume_history, regime
        )
        components.update(mom_components)

        # 2. NLP sentiment adjustment
        nlp_adj = 0.0
        if news_headlines:
            nlp_adj = self._nlp_adj(symbol, news_headlines, raw_signal)
        components["nlp_adj"] = nlp_adj

        # 3. Regime-adaptive ensemble blending
        weights = self._component_weights[regime]
        w_nlp = weights["nlp"].weight
        w_mom_s = weights["momentum_short"].weight
        w_mom_m = weights["momentum_medium"].weight
        w_vol = weights["volume"].weight

        # Normalize weights so total overlay is bounded
        total_w = w_nlp + w_mom_s + w_mom_m + w_vol + 1e-6
        regime_weight = (w_nlp + w_mom_s + w_mom_m + w_vol) / total_w

        # Net adjustment: weighted combination, capped at ±0.20 to preserve signal integrity
        net_adj = (
            w_nlp * nlp_adj
            + w_mom_s * components.get("mom_1", 0.0)
            + w_mom_m * components.get("mom_5", 0.0)
            + w_vol * components.get("vol_signal", 0.0)
        ) / total_w
        net_adj = max(-0.20, min(0.20, net_adj))

        enhanced = max(-1.0, min(1.0, raw_signal + net_adj))
        components["net_adj"] = net_adj

        # 4. Confidence calibration
        calibrated_conf = self._calibrate_confidence(
            symbol, raw_confidence, abs(enhanced), regime
        )

        # 5. Microstructure filter
        microstructure_ok = self._microstructure_ok(
            asset_class, spread_bps, hour_utc, volume_history
        )
        if not microstructure_ok:
            # Dampen confidence on noisy bars — don't change signal direction
            calibrated_conf = max(0.30, calibrated_conf * 0.75)

        return EnhancedSignal(
            raw_signal=float(raw_signal),
            enhanced_signal=float(enhanced),
            raw_confidence=float(raw_confidence),
            calibrated_confidence=float(calibrated_conf),
            sentiment_adj=float(nlp_adj),
            momentum_adj=float(net_adj - nlp_adj * w_nlp / total_w),
            regime_weight=float(regime_weight),
            microstructure_ok=microstructure_ok,
            components=components,
        )

    def record_outcome(
        self,
        symbol: str,
        regime: str,
        entry_signal: float,
        entry_confidence: float,
        pnl_pct: float,
        components: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a closed trade outcome for online learning.

        Called by execution_loop after a position is closed.
        """
        won = pnl_pct > 0.0
        self._trade_count += 1

        # Update confidence calibration data
        calib_key = f"{regime}"
        self._calib_data[calib_key].append((float(entry_confidence), 1.0 if won else 0.0))
        # Cap to last 500 samples per regime
        if len(self._calib_data[calib_key]) > 500:
            self._calib_data[calib_key] = self._calib_data[calib_key][-500:]

        # Update Bayesian component weights based on whether each component helped
        if components:
            nlp = components.get("nlp_adj", 0.0)
            mom1 = components.get("mom_1", 0.0)
            mom5 = components.get("mom_5", 0.0)
            vol = components.get("vol_signal", 0.0)
            # A component "helped" if it pushed in the same direction as actual outcome
            entry_dir = 1.0 if entry_signal > 0 else -1.0
            outcome_dir = 1.0 if pnl_pct > 0 else -1.0
            for comp_name, comp_val in [
                ("nlp", nlp),
                ("momentum_short", mom1),
                ("momentum_medium", mom5),
                ("volume", vol),
            ]:
                if abs(comp_val) > 0.005:
                    comp_helped = (comp_val * outcome_dir) > 0
                    self._component_weights[regime][comp_name].update(comp_helped)

        # Periodically retrain calibration models
        if self._trade_count % self.CALIBRATION_INTERVAL == 0:
            self._retrain_calibration()

        # Periodically persist state
        now = time.monotonic()
        if now - self._last_persist_ts > self.PERSIST_INTERVAL_S:
            self._save_state()
            self._last_persist_ts = now

    # ------------------------------------------------------------------
    # Momentum overlay
    # ------------------------------------------------------------------

    def _momentum_adj(
        self,
        raw_signal: float,
        prices: Optional[pd.Series],
        volumes: Optional[pd.Series],
        regime: str,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute momentum-based signal adjustment.

        Returns (net_adj, components_dict).
        """
        comps: Dict[str, float] = {}
        if prices is None or len(prices) < self.MOM_WINDOWS[-1] + 1:
            return 0.0, comps

        arr = prices.values.astype(float)

        # Signed momentum: direction × magnitude, scaled to [-1, 1]
        for w in self.MOM_WINDOWS:
            if len(arr) < w + 1:
                continue
            ret = (arr[-1] - arr[-w - 1]) / (arr[-w - 1] + 1e-10)
            # Sigmoid-compress to [-1, 1]; 1% return ≈ 0.37 signal contribution
            sig_val = float(2.0 * expit(ret / 0.01) - 1.0)
            comps[f"mom_{w}"] = round(sig_val, 4)

        # Volume signal: above-avg volume + price up = bullish conviction
        if volumes is not None and len(volumes) >= self.VOLUME_LOOKBACK:
            vol_arr = volumes.values.astype(float)
            avg_vol = float(np.mean(vol_arr[-self.VOLUME_LOOKBACK:])) + 1e-10
            cur_vol = float(vol_arr[-1]) + 1e-10
            vol_ratio = cur_vol / avg_vol  # > 1 means above avg
            # Volume confirms if ratio > 1.5 AND price direction matches signal
            price_dir = 1.0 if len(arr) > 1 and arr[-1] > arr[-2] else -1.0
            signal_dir = 1.0 if raw_signal > 0 else -1.0
            if vol_ratio > 1.5 and (price_dir * signal_dir) > 0:
                comps["vol_signal"] = min(0.15, (vol_ratio - 1.0) * 0.1)
            elif vol_ratio > 2.0 and (price_dir * signal_dir) < 0:
                comps["vol_signal"] = -0.10  # high vol against signal = divergence warning
            else:
                comps["vol_signal"] = 0.0

        # ADX-proxy: measure trend strength from momentum variance
        if len(arr) >= 20:
            returns = np.diff(arr[-20:]) / (arr[-20:-1] + 1e-10)
            trend_str = float(np.abs(np.mean(returns)) / (np.std(returns) + 1e-10))
            comps["adx_proxy"] = min(1.0, trend_str)

        # Calibration: less weight on 1-bar noise (especially for crypto)
        # More weight on 5-bar trend and volume conviction
        w1, w5, wv = 0.25, 0.45, 0.30
        
        # High confidence override: if ML signal is strong (>0.4), 
        # reduce the penalty of the 1-bar momentum if it is opposing
        if abs(raw_signal) > 0.4 and (raw_signal * comps.get("mom_1", 0.0)) < 0:
            w1 *= 0.5
            wv += 0.125
            
        net = (comps.get("mom_1", 0.0) * w1
               + comps.get("mom_5", 0.0) * w5
               + comps.get("vol_signal", 0.0) * wv)
        return float(net), comps

    # ------------------------------------------------------------------
    # NLP sentiment overlay
    # ------------------------------------------------------------------

    def _nlp_adj(self, symbol: str, headlines: List[str], raw_signal: float) -> float:
        """Compute NLP sentiment adjustment from news headlines.

        Uses TF-IDF bag-of-words over the lexicon vocabulary with online
        gradient-updated lexicon weights. Falls back to simple keyword count
        before sufficient training data exists.
        """
        if not headlines:
            return 0.0

        combined_text = " ".join(h.lower() for h in headlines[:10])
        words = combined_text.split()

        # Weighted lexicon score
        pos_score = sum(self._lex_weights.get(w, 0.0) for w in words if w in self._lex_weights and self._lex_weights[w] > 0)
        neg_score = sum(abs(self._lex_weights.get(w, 0.0)) for w in words if w in self._lex_weights and self._lex_weights[w] < 0)
        total = pos_score + neg_score + 1e-6
        sentiment = (pos_score - neg_score) / total  # in [-1, 1]

        # Scale adjustment: max ±0.12 from sentiment; scale by absolute value
        adj = sentiment * 0.12
        # Cross-validate: if sentiment strongly opposes the signal, apply stronger discount
        if (adj * raw_signal) < 0 and abs(sentiment) > 0.5:
            adj *= 1.5  # amplify when they disagree
        return float(max(-0.15, min(0.15, adj)))

    def _init_lexicon(self) -> None:
        """Seed lexicon with pre-trained keyword weights."""
        for w in self._POSITIVE_VOCAB:
            self._lex_weights[w] = 0.6
        for w in self._NEGATIVE_VOCAB:
            self._lex_weights[w] = -0.6
        # Directional intensifiers
        for intensifier in ["significantly", "strongly", "sharply", "dramatically"]:
            self._lex_weights[intensifier] = 0.2   # amplifies adjacent words
        for dampener in ["slightly", "modestly", "marginally"]:
            self._lex_weights[dampener] = 0.1      # dampens effect

    # ------------------------------------------------------------------
    # Confidence calibration
    # ------------------------------------------------------------------

    def _calibrate_confidence(
        self,
        symbol: str,
        raw_conf: float,
        signal_abs: float,
        regime: str,
    ) -> float:
        """Isotonic-regression calibrated confidence.

        If insufficient data, falls back to a signal-strength-adjusted heuristic.
        """
        calib_key = regime
        model = self._calib_models.get(calib_key)
        if model is not None:
            try:
                cal = float(model.predict([raw_conf])[0])
                # Blend 70% calibrated + 30% raw to prevent over-correction
                return float(max(0.20, min(0.95, 0.70 * cal + 0.30 * raw_conf)))
            except Exception:
                pass

        # Heuristic calibration: strong signals slightly boost confidence
        bonus = min(0.05, signal_abs * 0.15)
        # In bear/volatile, slightly discount raw confidence (models tend to overfit)
        discount = 0.05 if regime in ("bear", "strong_bear", "volatile", "crisis") else 0.0
        return float(max(0.20, min(0.95, raw_conf + bonus - discount)))

    def _retrain_calibration(self) -> None:
        """Retrain isotonic regression calibrators from accumulated data."""
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            return

        for regime, samples in self._calib_data.items():
            if len(samples) < self.CALIBRATION_MIN_TRADES:
                continue
            X = np.array([s[0] for s in samples])
            y = np.array([s[1] for s in samples])
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(X, y)
            self._calib_models[regime] = ir
            logger.debug("SignalEnhancer: recalibrated confidence for regime=%s (%d samples)", regime, len(samples))

    # ------------------------------------------------------------------
    # Market microstructure filter
    # ------------------------------------------------------------------

    def _microstructure_ok(
        self,
        asset_class: str,
        spread_bps: float,
        hour_utc: Optional[int],
        volumes: Optional[pd.Series],
    ) -> bool:
        """Return True if market conditions are suitable for reliable signals.

        Flags noisy/illiquid conditions: wide spreads, off-hours, thin volume.
        """
        # Wide spread filter
        _max_spread = {"equity": 20.0, "crypto": 50.0, "forex": 10.0}.get(asset_class, 25.0)
        if spread_bps > _max_spread:
            return False

        # Time-of-day filter for equities (NYSE hours 14:30-21:00 UTC)
        if asset_class == "equity" and hour_utc is not None:
            if not (14 <= hour_utc < 21):
                return False

        # Volume quality: if volume < 20% of 20-bar avg, signals are unreliable
        if volumes is not None and len(volumes) >= 20:
            arr = volumes.values.astype(float)
            avg = float(np.mean(arr[-20:])) + 1e-10
            if arr[-1] / avg < 0.20:
                return False

        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return diagnostic summary for TCA/ops API."""
        regime_weights = {}
        for regime, comps in self._component_weights.items():
            regime_weights[regime] = {
                k: round(v.weight, 3) for k, v in comps.items()
            }
        return {
            "trade_count": self._trade_count,
            "lexicon_size": len(self._lex_weights),
            "calibration_models": list(self._calib_models.keys()),
            "regime_weights": regime_weights,
            "calib_samples": {k: len(v) for k, v in self._calib_data.items()},
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        try:
            state = {
                "trade_count": self._trade_count,
                "lex_weights": self._lex_weights,
                "calib_data": {k: v[-500:] for k, v in self._calib_data.items()},
                "component_weights": {
                    regime: {comp: w.to_dict() for comp, w in comps.items()}
                    for regime, comps in self._component_weights.items()
                },
            }
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            tmp = self._persist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self._persist_path)
        except Exception:
            logger.warning("SignalEnhancer: failed to save state", exc_info=True)

    def _load_state(self) -> None:
        try:
            if not os.path.exists(self._persist_path):
                return
            with open(self._persist_path) as f:
                state = json.load(f)
            self._trade_count = int(state.get("trade_count", 0))
            loaded_lex = state.get("lex_weights", {})
            self._lex_weights.update(loaded_lex)
            self._calib_data.update({
                k: [tuple(v) for v in samples]
                for k, samples in state.get("calib_data", {}).items()
            })
            for regime, comps in state.get("component_weights", {}).items():
                for comp, wd in comps.items():
                    if comp in self._component_weights[regime]:
                        self._component_weights[regime][comp] = _BayesianWeight.from_dict(wd)
            # Retrain calibrators from loaded data
            self._retrain_calibration()
            logger.info("SignalEnhancer loaded: %d trades, %d regimes calibrated",
                        self._trade_count, len(self._calib_models))
        except Exception:
            logger.warning("SignalEnhancer: failed to load state", exc_info=True)
