"""
models/market_intelligence.py  —  Unified Market Intelligence Engine

Synthesises ALL available market signals into a single coherent assessment
of the current market environment.

Unlike individual gates that check signals in isolation, this engine:
  • Looks at ALL signals simultaneously
  • Detects coherent narratives
    (e.g. "risk-off: yield inversion + BTC/SPY both falling + extreme fear")
  • Builds a human-readable description of the dominant market theme
  • Assigns a quality score that the AdaptiveMetaController uses to enrich
    its TradeContext

No API keys.  All inputs come from existing system state.
No config values — the engine is fully self-contained and interprets raw data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketIntelligence:
    """Direction-independent assessment of the current market environment."""

    # Overall environment quality [-1, 1]
    # +1 = clear, supportive, coherent signals
    #  0 = neutral / unclear
    # -1 = hostile, contradicting, high-noise environment
    environment_score: float = 0.0

    # Risk appetite [-1, 1]: positive = risk-on, negative = risk-off
    risk_appetite: float = 0.0

    # How much do all sources agree? [0, 1]
    signal_coherence: float = 0.5

    # Human-readable dominant narrative
    narrative: str = "neutral"

    # Individual factor scores (for logging/transparency)
    factors: Dict[str, float] = field(default_factory=dict)

    @property
    def is_risk_on(self) -> bool:
        return self.risk_appetite > 0.20

    @property
    def is_risk_off(self) -> bool:
        return self.risk_appetite < -0.20

    @property
    def is_high_quality(self) -> bool:
        return self.environment_score > 0.30 and self.signal_coherence > 0.55

    @property
    def is_hostile(self) -> bool:
        return self.environment_score < -0.30


class MarketIntelligenceEngine:
    """
    Synthesises all available market context into a unified MarketIntelligence.

    Inputs are all optional — missing inputs default to neutral (0.0 / False).
    The engine degrades gracefully and always produces a valid result.

    The factor weights below are reasonable starting priors.  Over time the
    AdaptiveMetaController learns which sources actually predict outcomes and
    adjusts accordingly — no manual tuning needed.
    """

    # Source weights (sum ≈ 1).  These are PRIORS only — the meta-controller
    # learns the true weights from live trade outcomes.
    _WEIGHTS: Dict[str, float] = {
        "regime":        0.18,
        "news":          0.16,
        "macro":         0.15,
        "cross_asset":   0.13,
        "ofi":           0.12,
        "funding":       0.09,
        "pattern":       0.08,
        "vol":           0.06,
        "concentration": 0.03,
    }

    def __init__(self) -> None:
        logger.info("MarketIntelligenceEngine ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(
        self,
        # Regime
        regime:            str   = "neutral",
        regime_conviction: float = 0.5,
        # News / sentiment
        news_sentiment:    float         = 0.0,
        news_confidence:   float         = 0.0,
        news_momentum:     float         = 0.0,   # positive = improving
        fear_greed_index:  Optional[int] = None,  # 0-100; None = unavailable
        # Macro
        yield_curve_slope: float = 0.0,    # positive = normal, negative = inverted
        vix_structure:     float = 0.95,   # <1 = contango (calm), >1 = backwardation (stress)
        dxy_momentum:      float = 0.0,    # positive = strong USD (risk-off)
        # Cross-asset
        btc_signal: float = 0.0,
        spy_signal: float = 0.0,
        # Order flow & positioning
        ofi:                  float = 0.0,
        funding_rate_signal:  float = 0.0,  # crypto: -1 = longs crowded (bearish), +1 = shorts crowded
        # Technicals
        pattern_signal: float = 0.0,
        # Portfolio state
        vol_percentile: float = 0.5,
        hhi:            float = 0.0,
        daily_loss_pct: float = 0.0,
    ) -> MarketIntelligence:
        """
        Produce a unified market intelligence snapshot.

        All inputs are optional; any unavailable signal should be left at its
        default (0.0 / False) so it contributes a neutral score.
        """
        factors: Dict[str, float] = {}
        narrative_parts: List[str] = []

        # ── 1. Regime quality ─────────────────────────────────────────────
        _REGIME_MAP = {
            "strong_bull": 1.0, "bull": 0.6, "neutral": 0.0,
            "bear": -0.6, "strong_bear": -1.0,
            "volatile": -0.3, "crisis": -0.8,
        }
        regime_direction = _REGIME_MAP.get(str(regime).lower(), 0.0)
        regime_score     = float(regime_direction * regime_conviction)
        factors["regime"] = regime_score
        if abs(regime_score) > 0.35:
            narrative_parts.append(
                f"regime={regime}(conviction={regime_conviction:.1f})"
            )

        # ── 2. News & sentiment ───────────────────────────────────────────
        if fear_greed_index is not None:
            fg_norm = (float(fear_greed_index) - 50.0) / 50.0
            # Extreme readings (fear / greed) are more informative
            extreme_weight = min(1.0, abs(fg_norm) * 1.8)
            news_raw = news_sentiment * (1.0 - extreme_weight * 0.4) + fg_norm * (extreme_weight * 0.4)
            news_raw = news_raw * max(news_confidence, 0.3)
            if abs(fg_norm) > 0.6:
                narrative_parts.append(
                    f"fear_greed={'extreme_greed' if fg_norm > 0 else 'extreme_fear'}({fear_greed_index})"
                )
        else:
            news_raw = float(news_sentiment) * float(news_confidence)
        # News momentum (improving vs deteriorating adds directional info)
        news_raw += float(news_momentum) * 0.25
        factors["news"] = float(np.clip(news_raw, -1.0, 1.0))

        # ── 3. Macro environment ──────────────────────────────────────────
        macro_s = 0.0
        if yield_curve_slope < -0.001:           # inverted → recession risk
            macro_s -= 0.5
            narrative_parts.append("yield_curve_inverted")
        elif yield_curve_slope > 0.015:          # steep → growth
            macro_s += 0.3
        if float(vix_structure) > 1.0:           # backwardation → stress
            macro_s -= 0.4
            narrative_parts.append("vix_backwardation")
        if float(dxy_momentum) > 0.02:           # strong USD → risk-off
            macro_s -= 0.3
            narrative_parts.append("strong_USD")
        elif float(dxy_momentum) < -0.02:        # weak USD → risk-on
            macro_s += 0.2
        factors["macro"] = float(np.clip(macro_s, -1.0, 1.0))

        # ── 4. Cross-asset coherence ──────────────────────────────────────
        if abs(btc_signal) > 0.05 and abs(spy_signal) > 0.05:
            if btc_signal * spy_signal > 0:      # both same direction
                mag = (abs(btc_signal) + abs(spy_signal)) / 2.0
                cross_s = float(np.tanh(mag * 3.0)) * np.sign(btc_signal)
                label = "BTC+SPY_risk_on" if btc_signal > 0 else "BTC+SPY_risk_off"
                narrative_parts.append(label)
            else:                                # diverging → uncertainty
                cross_s = -0.30
                narrative_parts.append("BTC_SPY_diverging")
        elif abs(btc_signal) > 0.05:
            cross_s = float(np.tanh(btc_signal * 2.0)) * 0.5
        elif abs(spy_signal) > 0.05:
            cross_s = float(np.tanh(spy_signal * 2.0)) * 0.5
        else:
            cross_s = 0.0
        factors["cross_asset"] = float(np.clip(cross_s, -1.0, 1.0))

        # ── 5. Order flow (smart money positioning) ───────────────────────
        factors["ofi"] = float(np.clip(np.tanh(ofi * 3.0), -1.0, 1.0))

        # ── 6. Funding rate (crypto crowd positioning) ────────────────────
        # funding_rate_signal is already direction-encoded:
        #   negative = longs crowded (contrarian bearish if we're long)
        factors["funding"] = float(np.clip(funding_rate_signal, -1.0, 1.0))
        if abs(funding_rate_signal) > 0.5:
            label = "longs_crowded" if funding_rate_signal < 0 else "shorts_crowded"
            narrative_parts.append(f"funding:{label}")

        # ── 7. Candlestick patterns ───────────────────────────────────────
        factors["pattern"] = float(np.clip(pattern_signal, -1.0, 1.0))

        # ── 8. Volatility regime ──────────────────────────────────────────
        if vol_percentile < 0.25:
            vol_s = 0.3            # calm = supportive
        elif vol_percentile > 0.85:
            vol_s = -0.4           # spike = uncertain
            narrative_parts.append("vol_spike")
        else:
            vol_s = 0.0
        factors["vol"] = vol_s

        # ── 9. Portfolio concentration ────────────────────────────────────
        conc_s = -float(np.tanh(max(0.0, hhi - 0.20) * 5.0))
        factors["concentration"] = conc_s

        # ── Weighted aggregate environment score ──────────────────────────
        env_score = sum(
            self._WEIGHTS.get(k, 0.0) * v for k, v in factors.items()
        )
        env_score = float(np.clip(env_score, -1.0, 1.0))

        # ── Risk appetite (regime + news + macro + cross-asset) ───────────
        risk_apt = (
            regime_score             * 0.35
            + factors["news"]        * 0.20
            + factors["macro"]       * 0.25
            + factors["cross_asset"] * 0.20
        )
        risk_apt = float(np.clip(risk_apt, -1.0, 1.0))

        # ── Signal coherence: how many sources agree on direction? ─────────
        scored = [v for k, v in factors.items() if k not in ("vol", "concentration")]
        if scored:
            n_pos = sum(1 for v in scored if v >  0.05)
            n_neg = sum(1 for v in scored if v < -0.05)
            coherence = float(max(n_pos, n_neg)) / max(len(scored), 1)
        else:
            coherence = 0.5

        # ── Build narrative ───────────────────────────────────────────────
        if daily_loss_pct < -0.02:
            narrative_parts.insert(0, f"drawdown={daily_loss_pct:.1%}")
        if not narrative_parts:
            narrative = "neutral_environment"
        elif env_score > 0.30:
            narrative = "supportive: " + ", ".join(narrative_parts[:3])
        elif env_score < -0.30:
            narrative = "hostile: " + ", ".join(narrative_parts[:3])
        else:
            narrative = "mixed: " + ", ".join(narrative_parts[:2]) if narrative_parts else "neutral"

        intel = MarketIntelligence(
            environment_score=env_score,
            risk_appetite=risk_apt,
            signal_coherence=coherence,
            narrative=narrative,
            factors=factors,
        )
        logger.debug(
            "MarketIntel: env=%.3f risk_apt=%.3f coherence=%.2f | %s",
            env_score, risk_apt, coherence, narrative,
        )
        return intel
