"""
risk/social_risk_factor.py

Cross-platform social sentiment as a first-class risk factor.
Converts noisy platform inputs into a bounded risk score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Mapping, Optional


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class SocialRiskConfig:
    """Configuration for social risk normalization."""

    attention_trigger_z: float = 1.0
    attention_extreme_z: float = 3.0
    negative_sentiment_trigger: float = -0.35
    positive_sentiment_trigger: float = 0.75
    attention_weight: float = 0.60
    sentiment_weight: float = 0.40
    min_platforms: int = 2
    min_confidence: float = 0.15


@dataclass(frozen=True)
class SocialRiskSnapshot:
    """Normalized social risk snapshot for a specific asset class + regime."""

    timestamp: str
    asset_class: str
    regime: str
    platform_count: int
    attention_z: float
    sentiment_score: float
    confidence: float
    risk_score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "asset_class": self.asset_class,
            "regime": self.regime,
            "platform_count": int(self.platform_count),
            "attention_z": float(self.attention_z),
            "sentiment_score": float(self.sentiment_score),
            "confidence": float(self.confidence),
            "risk_score": float(self.risk_score),
            "reasons": list(self.reasons),
        }


class SocialRiskFactor:
    """Aggregates platform attention/sentiment into a bounded risk score [0, 1]."""

    def __init__(self, config: Optional[SocialRiskConfig] = None):
        self.config = config or SocialRiskConfig()

    def evaluate(
        self,
        *,
        asset_class: str,
        regime: str,
        platform_signals: Mapping[str, Mapping[str, object]],
        observed_at: Optional[datetime] = None,
    ) -> SocialRiskSnapshot:
        aggregate = self._aggregate_platforms(platform_signals)
        reasons: list[str] = []

        attention_pressure = self._attention_pressure(aggregate["attention_z"])
        sentiment_pressure = self._sentiment_pressure(aggregate["sentiment_score"])
        score = _clamp(
            self.config.attention_weight * attention_pressure
            + self.config.sentiment_weight * sentiment_pressure,
            0.0,
            1.0,
        )
        if aggregate["platform_count"] < self.config.min_platforms:
            # Degrade score confidence when feed breadth is weak.
            score *= 0.5
            reasons.append("insufficient_platform_coverage")

        if attention_pressure >= 0.8:
            reasons.append("attention_extreme")
        if sentiment_pressure >= 0.8:
            reasons.append("sentiment_extreme")
        if not reasons:
            reasons.append("social_conditions_normal")

        ts = (observed_at or datetime.utcnow()).isoformat()
        return SocialRiskSnapshot(
            timestamp=ts,
            asset_class=str(asset_class).upper(),
            regime=str(regime).lower(),
            platform_count=aggregate["platform_count"],
            attention_z=aggregate["attention_z"],
            sentiment_score=aggregate["sentiment_score"],
            confidence=aggregate["confidence"],
            risk_score=score,
            reasons=reasons,
        )

    def _aggregate_platforms(self, platform_signals: Mapping[str, Mapping[str, object]]) -> Dict[str, float]:
        weighted_attention = 0.0
        weighted_sentiment = 0.0
        total_weight = 0.0
        platform_count = 0

        for payload in (platform_signals or {}).values():
            if not isinstance(payload, Mapping):
                continue
            try:
                attention_z = float(payload.get("attention_z", 0.0) or 0.0)
                sentiment_score = float(payload.get("sentiment_score", 0.0) or 0.0)
                confidence = float(payload.get("confidence", 1.0) or 1.0)
            except (TypeError, ValueError):
                continue

            confidence = _clamp(confidence, self.config.min_confidence, 1.0)
            sentiment_score = _clamp(sentiment_score, -1.0, 1.0)
            attention_z = _clamp(attention_z, -5.0, 8.0)
            weighted_attention += attention_z * confidence
            weighted_sentiment += sentiment_score * confidence
            total_weight += confidence
            platform_count += 1

        if total_weight <= 0:
            return {
                "attention_z": 0.0,
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "platform_count": 0,
            }

        return {
            "attention_z": weighted_attention / total_weight,
            "sentiment_score": weighted_sentiment / total_weight,
            "confidence": _clamp(total_weight / max(1.0, float(platform_count)), 0.0, 1.0),
            "platform_count": platform_count,
        }

    def _attention_pressure(self, attention_z: float) -> float:
        lo = self.config.attention_trigger_z
        hi = max(lo + 0.01, self.config.attention_extreme_z)
        return _clamp((attention_z - lo) / (hi - lo), 0.0, 1.0)

    def _sentiment_pressure(self, sentiment_score: float) -> float:
        negative_pressure = _clamp(
            (
                self.config.negative_sentiment_trigger - sentiment_score
            ) / max(0.01, abs(self.config.negative_sentiment_trigger + 1.0)),
            0.0,
            1.0,
        )
        positive_pressure = _clamp(
            (
                sentiment_score - self.config.positive_sentiment_trigger
            ) / max(0.01, 1.0 - self.config.positive_sentiment_trigger),
            0.0,
            1.0,
        )
        # Downside sentiment shocks should carry more risk weight than euphoric spikes.
        return max(negative_pressure, positive_pressure * 0.75)
