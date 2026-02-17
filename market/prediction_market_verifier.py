"""
market/prediction_market_verifier.py

Treats prediction-market probabilities as noisy and only verifies
them for sizing/risk use after independent corroboration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class PredictionMarketVerificationConfig:
    min_independent_sources: int = 2
    max_probability_divergence: float = 0.15
    max_source_disagreement: float = 0.20
    minimum_market_probability: float = 0.05


@dataclass(frozen=True)
class PredictionEventInput:
    event_id: str
    market_probability: float
    independent_probability: float
    independent_source_count: int
    max_source_disagreement: float
    direction: str = "risk_off"


@dataclass(frozen=True)
class PredictionVerificationResult:
    event_id: str
    direction: str
    verified: bool
    verified_probability: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_id": self.event_id,
            "direction": self.direction,
            "verified": self.verified,
            "verified_probability": self.verified_probability,
            "reason": self.reason,
        }


class PredictionMarketVerificationGate:
    """Validation gate for prediction-market probabilities."""

    def __init__(self, config: PredictionMarketVerificationConfig | None = None):
        self.config = config or PredictionMarketVerificationConfig()

    def verify(self, event: PredictionEventInput) -> PredictionVerificationResult:
        market_prob = _clamp(float(event.market_probability), 0.0, 1.0)
        independent_prob = _clamp(float(event.independent_probability), 0.0, 1.0)
        source_count = int(event.independent_source_count)
        disagreement = _clamp(float(event.max_source_disagreement), 0.0, 1.0)
        direction = str(event.direction or "risk_off").lower()
        event_id = str(event.event_id or "unknown_event")

        if source_count < self.config.min_independent_sources:
            return PredictionVerificationResult(
                event_id=event_id,
                direction=direction,
                verified=False,
                verified_probability=0.0,
                reason="insufficient_independent_sources",
            )

        if disagreement > self.config.max_source_disagreement:
            return PredictionVerificationResult(
                event_id=event_id,
                direction=direction,
                verified=False,
                verified_probability=0.0,
                reason="independent_sources_disagree",
            )

        if abs(market_prob - independent_prob) > self.config.max_probability_divergence:
            return PredictionVerificationResult(
                event_id=event_id,
                direction=direction,
                verified=False,
                verified_probability=0.0,
                reason="market_vs_independent_divergence",
            )

        if market_prob < self.config.minimum_market_probability:
            return PredictionVerificationResult(
                event_id=event_id,
                direction=direction,
                verified=False,
                verified_probability=0.0,
                reason="probability_below_noise_floor",
            )

        # Use conservative blend to avoid over-trusting a single source family.
        verified_prob = (market_prob + independent_prob) / 2.0
        return PredictionVerificationResult(
            event_id=event_id,
            direction=direction,
            verified=True,
            verified_probability=_clamp(verified_prob, 0.0, 1.0),
            reason="verified",
        )
