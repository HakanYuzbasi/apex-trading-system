"""
risk/social_shock_governor.py

Converts social risk + verified event probabilities into
entry block / gross exposure throttling decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from market.prediction_market_verifier import PredictionVerificationResult
from risk.social_risk_factor import SocialRiskSnapshot


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class SocialShockGovernorConfig:
    reduce_threshold: float = 0.60
    block_threshold: float = 0.85
    min_gross_exposure_multiplier: float = 0.35
    verified_event_weight: float = 0.30
    verified_event_probability_floor: float = 0.55


@dataclass(frozen=True)
class SocialShockDecision:
    asset_class: str
    regime: str
    policy_version: str
    social_risk_score: float
    combined_risk_score: float
    gross_exposure_multiplier: float
    block_new_entries: bool
    verified_event_probability: float
    prediction_verification_failures: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "asset_class": self.asset_class,
            "regime": self.regime,
            "policy_version": self.policy_version,
            "social_risk_score": self.social_risk_score,
            "combined_risk_score": self.combined_risk_score,
            "gross_exposure_multiplier": self.gross_exposure_multiplier,
            "block_new_entries": self.block_new_entries,
            "verified_event_probability": self.verified_event_probability,
            "prediction_verification_failures": self.prediction_verification_failures,
            "reasons": list(self.reasons),
        }


class SocialShockGovernor:
    """Policy engine for social-shock and event-risk gating."""

    def __init__(self, config: SocialShockGovernorConfig | None = None):
        self.config = config or SocialShockGovernorConfig()

    def evaluate(
        self,
        snapshot: SocialRiskSnapshot,
        prediction_results: Sequence[PredictionVerificationResult] | None = None,
        policy_version: str = "runtime-config",
    ) -> SocialShockDecision:
        prediction_results = list(prediction_results or [])
        reasons = list(snapshot.reasons)

        verified_risk_off_prob = 0.0
        verification_failures = 0
        for result in prediction_results:
            if not result.verified:
                verification_failures += 1
                reasons.append(f"prediction_unverified:{result.event_id}:{result.reason}")
                continue
            if result.direction == "risk_off":
                verified_risk_off_prob = max(verified_risk_off_prob, result.verified_probability)
                reasons.append(f"prediction_verified:{result.event_id}:{result.verified_probability:.2f}")

        event_boost = 0.0
        if verified_risk_off_prob > self.config.verified_event_probability_floor:
            event_boost = (
                (verified_risk_off_prob - self.config.verified_event_probability_floor)
                / max(0.01, 1.0 - self.config.verified_event_probability_floor)
            ) * self.config.verified_event_weight

        combined = _clamp(snapshot.risk_score + event_boost, 0.0, 1.0)
        block_entries = combined >= self.config.block_threshold
        gross_multiplier = 1.0

        if block_entries:
            gross_multiplier = self.config.min_gross_exposure_multiplier
            reasons.append("social_shock_block")
        elif combined >= self.config.reduce_threshold:
            progress = (
                (combined - self.config.reduce_threshold)
                / max(0.01, self.config.block_threshold - self.config.reduce_threshold)
            )
            gross_multiplier = 1.0 - progress * (
                1.0 - self.config.min_gross_exposure_multiplier
            )
            gross_multiplier = _clamp(
                gross_multiplier,
                self.config.min_gross_exposure_multiplier,
                1.0,
            )
            reasons.append("social_shock_reduce")
        else:
            reasons.append("social_shock_normal")

        return SocialShockDecision(
            asset_class=snapshot.asset_class,
            regime=snapshot.regime,
            policy_version=str(policy_version or "runtime-config"),
            social_risk_score=float(snapshot.risk_score),
            combined_risk_score=float(combined),
            gross_exposure_multiplier=float(gross_multiplier),
            block_new_entries=bool(block_entries),
            verified_event_probability=float(verified_risk_off_prob),
            prediction_verification_failures=int(verification_failures),
            reasons=reasons,
        )
