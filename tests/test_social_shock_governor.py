from market.prediction_market_verifier import PredictionVerificationResult
from risk.social_risk_factor import SocialRiskSnapshot
from risk.social_shock_governor import SocialShockGovernor, SocialShockGovernorConfig


def test_social_shock_governor_blocks_entries_on_extreme_combined_risk():
    gov = SocialShockGovernor(
        SocialShockGovernorConfig(
            reduce_threshold=0.6,
            block_threshold=0.8,
            min_gross_exposure_multiplier=0.3,
            verified_event_weight=0.35,
            verified_event_probability_floor=0.55,
        )
    )
    snapshot = SocialRiskSnapshot(
        timestamp="2026-02-16T00:00:00",
        asset_class="EQUITY",
        regime="risk_off",
        platform_count=4,
        attention_z=3.0,
        sentiment_score=-0.8,
        confidence=0.9,
        risk_score=0.72,
        reasons=["attention_extreme", "sentiment_extreme"],
    )
    decision = gov.evaluate(
        snapshot,
        [
            PredictionVerificationResult(
                event_id="macro_event",
                direction="risk_off",
                verified=True,
                verified_probability=0.86,
                reason="verified",
            )
        ],
    )
    assert decision.block_new_entries
    assert decision.gross_exposure_multiplier == 0.3
    assert decision.combined_risk_score >= 0.8
    assert decision.policy_version == "runtime-config"


def test_social_shock_governor_tracks_unverified_predictions_without_boosting():
    gov = SocialShockGovernor()
    snapshot = SocialRiskSnapshot(
        timestamp="2026-02-16T00:00:00",
        asset_class="CRYPTO",
        regime="high_vol",
        platform_count=4,
        attention_z=1.2,
        sentiment_score=-0.2,
        confidence=0.8,
        risk_score=0.35,
        reasons=["social_conditions_normal"],
    )
    decision = gov.evaluate(
        snapshot,
        [
            PredictionVerificationResult(
                event_id="meme_shock",
                direction="risk_off",
                verified=False,
                verified_probability=0.0,
                reason="market_vs_independent_divergence",
            )
        ],
    )
    assert not decision.block_new_entries
    assert decision.verified_event_probability == 0.0
    assert decision.prediction_verification_failures == 1
    assert decision.policy_version == "runtime-config"
