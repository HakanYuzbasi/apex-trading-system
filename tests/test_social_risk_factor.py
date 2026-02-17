from risk.social_risk_factor import SocialRiskConfig, SocialRiskFactor


def test_social_risk_factor_scores_extreme_attention_and_sentiment_high():
    factor = SocialRiskFactor(
        SocialRiskConfig(
            attention_trigger_z=1.0,
            attention_extreme_z=3.0,
            negative_sentiment_trigger=-0.30,
            positive_sentiment_trigger=0.75,
        )
    )

    snapshot = factor.evaluate(
        asset_class="EQUITY",
        regime="risk_off",
        platform_signals={
            "X": {"attention_z": 3.2, "sentiment_score": -0.85, "confidence": 0.9},
            "TIKTOK": {"attention_z": 2.9, "sentiment_score": -0.75, "confidence": 0.8},
            "INSTAGRAM": {"attention_z": 3.1, "sentiment_score": -0.7, "confidence": 0.7},
            "YOUTUBE": {"attention_z": 2.7, "sentiment_score": -0.8, "confidence": 0.85},
        },
    )

    assert snapshot.risk_score > 0.8
    assert snapshot.platform_count == 4
    assert "attention_extreme" in snapshot.reasons


def test_social_risk_factor_penalizes_insufficient_platform_coverage():
    factor = SocialRiskFactor(SocialRiskConfig(min_platforms=3))
    snapshot = factor.evaluate(
        asset_class="FOREX",
        regime="carry",
        platform_signals={
            "X": {"attention_z": 2.2, "sentiment_score": -0.6, "confidence": 0.8},
        },
    )
    assert snapshot.platform_count == 1
    assert "insufficient_platform_coverage" in snapshot.reasons
    assert snapshot.risk_score < 0.5
