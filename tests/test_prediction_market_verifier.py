from market.prediction_market_verifier import (
    PredictionEventInput,
    PredictionMarketVerificationConfig,
    PredictionMarketVerificationGate,
)


def test_prediction_market_verifier_rejects_when_sources_insufficient():
    gate = PredictionMarketVerificationGate(
        PredictionMarketVerificationConfig(min_independent_sources=2)
    )
    result = gate.verify(
        PredictionEventInput(
            event_id="fed_cuts_soon",
            market_probability=0.72,
            independent_probability=0.68,
            independent_source_count=1,
            max_source_disagreement=0.05,
            direction="risk_off",
        )
    )
    assert not result.verified
    assert result.reason == "insufficient_independent_sources"


def test_prediction_market_verifier_accepts_when_corroborated():
    gate = PredictionMarketVerificationGate(
        PredictionMarketVerificationConfig(
            min_independent_sources=2,
            max_probability_divergence=0.2,
            max_source_disagreement=0.2,
        )
    )
    result = gate.verify(
        PredictionEventInput(
            event_id="earnings_recession_shock",
            market_probability=0.66,
            independent_probability=0.62,
            independent_source_count=3,
            max_source_disagreement=0.07,
            direction="risk_off",
        )
    )
    assert result.verified
    assert result.reason == "verified"
    assert 0.6 <= result.verified_probability <= 0.7
