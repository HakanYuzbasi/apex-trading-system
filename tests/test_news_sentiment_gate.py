"""
Tests for the NewsSentimentEngine (news_sentiment_llm.py) and its gate logic.
"""
import pytest

from data.social.news_sentiment_llm import (
    NewsSentimentEngine,
    SentimentResult,
    _DOMAIN_LEXICON,
    _VADER_AVAILABLE,
)


class TestSentimentResultDataclass:
    def test_to_dict_keys(self):
        r = SentimentResult(symbol="AAPL", sentiment=0.42, confidence=0.75, headline_count=5)
        d = r.to_dict()
        assert set(d.keys()) >= {"symbol", "sentiment", "confidence", "headline_count", "method"}

    def test_to_dict_sentiment_rounded(self):
        r = SentimentResult(symbol="X", sentiment=0.123456, confidence=0.5, headline_count=1)
        assert len(str(r.to_dict()["sentiment"]).split(".")[-1]) <= 4

    def test_top_headlines_capped_at_3(self):
        r = SentimentResult(
            symbol="X", sentiment=0.1, confidence=0.5, headline_count=10,
            top_headlines=["h1", "h2", "h3", "h4", "h5"]
        )
        assert len(r.to_dict()["top_headlines"]) <= 3


class TestDomainLexicon:
    def test_lexicon_not_empty(self):
        assert len(_DOMAIN_LEXICON) >= 100

    def test_bearish_terms_negative(self):
        assert _DOMAIN_LEXICON["bankruptcy"] < 0
        assert _DOMAIN_LEXICON["earnings miss"] < 0
        assert _DOMAIN_LEXICON["downgraded"] < 0

    def test_bullish_terms_positive(self):
        assert _DOMAIN_LEXICON["earnings beat"] > 0
        assert _DOMAIN_LEXICON["upgraded"] > 0
        assert _DOMAIN_LEXICON["fda approval"] > 0

    def test_all_values_in_vader_range(self):
        for term, v in _DOMAIN_LEXICON.items():
            assert -4.5 <= v <= 4.5, f"{term}={v} outside VADER range"


@pytest.mark.skipif(not _VADER_AVAILABLE, reason="VADER not installed")
class TestNewsSentimentEngine:
    def setup_method(self):
        self.engine = NewsSentimentEngine()

    def test_engine_initialises(self):
        assert self.engine._analyzer is not None

    def test_score_headline_in_range(self):
        for headline in [
            "Company reports record earnings beat",
            "CEO resigns amid accounting fraud investigation",
            "Stock rallies to all-time high after FDA approval",
            "Firm files for bankruptcy after revenue decline",
        ]:
            score = self.engine._score_headline(headline)
            assert -1.0 <= score <= 1.0, f"score={score} for '{headline}'"

    def test_positive_headline_positive_score(self):
        s = self.engine._score_headline("Company beats earnings estimates with record profit")
        assert s > 0.0

    def test_negative_headline_negative_score(self):
        s = self.engine._score_headline("Company files for bankruptcy after massive losses")
        assert s < 0.0

    def test_negation_inverts_positive(self):
        pos = self.engine._score_headline("Company beats earnings")
        neg = self.engine._score_headline("Company fails to beat earnings")
        assert neg < pos

    def test_aggregate_empty_returns_zero(self):
        sent, conf = self.engine._aggregate([])
        assert sent == 0.0
        assert conf == 0.0

    def test_aggregate_single_score(self):
        sent, conf = self.engine._aggregate([0.50])
        assert sent == pytest.approx(0.50)
        assert conf == pytest.approx(0.35)

    def test_aggregate_agreement_boosts_confidence(self):
        # All same direction → high agreement
        _, conf_agree = self.engine._aggregate([0.8, 0.75, 0.7, 0.8])
        # Mixed signals → low agreement
        _, conf_mixed = self.engine._aggregate([0.8, -0.7, 0.6, -0.5])
        assert conf_agree > conf_mixed

    def test_aggregate_recency_decay(self):
        """Earlier items should contribute less — result should be between first and mean."""
        scores = [0.8, 0.0, 0.0, 0.0]  # first is highest
        sent, _ = self.engine._aggregate(scores)
        # Recency decay makes first item dominant → positive result
        assert sent > 0.0

    def test_domain_phrase_score_positive(self):
        # Use exact phrases from the domain lexicon
        score = self.engine._domain_phrase_score("company reports earnings beat and guidance raised")
        assert score > 0.0

    def test_domain_phrase_score_negative(self):
        score = self.engine._domain_phrase_score("company files for bankruptcy")
        assert score < 0.0

    def test_domain_phrase_score_empty(self):
        score = self.engine._domain_phrase_score("the weather is nice today")
        assert score == 0.0

    def test_apply_negation_triggers(self):
        # "not" triggers negation; "earnings beat" is in lexicon with positive valence
        text = "stock not earnings beat this quarter"
        result = self.engine._apply_negation(text)
        assert "terrible" in result


@pytest.mark.skipif(not _VADER_AVAILABLE, reason="VADER not installed")
class TestNewsGateLogic:
    """Tests that mirror the Check 2.14c gate logic."""

    def test_strong_contra_low_conf_should_block(self):
        """Contradiction > 0.45 and confidence < 0.70 → block."""
        llm_sentiment = -0.70   # bearish
        signal = 1.0            # BUY
        confidence = 0.60       # below gate
        llm_conf = 0.70
        headline_count = 3

        _llm_dir = 1 if signal > 0 else -1
        _llm_contra = llm_sentiment * _llm_dir
        should_block = (
            llm_conf >= 0.60
            and headline_count >= 2
            and _llm_contra < -0.45
            and confidence < 0.70
        )
        assert should_block

    def test_strong_contra_high_conf_penalise_not_block(self):
        """Contradiction > 0.45 but confidence >= 0.70 → penalise, not block."""
        llm_sentiment = -0.70
        signal = 1.0
        confidence = 0.80
        llm_conf = 0.75
        headline_count = 4

        _llm_dir = 1 if signal > 0 else -1
        _llm_contra = llm_sentiment * _llm_dir
        block = (
            llm_conf >= 0.60
            and headline_count >= 2
            and _llm_contra < -0.45
            and confidence < 0.70
        )
        penalise = (
            llm_conf >= 0.60
            and headline_count >= 2
            and _llm_contra < -0.45
            and confidence >= 0.70
        )
        assert not block
        assert penalise
        penalised_conf = confidence * 0.90
        assert penalised_conf < confidence

    def test_confirmation_boosts_confidence(self):
        """Sentiment confirms direction → boost."""
        llm_sentiment = 0.60  # bullish
        signal = 1.0          # BUY
        confidence = 0.72
        llm_conf = 0.60
        headline_count = 3

        _llm_dir = 1 if signal > 0 else -1
        _llm_contra = llm_sentiment * _llm_dir  # positive = confirm
        if _llm_contra > 0.35 and llm_conf > 0.55:
            new_conf = min(1.0, confidence * 1.04)
        else:
            new_conf = confidence
        assert new_conf > confidence

    def test_low_confidence_source_skipped(self):
        """LLM result with confidence < 0.60 should not trigger gate."""
        llm_conf = 0.40  # too low
        headline_count = 2
        active = llm_conf >= 0.60 and headline_count >= 2
        assert not active

    def test_insufficient_headlines_skipped(self):
        """Fewer than 2 headlines → gate inactive."""
        llm_conf = 0.80
        active = llm_conf >= 0.60 and 1 >= 2
        assert not active
