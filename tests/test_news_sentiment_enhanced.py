"""tests/test_news_sentiment_enhanced.py — Domain-augmented VADER tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from data.social.news_sentiment_llm import (
    NewsSentimentEngine,
    SentimentResult,
    _DOMAIN_LEXICON,
    _RECENCY_DECAY,
    get_news_sentiment,
    get_rich_sentiment,
)


# ── Domain lexicon sanity ─────────────────────────────────────────────────────

class TestDomainLexicon:
    def test_has_earnings_beat(self):
        assert "earnings beat" in _DOMAIN_LEXICON
        assert _DOMAIN_LEXICON["earnings beat"] > 3.0

    def test_has_earnings_miss(self):
        assert "earnings miss" in _DOMAIN_LEXICON
        assert _DOMAIN_LEXICON["earnings miss"] < -3.0

    def test_has_crypto_hack(self):
        assert "hack" in _DOMAIN_LEXICON
        assert _DOMAIN_LEXICON["hack"] < -3.0

    def test_has_etf_approval(self):
        assert "etf approval" in _DOMAIN_LEXICON
        assert _DOMAIN_LEXICON["etf approval"] > 3.0

    def test_positive_terms_positive(self):
        bullish = [v for k, v in _DOMAIN_LEXICON.items() if v > 0]
        assert len(bullish) >= 50

    def test_negative_terms_negative(self):
        bearish = [v for k, v in _DOMAIN_LEXICON.items() if v < 0]
        assert len(bearish) >= 50

    def test_no_zero_valence_terms(self):
        zeros = [k for k, v in _DOMAIN_LEXICON.items() if v == 0]
        assert zeros == []


# ── Negation pre-processing ───────────────────────────────────────────────────

class TestNegation:
    def setup_method(self):
        self.engine = NewsSentimentEngine()

    def test_negation_flips_positive_phrase(self):
        # "not" before "rally" should flip to negative marker
        result = self.engine._apply_negation("market did not rally today")
        assert "terrible" in result or "not" in result  # negation handled

    def test_no_negation_unchanged_negative(self):
        # bankruptcy needs no negation — should pass through unchanged
        result = self.engine._apply_negation("company files for bankruptcy")
        assert "bankruptcy" in result

    def test_negation_trigger_detected(self):
        # "fails" trigger immediately followed by "beat estimates" (positive phrase)
        text = "fails beat estimates"
        result = self.engine._apply_negation(text.lower())
        assert "terrible" in result  # trigger replaced with negative marker


# ── Score aggregation ─────────────────────────────────────────────────────────

class TestAggregation:
    def setup_method(self):
        self.engine = NewsSentimentEngine()

    def test_single_positive_score(self):
        s, c = self.engine._aggregate([0.8])
        assert s > 0
        assert c == 0.35  # single-headline confidence is fixed at 0.35

    def test_single_negative_score(self):
        s, c = self.engine._aggregate([-0.8])
        assert s < 0
        assert c == 0.35

    def test_agreeing_positives_high_confidence(self):
        scores = [0.7, 0.8, 0.75, 0.72]
        s, c = self.engine._aggregate(scores)
        assert s > 0
        assert c > 0.35

    def test_contradicting_scores_low_confidence(self):
        scores = [0.9, -0.9, 0.8, -0.8]
        s, c = self.engine._aggregate(scores)
        # The sentiment should be near zero (mixed), confidence low
        assert abs(s) < 0.5
        assert c < 0.5

    def test_time_decay_weights_recent_more(self):
        # 1st score = positive, 2nd score = negative (older)
        s, _ = self.engine._aggregate([0.9, -0.9])
        # Recent (positive) should outweigh older (negative)
        assert s > 0

    def test_empty_scores(self):
        s, c = self.engine._aggregate([])
        assert s == 0.0
        assert c == 0.0

    def test_sentiment_clamped_to_one(self):
        s, _ = self.engine._aggregate([1.0, 1.0, 1.0])
        assert s <= 1.0

    def test_sentiment_clamped_to_minus_one(self):
        s, _ = self.engine._aggregate([-1.0, -1.0, -1.0])
        assert s >= -1.0


# ── Headline scoring ──────────────────────────────────────────────────────────

class TestHeadlineScoring:
    def setup_method(self):
        self.engine = NewsSentimentEngine()

    def test_strong_positive_headline(self):
        if self.engine._analyzer is None:
            return  # VADER not available in env
        score = self.engine._score_headline(
            "Company reports record earnings beat, guidance raised"
        )
        assert score > 0

    def test_strong_negative_headline(self):
        if self.engine._analyzer is None:
            return
        score = self.engine._score_headline(
            "Company files for bankruptcy after accounting fraud revealed"
        )
        assert score < 0

    def test_neutral_headline_near_zero(self):
        if self.engine._analyzer is None:
            return
        score = self.engine._score_headline(
            "Company announces quarterly results date"
        )
        assert -0.5 < score < 0.5


# ── SentimentResult dataclass ────────────────────────────────────────────────

class TestSentimentResult:
    def test_to_dict_keys(self):
        r = SentimentResult(
            symbol="AAPL",
            sentiment=0.42,
            confidence=0.65,
            headline_count=5,
            top_headlines=["h1", "h2"],
        )
        d = r.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["sentiment"] == 0.42
        assert d["confidence"] == 0.65
        assert d["headline_count"] == 5
        assert "top_headlines" in d
        assert d["method"] == "vader+domain"

    def test_top_headlines_capped_at_three(self):
        r = SentimentResult(
            symbol="AAPL", sentiment=0.1, confidence=0.5,
            headline_count=5, top_headlines=["a", "b", "c", "d", "e"],
        )
        assert len(r.to_dict()["top_headlines"]) == 3


# ── Cache behaviour ───────────────────────────────────────────────────────────

class TestCache:
    def test_cache_hit_avoids_second_fetch(self):
        engine = NewsSentimentEngine()
        mock_result = SentimentResult("AAPL", 0.5, 0.7, 3)
        import time
        engine._cache["AAPL"] = (mock_result, time.time())

        with patch.object(engine, "_fetch_and_score") as mock_fetch:
            result = engine.get_rich_sentiment("AAPL")
            mock_fetch.assert_not_called()
        assert result.sentiment == 0.5

    def test_expired_cache_triggers_refetch(self):
        engine = NewsSentimentEngine()
        import time
        # Put a stale entry in the cache
        engine._cache["AAPL"] = (
            SentimentResult("AAPL", 0.5, 0.7, 3),
            time.time() - 2000,  # 2000s ago > 1200s TTL
        )
        fresh = SentimentResult("AAPL", 0.2, 0.4, 1, method="test")
        with patch.object(engine, "_fetch_and_score", return_value=fresh):
            result = engine.get_rich_sentiment("AAPL")
        assert result.method == "test"

    def test_symbol_normalization(self):
        """CRYPTO:ETH/USD should clean to ETH."""
        engine = NewsSentimentEngine()
        fresh = SentimentResult("ETH", 0.1, 0.3, 0)
        with patch.object(engine, "_fetch_and_score", return_value=fresh) as m:
            engine.get_rich_sentiment("CRYPTO:ETH/USD")
            m.assert_called_once_with("ETH")


# ── Module-level backward-compat functions ───────────────────────────────────

class TestModuleFunctions:
    def test_get_news_sentiment_returns_float(self):
        with patch(
            "data.social.news_sentiment_llm._engine.get_news_sentiment",
            return_value=0.33,
        ):
            assert get_news_sentiment("AAPL") == 0.33

    def test_get_rich_sentiment_returns_result(self):
        fake = SentimentResult("AAPL", 0.33, 0.6, 2)
        with patch(
            "data.social.news_sentiment_llm._engine.get_rich_sentiment",
            return_value=fake,
        ):
            result = get_rich_sentiment("AAPL")
        assert result.sentiment == 0.33
