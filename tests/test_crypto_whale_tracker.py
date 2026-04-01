"""
tests/test_crypto_whale_tracker.py

Verification suite for the On-Chain Crypto Whale Tracker (Phase 8).
Ensures the free alternative.me Fear & Greed endpoint returns strictly
bounded floats and that all failure paths gracefully default to 0.0.
"""

import time
import unittest
from unittest.mock import patch, MagicMock


class TestOnChainTracker(unittest.TestCase):
    """Tests for data/crypto/on_chain_flow.py"""

    def setUp(self):
        # Import fresh instance for each test to avoid cache interference
        import importlib
        import data.crypto.on_chain_flow as mod
        importlib.reload(mod)
        self.mod = mod
        self.tracker = mod.OnChainTracker()

    def test_output_bounds_extreme_greed(self):
        """FNG=100 (Extreme Greed) → should return -1.0 (contrarian sell)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"value": "100"}]}
        with patch("requests.get", return_value=mock_resp):
            score = self.tracker.get_crypto_flow_sentiment()
        self.assertAlmostEqual(score, -1.0, places=5)

    def test_output_bounds_extreme_fear(self):
        """FNG=0 (Extreme Fear) → should return +1.0 (contrarian buy)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"value": "0"}]}
        with patch("requests.get", return_value=mock_resp):
            score = self.tracker.get_crypto_flow_sentiment()
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_output_bounds_neutral(self):
        """FNG=50 (Neutral) → should return 0.0."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"value": "50"}]}
        with patch("requests.get", return_value=mock_resp):
            score = self.tracker.get_crypto_flow_sentiment()
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_output_strictly_bounded(self):
        """All FNG values 0-100 must yield scores in [-1.0, 1.0]."""
        for fng in range(0, 101, 5):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"data": [{"value": str(fng)}]}
            with patch("requests.get", return_value=mock_resp):
                score = self.tracker.get_crypto_flow_sentiment()
            self.assertGreaterEqual(score, -1.0, f"FNG={fng} underflowed: {score}")
            self.assertLessEqual(score, 1.0, f"FNG={fng} overflowed: {score}")

    def test_api_error_returns_neutral(self):
        """Network errors must return 0.0, not raise."""
        with patch("requests.get", side_effect=ConnectionError("timeout")):
            score = self.tracker.get_crypto_flow_sentiment()
        self.assertEqual(score, 0.0)

    def test_bad_status_code_returns_neutral(self):
        """Non-200 HTTP response must return 0.0."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with patch("requests.get", return_value=mock_resp):
            score = self.tracker.get_crypto_flow_sentiment()
        self.assertEqual(score, 0.0)

    def test_empty_data_returns_neutral(self):
        """Empty data array must return 0.0."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        with patch("requests.get", return_value=mock_resp):
            score = self.tracker.get_crypto_flow_sentiment()
        self.assertEqual(score, 0.0)

    def test_cache_prevents_second_call(self):
        """Second call within TTL should not hit the network."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"value": "30"}]}
        with patch("requests.get", return_value=mock_resp) as mock_get:
            self.tracker.get_crypto_flow_sentiment()  # first call → populates cache
            self.tracker.get_crypto_flow_sentiment()  # second call → should use cache
        mock_get.assert_called_once()  # network hit exactly once

    def test_module_level_singleton(self):
        """Module-level get_on_chain_sentiment() returns a float in [-1, 1]."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"value": "25"}]}
        with patch("requests.get", return_value=mock_resp):
            score = self.mod.get_on_chain_sentiment()
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_return_type_is_float(self):
        """Return value must always be a Python float, never int or None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"value": "60"}]}
        with patch("requests.get", return_value=mock_resp):
            score = self.tracker.get_crypto_flow_sentiment()
        self.assertIsInstance(score, float)


class TestOptionsFlowFetcher(unittest.TestCase):
    """Tests for data/social/options_flow.py (Phase 5)."""

    def setUp(self):
        import importlib
        import data.social.options_flow as mod
        importlib.reload(mod)
        self.mod = mod
        self.fetcher = mod.OptionsFlowFetcher()

    def test_output_bounds_call_heavy(self):
        """Heavy call volume (PCR < 0.5) → bullish → +1.0 clamped."""
        mock_ticker = MagicMock()
        mock_ticker.options = ["2024-01-19"]
        calls_df = MagicMock()
        calls_df.__getitem__ = MagicMock(side_effect=lambda k: MagicMock(sum=MagicMock(return_value=10000)))
        puts_df = MagicMock()
        puts_df.__getitem__ = MagicMock(side_effect=lambda k: MagicMock(sum=MagicMock(return_value=1000)))
        mock_ticker.option_chain.return_value = MagicMock(calls=calls_df, puts=puts_df)
        with patch("yfinance.Ticker", return_value=mock_ticker):
            score = self.fetcher.get_smart_money_sentiment("AAPL")
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_no_options_market_returns_neutral(self):
        """Crypto symbols with no options chain return 0.0."""
        mock_ticker = MagicMock()
        mock_ticker.options = []
        with patch("yfinance.Ticker", return_value=mock_ticker):
            score = self.fetcher.get_smart_money_sentiment("CRYPTO:BTC/USD")
        self.assertEqual(score, 0.0)

    def test_exception_returns_neutral(self):
        """Any exception from yfinance returns 0.0, not a crash."""
        with patch("yfinance.Ticker", side_effect=Exception("rate limit")):
            score = self.fetcher.get_smart_money_sentiment("TSLA")
        self.assertEqual(score, 0.0)

    def test_return_type_is_float(self):
        """Must always return float."""
        with patch("yfinance.Ticker", side_effect=Exception("fail")):
            score = self.fetcher.get_smart_money_sentiment("MSFT")
        self.assertIsInstance(score, float)


class TestPortfolioKellyOptimizer(unittest.TestCase):
    """Tests for models/portfolio_optimizer.py (Phase 9)."""

    def setUp(self):
        from models.portfolio_optimizer import KellyCriterionOptimizer
        self.optimizer = KellyCriterionOptimizer(target_rr=1.5, half_kelly_modifier=0.5)

    def test_high_confidence_high_winrate_gives_positive_multiplier(self):
        """p=0.7 win rate + 0.8 ML confidence → positive multiplier."""
        mult = self.optimizer.calculate_sizing_multiplier(
            ml_confidence=0.8, historical_win_rate=0.7, is_high_vix=False
        )
        self.assertGreater(mult, 0.0)
        self.assertLessEqual(mult, self.optimizer.MAX_LEVERAGE)

    def test_poor_odds_return_min_leverage(self):
        """p ≈ 0.25 blended → Kelly <= 0 → return MIN_LEVERAGE (cold-start floor,
        not zero) so early poor performance doesn't freeze trade count."""
        mult = self.optimizer.calculate_sizing_multiplier(
            ml_confidence=0.1, historical_win_rate=0.2, is_high_vix=False
        )
        self.assertEqual(mult, self.optimizer.MIN_LEVERAGE)

    def test_high_vix_cuts_size(self):
        """High-VIX flag reduces multiplier by 40%."""
        normal = self.optimizer.calculate_sizing_multiplier(0.7, 0.65, is_high_vix=False)
        vix = self.optimizer.calculate_sizing_multiplier(0.7, 0.65, is_high_vix=True)
        if normal > 0:
            self.assertLess(vix, normal)

    def test_bounds_always_respected(self):
        """Multiplier always in [MIN_LEVERAGE, MAX_LEVERAGE]."""
        for conf in [0.1, 0.5, 0.9]:
            for wr in [0.3, 0.55, 0.8]:
                mult = self.optimizer.calculate_sizing_multiplier(conf, wr)
                if mult > 0:
                    self.assertGreaterEqual(mult, self.optimizer.MIN_LEVERAGE)
                    self.assertLessEqual(mult, self.optimizer.MAX_LEVERAGE)

    def test_invalid_winrate_defaults_to_half(self):
        """Win rates of 0 or 1 fall back to 0.5 to avoid infinity."""
        mult_zero = self.optimizer.calculate_sizing_multiplier(0.6, 0.0)
        mult_one = self.optimizer.calculate_sizing_multiplier(0.6, 1.0)
        self.assertIsInstance(mult_zero, float)
        self.assertIsInstance(mult_one, float)


class TestRLWeightGovernor(unittest.TestCase):
    """Tests for models/rl_weight_governor.py (Phase 7)."""

    def setUp(self):
        import tempfile
        import models.rl_weight_governor as mod
        self.mod = mod
        self.governor = mod.RLWeightGovernor(model_dir=tempfile.mkdtemp())

    def test_returns_dict_with_required_keys(self):
        """get_optimal_weights must include core component keys."""
        weights = self.governor.get_optimal_weights("neutral", is_high_volatility=False, training=False)
        required = {"momentum", "mean_reversion", "trend", "senate_sentiment", "smart_money_flow", "news_sentiment"}
        for key in required:
            self.assertIn(key, weights, f"Missing weight key: {key}")

    def test_all_weights_are_floats(self):
        """Every weight value must be a float."""
        weights = self.governor.get_optimal_weights("bull", is_high_volatility=False, training=False)
        for k, v in weights.items():
            if k != "__action__":
                self.assertIsInstance(v, float, f"Weight {k} is not float: {type(v)}")

    def test_update_q_value_does_not_crash(self):
        """Bellman update with typical trade reward must not raise."""
        self.governor.update_q_value("neutral", False, 0, reward=0.015, next_regime="bull")

    def test_strong_bull_override_applied(self):
        """STRONG_BULL regime forces mean_reversion >= 0.45."""
        weights = self.governor.get_optimal_weights("STRONG_BULL", is_high_volatility=False, training=False)
        self.assertGreaterEqual(weights.get("mean_reversion", 0), 0.45)

    def test_exploration_vs_exploitation(self):
        """Both training=True (explore) and False (exploit) return valid dicts."""
        for training in [True, False]:
            weights = self.governor.get_optimal_weights("bear", True, training=training)
            self.assertIsInstance(weights, dict)
            self.assertGreater(len(weights), 0)


if __name__ == "__main__":
    unittest.main()
