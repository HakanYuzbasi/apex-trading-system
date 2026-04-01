"""
tests/test_liquidation_monitor_live.py
Tests for Fix #5: Binance REST API funding rates + OI integration
in LiquidationMonitor._fetch_funding_rate / _fetch_oi_series / _fetch_price_oi_change.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


from monitoring.liquidation_monitor import (
    LiquidationMonitor,
    funding_rate_signal,
    oi_velocity_signal,
    price_oi_divergence_signal,
)


# ── Symbol normalization ──────────────────────────────────────────────────────

class TestBinanceSymbolNormalization(unittest.TestCase):
    def _sym(self, s):
        return LiquidationMonitor._to_binance_symbol(s)

    def test_crypto_btc_usd(self):
        self.assertEqual(self._sym("CRYPTO:BTC/USD"), "BTCUSDT")

    def test_crypto_eth_usd(self):
        self.assertEqual(self._sym("CRYPTO:ETH/USD"), "ETHUSDT")

    def test_bare_btc_usdt(self):
        """Already in USDT format should not double-append."""
        self.assertEqual(self._sym("BTC/USDT"), "BTCUSDT")

    def test_crypto_sol_usd(self):
        self.assertEqual(self._sym("CRYPTO:SOL/USD"), "SOLUSDT")

    def test_crypto_xrp_usd(self):
        self.assertEqual(self._sym("CRYPTO:XRP/USD"), "XRPUSDT")

    def test_result_is_uppercase(self):
        result = self._sym("crypto:btc/usd")
        self.assertEqual(result, result.upper())


# ── Funding rate fetch (mocked HTTP) ─────────────────────────────────────────

class TestFundingRateFetch(unittest.TestCase):
    def _make_monitor(self):
        m = LiquidationMonitor()
        return m

    def test_positive_funding_rate_extracted(self):
        """A positive Binance response should return the funding rate."""
        m = self._make_monitor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"fundingRate": "0.0001", "fundingTime": 1700000000000}]

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        m._requests_session = mock_session

        rate = m._fetch_funding_rate("CRYPTO:BTC/USD")
        self.assertAlmostEqual(rate, 0.0001, places=6)

    def test_negative_funding_rate_extracted(self):
        m = self._make_monitor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"fundingRate": "-0.00025", "fundingTime": 1700000000000}]

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        m._requests_session = mock_session

        rate = m._fetch_funding_rate("CRYPTO:ETH/USD")
        self.assertAlmostEqual(rate, -0.00025, places=6)

    def test_http_error_returns_zero(self):
        m = self._make_monitor()
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        m._requests_session = mock_session

        rate = m._fetch_funding_rate("CRYPTO:BTC/USD")
        self.assertEqual(rate, 0.0)

    def test_empty_response_returns_zero(self):
        m = self._make_monitor()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        m._requests_session = mock_session

        rate = m._fetch_funding_rate("CRYPTO:BTC/USD")
        self.assertEqual(rate, 0.0)

    def test_exception_returns_zero(self):
        m = self._make_monitor()
        mock_session = MagicMock()
        mock_session.get.side_effect = ConnectionError("Network down")
        m._requests_session = mock_session

        rate = m._fetch_funding_rate("CRYPTO:BTC/USD")
        self.assertEqual(rate, 0.0)

    def test_session_reused_across_calls(self):
        """_get_session() must return the same object on repeated calls."""
        m = self._make_monitor()
        s1 = m._get_session()
        s2 = m._get_session()
        self.assertIs(s1, s2)


# ── OI series fetch (mocked HTTP) ────────────────────────────────────────────

class TestOISeriesFetch(unittest.TestCase):
    def _make_monitor_with_session(self, response_data, status=200):
        m = LiquidationMonitor()
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.json.return_value = response_data
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        m._requests_session = mock_session
        return m

    def test_oi_series_extracted_correctly(self):
        data = [
            {"sumOpenInterest": "100000.0", "timestamp": 1000},
            {"sumOpenInterest": "98000.0", "timestamp": 2000},
            {"sumOpenInterest": "95000.0", "timestamp": 3000},
        ]
        m = self._make_monitor_with_session(data)
        oi = m._fetch_oi_series("CRYPTO:BTC/USD")
        self.assertEqual(len(oi), 3)
        self.assertAlmostEqual(oi[0], 100000.0)
        self.assertAlmostEqual(oi[-1], 95000.0)

    def test_http_error_returns_empty_list(self):
        m = self._make_monitor_with_session([], status=500)
        oi = m._fetch_oi_series("CRYPTO:BTC/USD")
        self.assertEqual(oi, [])

    def test_exception_returns_empty_list(self):
        m = LiquidationMonitor()
        mock_session = MagicMock()
        mock_session.get.side_effect = TimeoutError("Request timed out")
        m._requests_session = mock_session
        oi = m._fetch_oi_series("CRYPTO:ETH/USD")
        self.assertEqual(oi, [])


# ── Price/OI change fetch ─────────────────────────────────────────────────────

class TestPriceOIChangeFetch(unittest.TestCase):
    def test_price_change_computed_from_klines(self):
        """4h price change: klines[-4][1] open to klines[-1][4] close."""
        m = LiquidationMonitor()

        # klines format: [openTime, open, high, low, close, volume, ...]
        # With 5 klines, index -4 = index 1; its open ("50000.0") is the 4h-ago price.
        klines = [
            [1000, "49900.0", "50000", "49800", "50000", "1000"],  # oldest (index 0)
            [2000, "50000.0", "50500", "49500", "50100", "1000"],  # index 1 = index -4
            [3000, "50100.0", "50600", "49600", "50200", "1000"],
            [4000, "50200.0", "50700", "49700", "50300", "1000"],
            [5000, "50300.0", "50900", "49900", "51000", "1000"],  # last close (index -1)
        ]
        oi_data = [
            {"sumOpenInterest": "100000.0"},
            {"sumOpenInterest": "98000.0"},
            {"sumOpenInterest": "96000.0"},
            {"sumOpenInterest": "94000.0"},
        ]

        def mock_get(url, params=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            if "klines" in url:
                resp.json.return_value = klines
            else:
                resp.json.return_value = oi_data
            return resp

        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get
        m._requests_session = mock_session

        price_chg, oi_chg = m._fetch_price_oi_change("CRYPTO:BTC/USD")

        # price: (51000 - 50000) / 50000 = +2%
        self.assertAlmostEqual(price_chg, 0.02, places=4)
        # OI: (94000 - 100000) / 100000 = -6%
        self.assertAlmostEqual(oi_chg, -0.06, places=4)

    def test_exception_returns_zeros(self):
        m = LiquidationMonitor()
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Timeout")
        m._requests_session = mock_session
        pc, oc = m._fetch_price_oi_change("CRYPTO:BTC/USD")
        self.assertEqual(pc, 0.0)
        self.assertEqual(oc, 0.0)


# ── End-to-end compute with mocked data ──────────────────────────────────────

class TestLiquidationMonitorEndToEnd(unittest.TestCase):
    def test_extreme_funding_generates_critical_risk(self):
        """High positive funding rate should produce critical risk level."""
        m = LiquidationMonitor()

        def mock_get(url, params=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            if "fundingRate" in url:
                resp.json.return_value = [{"fundingRate": "0.003"}]  # 0.3% per 8h — extreme
            elif "openInterestHist" in url:
                resp.json.return_value = [{"sumOpenInterest": str(i)} for i in range(100000, 90000, -1000)]
            elif "klines" in url:
                resp.json.return_value = [
                    [i * 1000, "100.0", "100.5", "99.5", str(95 + i), "1000"]
                    for i in range(5)
                ]
            return resp

        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get
        m._requests_session = mock_session

        state = m.get_state("CRYPTO:BTC/USD")
        self.assertLess(state.composite_signal, 0)
        self.assertLess(state.sizing_multiplier, 1.0)

    def test_zero_funding_normal_oi_gives_ok_state(self):
        """Neutral/low funding + stable OI → normal risk level."""
        m = LiquidationMonitor()

        def mock_get(url, params=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            if "fundingRate" in url:
                # 0.0 funding rate — no cascade signal
                resp.json.return_value = [{"fundingRate": "0.0"}]
            elif "openInterestHist" in url:
                # Stable OI — no change
                resp.json.return_value = [{"sumOpenInterest": "100000.0"}] * 6
            elif "klines" in url:
                resp.json.return_value = [
                    [i * 1000, "100.0", "100.5", "99.5", "100.2", "1000"]
                    for i in range(5)
                ]
            return resp

        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get
        m._requests_session = mock_session

        state = m.get_state("CRYPTO:BTC/USD")
        self.assertEqual(state.risk_level, "normal")
        # With 0 funding + 0 OI change + 0 price change → composite=0 → mult=1.0
        self.assertAlmostEqual(state.sizing_multiplier, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
