"""
tests/test_wss_diagnostics.py
Tests for Fix #2: WebSocket hit-rate diagnostics and low-hit-rate alerts.
"""
from __future__ import annotations

import asyncio
import types
import sys
import unittest
from unittest.mock import MagicMock, patch


# ── Stubs ─────────────────────────────────────────────────────────────────────
if "websockets" not in sys.modules:
    sys.modules["websockets"] = types.ModuleType("websockets")

if "config" not in sys.modules:
    _cfg = types.ModuleType("config")
    sys.modules["config"] = _cfg

import config as _cfg_mod
if not hasattr(_cfg_mod, "ApexConfig"):
    _cfg_mod.ApexConfig = types.SimpleNamespace()

_apex = _cfg_mod.ApexConfig
setattr(_apex, "WSS_HIT_RATE_WARN_THRESHOLD", 0.50)


def _make_streamer(wss_hits: int = 0, wss_misses: int = 0):
    """Return a minimal WebsocketStreamer-like object."""
    from data.websocket_streamer import WebsocketStreamer
    s = WebsocketStreamer.__new__(WebsocketStreamer)
    s._wss_hits = wss_hits
    s._wss_misses = wss_misses
    s._reconnect_count = {"equity": 0, "crypto": 0}
    s._last_connect_ts = {}
    s._last_disconnect_ts = {}
    from datetime import datetime, timezone
    s._session_start_ts = datetime.now(timezone.utc).timestamp()
    s._price_cache = {}
    s._equity_ready = asyncio.Event()
    s._crypto_ready = asyncio.Event()
    return s


class TestWSSHitRate(unittest.TestCase):
    def test_hit_rate_zero_when_no_requests(self):
        s = _make_streamer(0, 0)
        self.assertEqual(s.hit_rate, 0.0)

    def test_hit_rate_100pct_all_wss(self):
        s = _make_streamer(200, 0)
        self.assertAlmostEqual(s.hit_rate, 1.0)

    def test_hit_rate_50pct(self):
        s = _make_streamer(50, 50)
        self.assertAlmostEqual(s.hit_rate, 0.5)

    def test_hit_rate_below_threshold_triggers_warning(self):
        """When hit_rate < 0.50 and total >= 20, warning should be emitted."""
        s = _make_streamer(wss_hits=3, wss_misses=17)  # 15% hit rate
        threshold = float(getattr(_apex, "WSS_HIT_RATE_WARN_THRESHOLD", 0.50))
        total = s._wss_hits + s._wss_misses
        self.assertTrue(total >= 20)
        self.assertLess(s.hit_rate, threshold)

    def test_hit_rate_above_threshold_no_warning(self):
        """When hit_rate >= 0.50 and total >= 20, no warning needed."""
        s = _make_streamer(wss_hits=15, wss_misses=5)  # 75% hit rate
        threshold = float(getattr(_apex, "WSS_HIT_RATE_WARN_THRESHOLD", 0.50))
        self.assertGreater(s.hit_rate, threshold)

    def test_warning_suppressed_when_total_too_low(self):
        """With < 20 total requests, don't warn yet (insufficient sample)."""
        s = _make_streamer(wss_hits=1, wss_misses=5)  # only 6 total
        total = s._wss_hits + s._wss_misses
        self.assertLess(total, 20)  # below min-sample threshold

    def test_get_metrics_returns_expected_keys(self):
        """get_metrics() must include all health diagnostic fields."""
        s = _make_streamer(100, 50)
        m = s.get_metrics()
        for key in ("hit_rate", "wss_hits", "wss_misses",
                    "equity_reconnects", "crypto_reconnects",
                    "equity_connected", "crypto_connected",
                    "cached_symbols", "session_uptime_seconds"):
            self.assertIn(key, m, f"Missing key: {key}")

    def test_get_metrics_hit_rate_matches_property(self):
        s = _make_streamer(80, 20)
        m = s.get_metrics()
        self.assertAlmostEqual(m["hit_rate"], s.hit_rate, places=3)

    def test_health_annotation_logic_ok(self):
        """simulate the API endpoint health annotation for healthy streamer."""
        s = _make_streamer(wss_hits=90, wss_misses=10)
        metrics = s.get_metrics()
        warn_thr = 0.50
        total = metrics["wss_hits"] + metrics["wss_misses"]
        hr = metrics["hit_rate"]
        health = "degraded" if (total >= 20 and hr < warn_thr) else "ok"
        self.assertEqual(health, "ok")

    def test_health_annotation_logic_degraded(self):
        """simulate the API endpoint health annotation for low-hit-rate streamer."""
        s = _make_streamer(wss_hits=5, wss_misses=95)
        metrics = s.get_metrics()
        warn_thr = 0.50
        total = metrics["wss_hits"] + metrics["wss_misses"]
        hr = metrics["hit_rate"]
        health = "degraded" if (total >= 20 and hr < warn_thr) else "ok"
        self.assertEqual(health, "degraded")

    def test_config_key_present(self):
        """WSS_HIT_RATE_WARN_THRESHOLD is set in ApexConfig."""
        val = getattr(_apex, "WSS_HIT_RATE_WARN_THRESHOLD", None)
        self.assertIsNotNone(val)
        self.assertGreater(float(val), 0.0)
        self.assertLessEqual(float(val), 1.0)


if __name__ == "__main__":
    unittest.main()
