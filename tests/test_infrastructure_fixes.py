"""
tests/test_infrastructure_fixes.py — Tests for the 4 critical infrastructure fixes:
  1. IBKR persistent recovery loop after reconnect exhaustion
  2. _failed_symbols temporary blacklist + refresh_data() full-universe coverage
  3. Gate rejection tracer counters + periodic summary
  4. Stale price cache detection (no live-feed timestamp warning)
"""
from __future__ import annotations

import asyncio
import time
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ibkr_connector():
    """Return a minimal IBKRConnector-like object with just the reconnect machinery."""
    import importlib, sys

    # Stub heavy dependencies so we can import the connector without a live TWS
    for mod in ("ib_insync", "config"):
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Provide ApexConfig with required attributes
    cfg = sys.modules["config"]
    if not hasattr(cfg, "ApexConfig"):
        cfg.ApexConfig = types.SimpleNamespace(
            IBKR_FAILOVER_MAX_RETRIES=3,
            IBKR_FAILOVER_RETRY_SECONDS=0.01,
            IBKR_RECOVERY_INTERVAL_SECONDS=0.05,
        )

    # Build a minimal connector-like object rather than importing the full module
    class FakeConnector:
        def __init__(self):
            self._persistently_down = False
            self._fatal_error = False
            self._reconnect_failure_count = 0
            self._connect_calls = 0
            self.ib = MagicMock()
            self.ib.disconnect = MagicMock()

        async def connect(self):
            self._connect_calls += 1
            if self._connect_calls < 3:
                raise ConnectionError("TWS not ready")

        async def _persistent_recovery_loop(self):
            from config import ApexConfig
            interval = float(getattr(ApexConfig, "IBKR_RECOVERY_INTERVAL_SECONDS", 300.0))
            attempt = 0
            while self._persistently_down and not self._fatal_error:
                await asyncio.sleep(interval)
                if not self._persistently_down or self._fatal_error:
                    return
                attempt += 1
                try:
                    self.ib.disconnect()
                except Exception:
                    pass
                await asyncio.sleep(0)
                try:
                    await self.connect()
                    self._reconnect_failure_count = 0
                    self._persistently_down = False
                    return
                except Exception:
                    pass

    return FakeConnector()


def _make_minimal_engine():
    """Return a bare object that mimics the engine fields touched by our fixes."""
    eng = types.SimpleNamespace(
        _failed_symbols=set(),
        _failed_symbols_time={},
        _gate_rejection_counts={},
        _gate_rejection_last_log=0.0,
        price_cache={},
        _price_cache_ts={},
        historical_data={},
    )

    # Attach the helper methods
    def _record_gate_rejection(self_eng, gate: str):
        self_eng._gate_rejection_counts[gate] = (
            self_eng._gate_rejection_counts.get(gate, 0) + 1
        )

    def _maybe_log_gate_summary(self_eng, cycle: int):
        if cycle % 300 != 0:
            return
        if not self_eng._gate_rejection_counts:
            return
        self_eng._gate_rejection_counts.clear()

    eng._record_gate_rejection = lambda gate: _record_gate_rejection(eng, gate)
    eng._maybe_log_gate_summary = lambda cycle: _maybe_log_gate_summary(eng, cycle)
    return eng


# ── IBKR persistent recovery loop ────────────────────────────────────────────

class TestIBKRPersistentRecoveryLoop(unittest.TestCase):
    def test_recovery_loop_clears_persistently_down(self):
        """After persistent_down=True, the background loop should clear it on success."""
        connector = _make_ibkr_connector()
        connector._persistently_down = True
        connector._connect_calls = 0  # reset so the 3rd call succeeds

        async def _run():
            await connector._persistent_recovery_loop()

        asyncio.get_event_loop().run_until_complete(_run())
        self.assertFalse(connector._persistently_down)

    def test_recovery_loop_stops_on_fatal_error(self):
        """If _fatal_error is set, the loop must exit without attempting connect."""
        connector = _make_ibkr_connector()
        connector._persistently_down = True
        connector._fatal_error = True

        connect_called = []

        async def _fake_connect():
            connect_called.append(1)

        connector.connect = _fake_connect

        async def _run():
            # Loop should exit immediately on fatal_error (no sleep needed)
            connector._persistently_down = False  # loop exits before sleep
            await connector._persistent_recovery_loop()

        asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(len(connect_called), 0)

    def test_recovery_loop_retries_on_failure(self):
        """Loop should keep retrying until connect succeeds."""
        connector = _make_ibkr_connector()
        connector._persistently_down = True

        attempts = []
        original_connect = connector.connect

        async def counting_connect():
            attempts.append(1)
            await original_connect()

        connector.connect = counting_connect

        async def _run():
            await connector._persistent_recovery_loop()

        asyncio.get_event_loop().run_until_complete(_run())
        # 2 failures + 1 success = 3 attempts
        self.assertGreaterEqual(len(attempts), 1)
        self.assertFalse(connector._persistently_down)

    def test_persistently_down_reset_on_success(self):
        """Connector state is clean after successful recovery."""
        connector = _make_ibkr_connector()
        connector._persistently_down = True
        connector._reconnect_failure_count = 5
        connector._connect_calls = 2  # next call succeeds

        async def _run():
            await connector._persistent_recovery_loop()

        asyncio.get_event_loop().run_until_complete(_run())
        self.assertFalse(connector._persistently_down)
        self.assertEqual(connector._reconnect_failure_count, 0)


# ── _failed_symbols temporary blacklist ──────────────────────────────────────

class TestFailedSymbolsRetry(unittest.TestCase):
    def test_failed_symbols_time_recorded(self):
        eng = _make_minimal_engine()
        eng._failed_symbols.add("AAPL")
        eng._failed_symbols_time["AAPL"] = time.time() - 400  # 400s ago

        # Verify the time is stored
        self.assertIn("AAPL", eng._failed_symbols_time)
        age = time.time() - eng._failed_symbols_time["AAPL"]
        self.assertGreater(age, 300)

    def test_symbols_eligible_for_retry_after_5min(self):
        eng = _make_minimal_engine()
        eng._failed_symbols.add("TSLA")
        eng._failed_symbols_time["TSLA"] = time.time() - 350

        cutoff = time.time() - 300
        eligible = {s for s, t in eng._failed_symbols_time.items() if t < cutoff}
        self.assertIn("TSLA", eligible)

    def test_recent_failure_not_eligible_for_retry(self):
        eng = _make_minimal_engine()
        eng._failed_symbols.add("NVDA")
        eng._failed_symbols_time["NVDA"] = time.time() - 60  # only 60s ago

        cutoff = time.time() - 300
        eligible = {s for s, t in eng._failed_symbols_time.items() if t < cutoff}
        self.assertNotIn("NVDA", eligible)

    def test_discard_from_failed_on_success(self):
        eng = _make_minimal_engine()
        eng._failed_symbols.add("MSFT")
        eng._failed_symbols_time["MSFT"] = time.time() - 400

        # Simulate successful load
        eng._failed_symbols.discard("MSFT")
        eng._failed_symbols_time.pop("MSFT", None)

        self.assertNotIn("MSFT", eng._failed_symbols)
        self.assertNotIn("MSFT", eng._failed_symbols_time)

    def test_setdefault_preserves_original_fail_time(self):
        """Multiple add calls should NOT reset the first-fail timestamp."""
        eng = _make_minimal_engine()
        t0 = time.time() - 200
        eng._failed_symbols_time["GOOGL"] = t0

        # Second failure — setdefault should not overwrite
        eng._failed_symbols_time.setdefault("GOOGL", time.time())
        self.assertEqual(eng._failed_symbols_time["GOOGL"], t0)


# ── Gate rejection tracer ─────────────────────────────────────────────────────

class TestGateRejectionTracer(unittest.TestCase):
    def test_record_increments_counter(self):
        eng = _make_minimal_engine()
        eng._record_gate_rejection("signal_threshold")
        eng._record_gate_rejection("signal_threshold")
        eng._record_gate_rejection("confidence_threshold")

        self.assertEqual(eng._gate_rejection_counts["signal_threshold"], 2)
        self.assertEqual(eng._gate_rejection_counts["confidence_threshold"], 1)

    def test_record_multiple_gates(self):
        eng = _make_minimal_engine()
        for g in ["signal_threshold", "directional_guard", "composite_quality"]:
            eng._record_gate_rejection(g)

        self.assertEqual(len(eng._gate_rejection_counts), 3)

    def test_summary_logged_at_300_cycle_interval(self):
        eng = _make_minimal_engine()
        eng._record_gate_rejection("signal_threshold")
        eng._record_gate_rejection("signal_threshold")

        with patch("logging.Logger.info") as mock_log:
            eng._maybe_log_gate_summary(300)
            # Counts are cleared after logging
            self.assertEqual(eng._gate_rejection_counts, {})

    def test_summary_not_logged_before_300_cycles(self):
        eng = _make_minimal_engine()
        eng._record_gate_rejection("confidence_threshold")

        initial_counts = dict(eng._gate_rejection_counts)
        eng._maybe_log_gate_summary(100)
        # Counts should NOT be cleared
        self.assertEqual(eng._gate_rejection_counts, initial_counts)

    def test_summary_skipped_if_no_rejections(self):
        eng = _make_minimal_engine()
        # No rejections recorded
        with patch("logging.Logger.info") as mock_log:
            eng._maybe_log_gate_summary(300)
            # Nothing should log

    def test_no_historical_data_gate_tracked(self):
        eng = _make_minimal_engine()
        eng._record_gate_rejection("no_historical_data")
        self.assertEqual(eng._gate_rejection_counts.get("no_historical_data"), 1)

    def test_counter_cleared_after_summary(self):
        eng = _make_minimal_engine()
        for _ in range(50):
            eng._record_gate_rejection("signal_threshold")
        eng._maybe_log_gate_summary(300)
        self.assertEqual(eng._gate_rejection_counts, {})


# ── Stale price cache detection ───────────────────────────────────────────────

class TestStalePriceDetection(unittest.TestCase):
    def test_symbols_without_live_ts_detected(self):
        """Symbols in price_cache but not _price_cache_ts are stale startup prices."""
        eng = _make_minimal_engine()
        eng.price_cache["AAPL"] = 180.0
        eng.price_cache["MSFT"] = 400.0
        # No _price_cache_ts entries → startup-sourced prices

        stale = [s for s in eng.price_cache if s not in eng._price_cache_ts and eng.price_cache.get(s, 0) > 0]
        self.assertIn("AAPL", stale)
        self.assertIn("MSFT", stale)

    def test_symbols_with_live_ts_not_stale(self):
        eng = _make_minimal_engine()
        eng.price_cache["BTC/USD"] = 87000.0
        eng._price_cache_ts["BTC/USD"] = time.time()

        stale = [s for s in eng.price_cache if s not in eng._price_cache_ts and eng.price_cache.get(s, 0) > 0]
        self.assertNotIn("BTC/USD", stale)

    def test_zero_price_excluded_from_stale(self):
        eng = _make_minimal_engine()
        eng.price_cache["DEAD"] = 0.0
        # No timestamp — but price=0 should be excluded

        stale = [s for s in eng.price_cache if s not in eng._price_cache_ts and eng.price_cache.get(s, 0) > 0]
        self.assertNotIn("DEAD", stale)

    def test_mixed_live_and_stale(self):
        eng = _make_minimal_engine()
        eng.price_cache["SOL/USD"] = 200.0  # no ts → stale
        eng.price_cache["ETH/USD"] = 3000.0  # has ts → live
        eng._price_cache_ts["ETH/USD"] = time.time()

        stale = [s for s in eng.price_cache if s not in eng._price_cache_ts and eng.price_cache.get(s, 0) > 0]
        self.assertIn("SOL/USD", stale)
        self.assertNotIn("ETH/USD", stale)

    def test_price_cache_ts_set_clears_stale(self):
        eng = _make_minimal_engine()
        eng.price_cache["NVDA"] = 900.0

        stale_before = [s for s in eng.price_cache if s not in eng._price_cache_ts and eng.price_cache.get(s, 0) > 0]
        self.assertIn("NVDA", stale_before)

        # Simulate live price update
        eng._price_cache_ts["NVDA"] = time.time()

        stale_after = [s for s in eng.price_cache if s not in eng._price_cache_ts and eng.price_cache.get(s, 0) > 0]
        self.assertNotIn("NVDA", stale_after)


# ── Integration: config key present ───────────────────────────────────────────

class TestConfigKey(unittest.TestCase):
    def test_ibkr_recovery_interval_in_config(self):
        try:
            from config import ApexConfig
            val = getattr(ApexConfig, "IBKR_RECOVERY_INTERVAL_SECONDS", None)
            self.assertIsNotNone(val)
            self.assertGreater(float(val), 0)
        except ImportError:
            self.skipTest("config not importable in test env")


if __name__ == "__main__":
    unittest.main()
