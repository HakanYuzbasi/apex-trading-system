"""
tests/test_regime_router.py
===========================
Full unit-test suite for risk/regime_router.py.

Coverage targets
----------------
- RegimeRouter initialisation (valid/invalid params)
- record_btc_price: thread-safety and buffer capping
- evaluate: caching contract (second call within cache_seconds returns same object)
- _compute: each of the 3 regime branches (RANGING, TRENDING, HIGH_VOL_PANIC)
  driven by:
    * VIX regime label  (mocked VIXRegimeManager)
    * crypto vol-ratio  (injected via record_btc_price)
    * panic_vol_threshold override
- notional_multiplier: stays in [0.10, 1.50] under extreme VIX mult
- block_new_entries: True only for HIGH_VOL_PANIC
- get_global_regime_router singleton: same instance on repeated calls,
  reset between tests via module-level teardown
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_vix_state(regime_value: str, risk_mult: float = 1.0) -> SimpleNamespace:
    """Build a minimal VIXState-like object that RegimeRouter._compute expects."""
    return SimpleNamespace(
        regime=SimpleNamespace(value=regime_value),
        risk_multiplier=risk_mult,
    )


def _router_with_mocked_vix(
    vix_regime: str = "normal",
    vix_risk_mult: float = 1.0,
    *,
    crypto_beta_scalar: float = 0.55,
    cache_seconds: float = 0.0,       # disable cache so every evaluate() recomputes
    btc_lookback: int = 10,
    panic_vol_threshold: float = 3.5,
) -> "RegimeRouter":
    """
    Create a RegimeRouter where the VIXRegimeManager is replaced by a mock
    that returns the given regime / risk_multiplier.
    """
    from risk.regime_router import RegimeRouter

    router = RegimeRouter(
        crypto_beta_scalar=crypto_beta_scalar,
        cache_seconds=cache_seconds,
        btc_lookback=btc_lookback,
        panic_vol_threshold=panic_vol_threshold,
    )
    vix_state = _make_vix_state(vix_regime, vix_risk_mult)
    mock_mgr = MagicMock()
    mock_mgr.get_current_state.return_value = vix_state
    router._vix_manager_fn = lambda: mock_mgr  # type: ignore[assignment]
    return router


# ── imports ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_global_router():
    """
    Reset the process-wide singleton before and after each test so tests
    cannot bleed state into each other.
    """
    import risk.regime_router as _rr_mod
    original = _rr_mod._GLOBAL_ROUTER
    _rr_mod._GLOBAL_ROUTER = None
    yield
    _rr_mod._GLOBAL_ROUTER = original


# ═════════════════════════════════════════════════════════════════════════════
# Initialisation
# ═════════════════════════════════════════════════════════════════════════════

class TestRegimeRouterInit:
    def test_default_params_accepted(self):
        from risk.regime_router import RegimeRouter
        r = RegimeRouter()
        assert r._crypto_beta_scalar == pytest.approx(0.55)
        assert r._panic_vol_threshold == pytest.approx(3.5)

    def test_custom_beta_scalar(self):
        from risk.regime_router import RegimeRouter
        r = RegimeRouter(crypto_beta_scalar=0.30)
        assert r._crypto_beta_scalar == pytest.approx(0.30)

    def test_invalid_beta_scalar_zero_raises(self):
        from risk.regime_router import RegimeRouter
        with pytest.raises(ValueError, match="crypto_beta_scalar"):
            RegimeRouter(crypto_beta_scalar=0.0)

    def test_invalid_beta_scalar_over_one_raises(self):
        from risk.regime_router import RegimeRouter
        with pytest.raises(ValueError, match="crypto_beta_scalar"):
            RegimeRouter(crypto_beta_scalar=1.01)

    def test_boundary_beta_scalar_one_accepted(self):
        from risk.regime_router import RegimeRouter
        r = RegimeRouter(crypto_beta_scalar=1.0)
        assert r._crypto_beta_scalar == pytest.approx(1.0)


# ═════════════════════════════════════════════════════════════════════════════
# record_btc_price
# ═════════════════════════════════════════════════════════════════════════════

class TestRecordBtcPrice:
    def test_valid_prices_accumulate(self):
        from risk.regime_router import RegimeRouter
        r = RegimeRouter(btc_lookback=5)
        for p in [100.0, 200.0, 300.0]:
            r.record_btc_price(p)
        assert len(r._btc_prices) == 3

    def test_nonfinite_price_ignored(self):
        from risk.regime_router import RegimeRouter
        r = RegimeRouter()
        r.record_btc_price(float("nan"))
        r.record_btc_price(float("inf"))
        r.record_btc_price(-float("inf"))
        assert r._btc_prices == []

    def test_zero_price_ignored(self):
        from risk.regime_router import RegimeRouter
        r = RegimeRouter()
        r.record_btc_price(0.0)
        assert r._btc_prices == []

    def test_negative_price_ignored(self):
        from risk.regime_router import RegimeRouter
        r = RegimeRouter()
        r.record_btc_price(-50.0)
        assert r._btc_prices == []

    def test_buffer_capped_at_2x_lookback(self):
        from risk.regime_router import RegimeRouter
        lookback = 10
        r = RegimeRouter(btc_lookback=lookback)
        for i in range(1, 100):
            r.record_btc_price(float(i))
        assert len(r._btc_prices) <= lookback * 2

    def test_thread_safety(self):
        """Concurrent writers must not corrupt the buffer."""
        from risk.regime_router import RegimeRouter
        r = RegimeRouter(btc_lookback=50)
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(200):
                    r.record_btc_price(float(i + 1))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(r._btc_prices) <= 50 * 2


# ═════════════════════════════════════════════════════════════════════════════
# evaluate() — caching
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluateCaching:
    def test_second_call_within_cache_returns_same_object(self):
        router = _router_with_mocked_vix(cache_seconds=300.0)
        d1 = router.evaluate()
        d2 = router.evaluate()
        assert d1 is d2  # exact same object — cached

    def test_call_after_cache_expiry_recomputes(self):
        router = _router_with_mocked_vix(cache_seconds=0.0)
        d1 = router.evaluate()
        time.sleep(0.01)
        d2 = router.evaluate()
        # Not the same object because cache_seconds=0 forces recompute
        assert d1 is not d2

    def test_timestamp_is_recent(self):
        router = _router_with_mocked_vix(cache_seconds=0.0)
        before = datetime.utcnow()
        decision = router.evaluate()
        after = datetime.utcnow()
        assert before <= decision.timestamp <= after


# ═════════════════════════════════════════════════════════════════════════════
# _compute() — regime classification branches
# ═════════════════════════════════════════════════════════════════════════════

class TestRegimeBranches:
    """Drive each of the 3 CompositeRegime values."""

    def test_ranging_regime_default(self):
        """No BTC data + normal VIX → RANGING."""
        from risk.regime_router import CompositeRegime
        router = _router_with_mocked_vix(vix_regime="normal")
        d = router.evaluate()
        assert d.regime is CompositeRegime.RANGING
        assert d.block_new_entries is False
        assert d.notional_multiplier == pytest.approx(1.0)

    def test_high_vol_panic_from_vix_panic(self):
        """VIX panic label → HIGH_VOL_PANIC regardless of crypto."""
        from risk.regime_router import CompositeRegime
        router = _router_with_mocked_vix(vix_regime="panic", vix_risk_mult=0.45)
        d = router.evaluate()
        assert d.regime is CompositeRegime.HIGH_VOL_PANIC
        assert d.block_new_entries is True
        # notional_multiplier is clamped to [0.10, 1.50]
        assert 0.10 <= d.notional_multiplier <= 1.50

    def test_high_vol_panic_from_vix_fear(self):
        """VIX fear label also maps to HIGH_VOL_PANIC."""
        from risk.regime_router import CompositeRegime
        router = _router_with_mocked_vix(vix_regime="fear", vix_risk_mult=0.75)
        d = router.evaluate()
        assert d.regime is CompositeRegime.HIGH_VOL_PANIC
        assert d.block_new_entries is True

    def test_high_vol_panic_from_crypto_vol_spike(self):
        """
        Even with normal VIX, if BTC vol_ratio exceeds panic_vol_threshold
        the router must return HIGH_VOL_PANIC.

        We synthesise a BTC price series whose 5-day realised vol is far above
        the 60-day baseline by injecting an extreme price series.
        """
        from risk.regime_router import CompositeRegime, RegimeRouter
        from market.market_regime_detector import MarketRegimeDetector

        # Build a router with very tight panic threshold so we can trigger it
        # without an astronomically volatile price series.
        router = _router_with_mocked_vix(
            vix_regime="normal",
            vix_risk_mult=1.0,
            btc_lookback=10,
            panic_vol_threshold=0.01,   # effectively always panic if any vol detected
        )

        # Feed enough prices to fill the lookback
        np.random.seed(0)
        for p in np.linspace(100, 200, 20):
            router.record_btc_price(float(p))

        # Manually set the crypto detector to return a vol_ratio above threshold
        mock_detector = MagicMock()
        mock_detector.detect_regime.return_value = {"regime": "BULL", "vol_ratio": 99.9}
        router._crypto_detector = mock_detector

        d = router.evaluate()
        assert d.regime is CompositeRegime.HIGH_VOL_PANIC
        assert d.block_new_entries is True

    def test_trending_regime(self):
        """BULL crypto + normal VIX → TRENDING."""
        from risk.regime_router import CompositeRegime

        router = _router_with_mocked_vix(
            vix_regime="normal",
            vix_risk_mult=1.0,
            btc_lookback=5,
            panic_vol_threshold=99.0,   # never panic
        )
        # Fill enough BTC bars
        for p in range(1, 15):
            router.record_btc_price(float(p * 100))

        mock_detector = MagicMock()
        mock_detector.detect_regime.return_value = {"regime": "BULL", "vol_ratio": 1.2}
        router._crypto_detector = mock_detector

        d = router.evaluate()
        assert d.regime is CompositeRegime.TRENDING
        assert d.block_new_entries is False

    def test_ranging_when_not_enough_btc_data(self):
        """Fewer BTC bars than btc_lookback → crypto_regime = RANGING_DEFAULT."""
        from risk.regime_router import CompositeRegime
        router = _router_with_mocked_vix(vix_regime="normal", btc_lookback=50)
        # Only 3 bars — not enough
        for p in [100.0, 101.0, 102.0]:
            router.record_btc_price(p)
        d = router.evaluate()
        assert d.regime is CompositeRegime.RANGING
        assert d.crypto_regime == "RANGING_DEFAULT"


# ═════════════════════════════════════════════════════════════════════════════
# RegimeDecision fields
# ═════════════════════════════════════════════════════════════════════════════

class TestRegimeDecisionFields:
    def test_crypto_beta_scalar_propagated(self):
        router = _router_with_mocked_vix(crypto_beta_scalar=0.40)
        d = router.evaluate()
        assert d.crypto_beta_scalar == pytest.approx(0.40)

    def test_notional_multiplier_clamped_high(self):
        """Even if VIX risk_mult returns >1.5, result is clamped to 1.5."""
        router = _router_with_mocked_vix(vix_regime="normal", vix_risk_mult=5.0)
        d = router.evaluate()
        assert d.notional_multiplier <= 1.50

    def test_notional_multiplier_clamped_low(self):
        """Even if the table × VIX product would go below 0.10, it is clamped."""
        router = _router_with_mocked_vix(vix_regime="panic", vix_risk_mult=0.001)
        d = router.evaluate()
        assert d.notional_multiplier >= 0.10

    def test_reason_is_non_empty_string(self):
        router = _router_with_mocked_vix()
        d = router.evaluate()
        assert isinstance(d.reason, str) and len(d.reason) > 0

    def test_vix_regime_field_matches_input(self):
        router = _router_with_mocked_vix(vix_regime="elevated")
        d = router.evaluate()
        assert d.vix_regime == "elevated"

    def test_block_entries_false_for_ranging(self):
        from risk.regime_router import CompositeRegime
        router = _router_with_mocked_vix(vix_regime="normal")
        d = router.evaluate()
        assert d.block_new_entries is False

    def test_block_entries_false_for_trending(self):
        from risk.regime_router import CompositeRegime
        router = _router_with_mocked_vix(vix_regime="normal", btc_lookback=5, panic_vol_threshold=99.0)
        for p in range(1, 15):
            router.record_btc_price(float(p * 100))
        mock_det = MagicMock()
        mock_det.detect_regime.return_value = {"regime": "BULL", "vol_ratio": 1.0}
        router._crypto_detector = mock_det
        d = router.evaluate()
        assert d.block_new_entries is False


# ═════════════════════════════════════════════════════════════════════════════
# Singleton
# ═════════════════════════════════════════════════════════════════════════════

class TestGetGlobalRegimeRouter:
    def test_returns_same_instance_on_two_calls(self):
        from risk.regime_router import get_global_regime_router
        r1 = get_global_regime_router()
        r2 = get_global_regime_router()
        assert r1 is r2

    def test_beta_scalar_set_on_first_call(self):
        from risk.regime_router import get_global_regime_router
        r = get_global_regime_router(crypto_beta_scalar=0.33)
        assert r._crypto_beta_scalar == pytest.approx(0.33)

    def test_second_call_ignores_different_scalar(self):
        """After the singleton is created, kwarg on 2nd call is silently ignored."""
        from risk.regime_router import get_global_regime_router
        r1 = get_global_regime_router(crypto_beta_scalar=0.33)
        r2 = get_global_regime_router(crypto_beta_scalar=0.99)
        assert r1 is r2
        assert r2._crypto_beta_scalar == pytest.approx(0.33)

    def test_thread_safe_creation(self):
        """Concurrent first-calls from many threads must all return same object."""
        import risk.regime_router as _rr_mod
        _rr_mod._GLOBAL_ROUTER = None

        from risk.regime_router import get_global_regime_router
        results: list = []

        def _call():
            results.append(get_global_regime_router())

        threads = [threading.Thread(target=_call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(r) for r in results)) == 1, "Multiple singleton instances created"


# ═════════════════════════════════════════════════════════════════════════════
# Graceful degradation when dependencies are absent
# ═════════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    def test_vix_manager_unavailable_falls_back_to_ranging(self):
        """When VIX manager is None, _compute must still return a valid decision."""
        from risk.regime_router import CompositeRegime, RegimeRouter
        router = RegimeRouter(cache_seconds=0.0)
        router._vix_manager_fn = None  # type: ignore[assignment]
        d = router.evaluate()
        # Should not raise; should default to RANGING or TRENDING (not panic without data)
        assert d.regime in {CompositeRegime.RANGING, CompositeRegime.TRENDING}
        assert 0.10 <= d.notional_multiplier <= 1.50

    def test_vix_manager_raises_falls_back(self):
        """If VIX manager raises on get_current_state, router still returns valid decision."""
        from risk.regime_router import RegimeRouter
        router = RegimeRouter(cache_seconds=0.0)
        bad_mgr = MagicMock()
        bad_mgr.get_current_state.side_effect = RuntimeError("network down")
        router._vix_manager_fn = lambda: bad_mgr  # type: ignore[assignment]
        d = router.evaluate()
        assert d is not None
        assert 0.10 <= d.notional_multiplier <= 1.50

    def test_crypto_detector_raises_ignored(self):
        """If crypto detector raises, the router continues without panic."""
        from risk.regime_router import CompositeRegime
        router = _router_with_mocked_vix(vix_regime="normal", btc_lookback=5, panic_vol_threshold=99.0)
        for p in range(1, 15):
            router.record_btc_price(float(p * 100))

        bad_det = MagicMock()
        bad_det.detect_regime.side_effect = Exception("detector crash")
        router._crypto_detector = bad_det

        d = router.evaluate()
        # Should not raise; fallback to RANGING or TRENDING
        assert d.regime in {CompositeRegime.RANGING, CompositeRegime.TRENDING}
