"""
tests/test_iv_crush_strategy.py — Unit tests for models/iv_crush_strategy.py
Tests use mock/patching to avoid live yfinance calls.
"""
import json
import time
from unittest.mock import patch

from models.iv_crush_strategy import (
    IVCrushStrategy,
    IVCrushSignal,
    IV_CRUSH_SIGNAL_SCALE,
    IV_ELEVATION_THRESHOLD,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_strat(tmp_path=None, threshold=IV_ELEVATION_THRESHOLD) -> IVCrushStrategy:
    if tmp_path:
        return IVCrushStrategy(iv_elevation_threshold=threshold, state_dir=tmp_path)
    return IVCrushStrategy(iv_elevation_threshold=threshold)


def _mock_earnings(days_to=3, surprise=0.05, price_gap=0.0) -> dict:
    return {
        "days_to_earnings": float(days_to),
        "earnings_date": "2026-04-25",
        "surprise_pct": float(surprise),
        "price_gap_pct": float(price_gap),
    }


def _mock_iv(current_iv=0.60, avg_iv=0.40) -> dict:
    return {"current_iv": current_iv, "avg_iv": avg_iv}


# ─── TestInit ─────────────────────────────────────────────────────────────────

class TestInit:
    def test_default_threshold(self):
        strat = make_strat()
        assert strat._iv_threshold == IV_ELEVATION_THRESHOLD

    def test_custom_threshold(self):
        strat = make_strat(threshold=1.5)
        assert strat._iv_threshold == 1.5

    def test_no_signals_on_init(self):
        strat = make_strat()
        assert len(strat._signals) == 0


# ─── TestComputeSignal ────────────────────────────────────────────────────────

class TestComputeSignal:
    def test_iv_crush_signal_pre_earnings(self):
        strat = make_strat(threshold=1.4)
        with patch.object(strat, "_get_earnings_info", return_value=_mock_earnings(days_to=3)):
            with patch.object(strat, "_get_iv_data", return_value=_mock_iv(current_iv=0.70, avg_iv=0.40)):
                sig = strat._compute_signal("AAPL")
        assert sig.strategy == "iv_crush"
        assert sig.signal < 0  # bearish (short vol)
        assert sig.confidence > 0

    def test_no_signal_when_iv_not_elevated(self):
        strat = make_strat(threshold=1.4)
        with patch.object(strat, "_get_earnings_info", return_value=_mock_earnings(days_to=3)):
            with patch.object(strat, "_get_iv_data", return_value=_mock_iv(current_iv=0.45, avg_iv=0.40)):
                sig = strat._compute_signal("AAPL")
        # IV elevation = 1.125 < 1.4 → no signal
        assert sig.strategy == "none"
        assert sig.signal == 0.0

    def test_no_signal_when_too_far_from_earnings(self):
        strat = make_strat(threshold=1.4)
        with patch.object(strat, "_get_earnings_info", return_value=_mock_earnings(days_to=10)):
            with patch.object(strat, "_get_iv_data", return_value=_mock_iv(current_iv=0.70, avg_iv=0.40)):
                sig = strat._compute_signal("AAPL")
        assert sig.strategy == "none"

    def test_pead_long_signal_after_positive_gap(self):
        strat = make_strat()
        with patch.object(strat, "_get_earnings_info",
                          return_value=_mock_earnings(days_to=-1, price_gap=0.07)):
            with patch.object(strat, "_get_iv_data", return_value=_mock_iv()):
                sig = strat._compute_signal("AAPL")
        assert sig.strategy == "pead_long"
        assert sig.signal > 0  # bullish drift

    def test_pead_short_signal_after_negative_gap(self):
        strat = make_strat()
        with patch.object(strat, "_get_earnings_info",
                          return_value=_mock_earnings(days_to=-2, price_gap=-0.05)):
            with patch.object(strat, "_get_iv_data", return_value=_mock_iv()):
                sig = strat._compute_signal("AAPL")
        assert sig.strategy == "pead_short"
        assert sig.signal < 0

    def test_no_pead_signal_when_gap_too_small(self):
        strat = make_strat()
        with patch.object(strat, "_get_earnings_info",
                          return_value=_mock_earnings(days_to=-1, price_gap=0.005)):
            with patch.object(strat, "_get_iv_data", return_value=_mock_iv()):
                sig = strat._compute_signal("AAPL")
        assert sig.strategy == "none"

    def test_no_signal_when_no_earnings_data(self):
        strat = make_strat()
        with patch.object(strat, "_get_earnings_info", return_value={}):
            with patch.object(strat, "_get_iv_data", return_value=_mock_iv(current_iv=0.70, avg_iv=0.40)):
                sig = strat._compute_signal("AAPL")
        assert sig.strategy == "none"

    def test_graceful_on_exception(self):
        strat = make_strat()
        with patch.object(strat, "_get_earnings_info", side_effect=RuntimeError("api error")):
            sig = strat._compute_signal("AAPL")
        assert sig.strategy == "none"
        assert sig.signal == 0.0

    def test_signal_magnitude_capped(self):
        strat = make_strat(threshold=1.1)
        with patch.object(strat, "_get_earnings_info", return_value=_mock_earnings(days_to=1)):
            with patch.object(strat, "_get_iv_data", return_value=_mock_iv(current_iv=5.0, avg_iv=0.40)):
                sig = strat._compute_signal("AAPL")
        # raw_strength is capped at 2.0, so max signal = 2 * IV_CRUSH_SIGNAL_SCALE
        assert abs(sig.signal) <= 2 * IV_CRUSH_SIGNAL_SCALE


# ─── TestGetSignal ────────────────────────────────────────────────────────────

class TestGetSignal:
    def test_returns_cached_signal(self):
        strat = make_strat()
        cached = IVCrushSignal(
            symbol="AAPL", days_to_earnings=3, iv_elevation=1.6,
            signal=-0.10, confidence=0.70, strategy="iv_crush",
            last_updated=time.time(),
        )
        strat._signals["AAPL"] = cached
        sig = strat.get_signal("AAPL")
        assert sig is cached

    def test_refreshes_expired_cache(self):
        strat = make_strat()
        old_sig = IVCrushSignal(
            symbol="AAPL", days_to_earnings=3, iv_elevation=1.6,
            signal=-0.10, confidence=0.70, strategy="iv_crush",
            last_updated=0.0,  # expired
        )
        strat._signals["AAPL"] = old_sig
        with patch.object(strat, "_get_earnings_info", return_value={}):
            with patch.object(strat, "_get_iv_data", return_value={}):
                sig = strat.get_signal("AAPL")
        # Should have computed a fresh signal
        assert sig.last_updated > 0

    def test_stores_signal_after_compute(self):
        strat = make_strat()
        with patch.object(strat, "_get_earnings_info", return_value={}):
            with patch.object(strat, "_get_iv_data", return_value={}):
                strat.get_signal("MSFT")
        assert "MSFT" in strat._signals


# ─── TestGetAllSignals ────────────────────────────────────────────────────────

class TestGetAllSignals:
    def test_returns_dict_keyed_by_symbol(self):
        strat = make_strat()
        with patch.object(strat, "_get_earnings_info", return_value={}):
            with patch.object(strat, "_get_iv_data", return_value={}):
                result = strat.get_all_signals(["AAPL", "MSFT"])
        assert set(result.keys()) == {"AAPL", "MSFT"}


# ─── TestGetSnapshot ─────────────────────────────────────────────────────────

class TestGetSnapshot:
    def test_snapshot_keys(self):
        strat = make_strat()
        snap = strat.get_snapshot()
        for key in ("available", "total_tracked", "active_signals",
                    "upcoming_earnings", "iv_elevation_threshold"):
            assert key in snap

    def test_available_true(self):
        assert make_strat().get_snapshot()["available"] is True

    def test_active_signals_only_non_zero(self):
        strat = make_strat()
        strat._signals["AAPL"] = IVCrushSignal(
            symbol="AAPL", days_to_earnings=3, iv_elevation=1.6,
            signal=-0.10, confidence=0.7, strategy="iv_crush",
        )
        strat._signals["MSFT"] = IVCrushSignal(
            symbol="MSFT", days_to_earnings=None, iv_elevation=1.0,
            signal=0.0, confidence=0.0, strategy="none",
        )
        snap = strat.get_snapshot()
        active_syms = [s["symbol"] for s in snap["active_signals"]]
        assert "AAPL" in active_syms
        assert "MSFT" not in active_syms

    def test_upcoming_earnings_within_7_days(self):
        strat = make_strat()
        strat._signals["NVDA"] = IVCrushSignal(
            symbol="NVDA", days_to_earnings=4, iv_elevation=1.5,
            signal=-0.08, confidence=0.6, strategy="iv_crush",
        )
        strat._signals["TSLA"] = IVCrushSignal(
            symbol="TSLA", days_to_earnings=10, iv_elevation=1.2,
            signal=0.0, confidence=0.0, strategy="none",
        )
        snap = strat.get_snapshot()
        upcoming_syms = [s["symbol"] for s in snap["upcoming_earnings"]]
        assert "NVDA" in upcoming_syms
        assert "TSLA" not in upcoming_syms


# ─── TestIVCrushSignal ────────────────────────────────────────────────────────

class TestIVCrushSignal:
    def test_to_dict_roundtrip(self):
        sig = IVCrushSignal(
            symbol="AAPL", days_to_earnings=3, iv_elevation=1.6,
            signal=-0.10, confidence=0.7, strategy="iv_crush",
            earnings_date="2026-04-25",
        )
        d = sig.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["strategy"] == "iv_crush"
        assert abs(d["signal"] - (-0.10)) < 0.001

    def test_rounding(self):
        sig = IVCrushSignal(
            symbol="X", days_to_earnings=1, iv_elevation=1.666666,
            signal=-0.123456, confidence=0.777777, strategy="iv_crush",
        )
        d = sig.to_dict()
        assert len(str(d["iv_elevation"]).split(".")[-1]) <= 5


# ─── TestPersistence ─────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        strat = IVCrushStrategy(state_dir=tmp_path)
        strat._signals["AAPL"] = IVCrushSignal(
            symbol="AAPL", days_to_earnings=3, iv_elevation=1.6,
            signal=-0.10, confidence=0.70, strategy="iv_crush",
            last_updated=time.time(),
        )
        strat._save()

        strat2 = IVCrushStrategy(state_dir=tmp_path)
        assert "AAPL" in strat2._signals
        assert strat2._signals["AAPL"].strategy == "iv_crush"

    def test_state_file_valid_json(self, tmp_path):
        strat = IVCrushStrategy(state_dir=tmp_path)
        strat._signals["AAPL"] = IVCrushSignal(
            symbol="AAPL", days_to_earnings=3, iv_elevation=1.6,
            signal=-0.10, confidence=0.70, strategy="iv_crush",
            last_updated=time.time(),
        )
        strat._save()
        p = tmp_path / "iv_crush_state.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert "signals" in data

    def test_load_missing_file_no_crash(self, tmp_path):
        strat = IVCrushStrategy(state_dir=tmp_path)
        assert len(strat._signals) == 0

    def test_no_state_dir_no_save(self):
        strat = IVCrushStrategy()
        strat._signals["X"] = IVCrushSignal(
            symbol="X", days_to_earnings=None, iv_elevation=1.0,
            signal=0.0, confidence=0.0, strategy="none",
        )
        strat._save()
        assert strat._state_dir is None
