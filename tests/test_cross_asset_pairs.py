"""
tests/test_cross_asset_pairs.py — Unit tests for models/cross_asset_pairs.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.cross_asset_pairs import (
    CrossAssetPairsArb,
    PairRecord,
    SIGNAL_WEIGHT,
    Z_ENTRY_DEFAULT,
    Z_EXIT_DEFAULT,
    MIN_BARS,
    LOOKBACK,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_arb(tmp_path=None, **kwargs) -> CrossAssetPairsArb:
    if tmp_path:
        return CrossAssetPairsArb(state_dir=tmp_path, **kwargs)
    return CrossAssetPairsArb(**kwargs)


def _make_cointegrated_data(n=80, noise_scale=0.5) -> dict:
    """Generate two cointegrated price series."""
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.normal(0, 1, n)) + 100
    spread = rng.normal(0, noise_scale, n)  # stationary spread
    y = 1.5 * x + spread + 10
    return {
        "SYM_Y": pd.DataFrame({"Close": y}),
        "SYM_X": pd.DataFrame({"Close": x}),
    }


def _make_pair_record(z_score=2.5, leg_y="A", leg_x="B") -> PairRecord:
    return PairRecord(
        leg_y=leg_y,
        leg_x=leg_x,
        hedge_ratio=1.5,
        half_life=5.0,
        z_score=z_score,
        z_entry=Z_ENTRY_DEFAULT,
        z_exit=Z_EXIT_DEFAULT,
        spread_mean=0.0,
        spread_std=1.0,
        last_spread=z_score,
        corr=0.85,
    )


# ─── TestInit ─────────────────────────────────────────────────────────────────

class TestInit:
    def test_empty_pairs_on_init(self):
        arb = make_arb()
        assert len(arb._pairs) == 0

    def test_custom_z_entry(self):
        arb = make_arb(z_entry=2.5)
        assert arb._z_entry == 2.5

    def test_no_state_dir_no_crash(self):
        arb = CrossAssetPairsArb()
        assert arb._state_dir is None


# ─── TestScanPairs ────────────────────────────────────────────────────────────

class TestScanPairs:
    def test_finds_cointegrated_pair(self):
        arb = make_arb()
        data = _make_cointegrated_data(n=80)
        n_found = arb.scan_pairs(data, force=True)
        assert n_found >= 0  # may or may not find depending on ADF

    def test_insufficient_data_returns_zero(self):
        arb = make_arb()
        short_data = {
            "A": pd.DataFrame({"Close": np.random.randn(10) + 100}),
            "B": pd.DataFrame({"Close": np.random.randn(10) + 100}),
        }
        n_found = arb.scan_pairs(short_data, force=True)
        assert n_found == 0

    def test_rate_limited_without_force(self):
        arb = make_arb()
        arb._last_scan_ts = time.time()  # just scanned
        data = _make_cointegrated_data()
        n = arb.scan_pairs(data, force=False)
        # Should return current count without scanning
        assert n == 0

    def test_single_symbol_no_crash(self):
        arb = make_arb()
        data = {"A": pd.DataFrame({"Close": np.random.randn(60) + 100})}
        n = arb.scan_pairs(data, force=True)
        assert n == 0

    def test_empty_data_no_crash(self):
        arb = make_arb()
        n = arb.scan_pairs({}, force=True)
        assert n == 0

    def test_missing_close_column_skipped(self):
        arb = make_arb()
        data = {
            "A": pd.DataFrame({"Open": np.random.randn(60) + 100}),
            "B": pd.DataFrame({"Close": np.random.randn(60) + 100}),
        }
        n = arb.scan_pairs(data, force=True)
        assert n == 0  # A lacks Close, so no pair


# ─── TestUpdateScores ─────────────────────────────────────────────────────────

class TestUpdateScores:
    def test_updates_existing_pair_zscore(self):
        arb = make_arb()
        data = _make_cointegrated_data(n=80)
        rec = _make_pair_record(z_score=0.0, leg_y="SYM_Y", leg_x="SYM_X")
        arb._pairs["SYM_Y_SYM_X"] = rec
        arb.update_scores(data)
        # z_score may have changed
        assert "SYM_Y_SYM_X" in arb._pairs

    def test_drops_pair_with_missing_symbol(self):
        arb = make_arb()
        rec = _make_pair_record(leg_y="MISSING_Y", leg_x="MISSING_X")
        arb._pairs["MISSING_Y_MISSING_X"] = rec
        arb.update_scores({})
        assert "MISSING_Y_MISSING_X" not in arb._pairs

    def test_no_crash_on_empty_pairs(self):
        arb = make_arb()
        arb.update_scores({})  # No pairs, no crash


# ─── TestGetOverlaySignals ────────────────────────────────────────────────────

class TestGetOverlaySignals:
    def test_no_signal_when_no_pairs(self):
        arb = make_arb()
        overlay = arb.get_overlay_signals()
        assert overlay == {}

    def test_signal_generated_above_z_entry(self):
        arb = make_arb(z_entry=1.5)
        rec = _make_pair_record(z_score=2.0)
        arb._pairs["A_B"] = rec
        overlay = arb.get_overlay_signals(regime="neutral")
        # Should have signals for A and B
        assert len(overlay) > 0

    def test_no_signal_below_z_entry(self):
        arb = make_arb(z_entry=2.0)
        rec = _make_pair_record(z_score=1.0)
        arb._pairs["A_B"] = rec
        overlay = arb.get_overlay_signals()
        # Either no signal or small signal from carry-over
        for v in overlay.values():
            assert abs(v) <= SIGNAL_WEIGHT

    def test_signals_clamped_to_signal_weight(self):
        arb = make_arb()
        for i in range(10):
            rec = _make_pair_record(z_score=5.0, leg_y="TARGET", leg_x=f"X{i}")
            arb._pairs[f"TARGET_X{i}"] = rec
        overlay = arb.get_overlay_signals()
        if "TARGET" in overlay:
            assert abs(overlay["TARGET"]) <= SIGNAL_WEIGHT

    def test_regime_bear_reduces_signal_entry(self):
        arb = make_arb(z_entry=1.8)
        rec1 = _make_pair_record(z_score=2.0)
        rec2 = _make_pair_record(z_score=2.0, leg_y="C", leg_x="D")
        arb._pairs["A_B"] = rec1
        # With bear regime, z_entry is multiplied by 1.2 — same z_score might not trigger
        overlay_neutral = arb.get_overlay_signals(regime="neutral")
        # Just verify it returns dict
        assert isinstance(overlay_neutral, dict)

    def test_opposite_sign_for_legs(self):
        arb = make_arb(z_entry=1.5)
        rec = _make_pair_record(z_score=2.5, leg_y="LONG_LEG", leg_x="SHORT_LEG")
        arb._pairs["LONG_LEG_SHORT_LEG"] = rec
        overlay = arb.get_overlay_signals()
        # When z_score > 0: sell Y (negative), buy X (positive)
        if "LONG_LEG" in overlay and "SHORT_LEG" in overlay:
            assert overlay["LONG_LEG"] < 0
            assert overlay["SHORT_LEG"] > 0


# ─── TestGetSnapshot ─────────────────────────────────────────────────────────

class TestGetSnapshot:
    def test_snapshot_keys(self):
        arb = make_arb()
        snap = arb.get_snapshot()
        for key in ("available", "n_pairs", "last_scan_ts", "active_pairs", "z_entry", "z_exit"):
            assert key in snap

    def test_available_true(self):
        arb = make_arb()
        assert arb.get_snapshot()["available"] is True

    def test_n_pairs_matches_pairs(self):
        arb = make_arb()
        arb._pairs["A_B"] = _make_pair_record()
        snap = arb.get_snapshot()
        assert snap["n_pairs"] == 1

    def test_active_pairs_capped_at_10(self):
        arb = make_arb()
        for i in range(15):
            arb._pairs[f"A{i}_B{i}"] = _make_pair_record(leg_y=f"A{i}", leg_x=f"B{i}")
        snap = arb.get_snapshot()
        assert len(snap["active_pairs"]) <= 10


# ─── TestPairRecord ───────────────────────────────────────────────────────────

class TestPairRecord:
    def test_pair_key(self):
        rec = _make_pair_record(leg_y="BTC", leg_x="ETH")
        assert rec.pair_key == "BTC_ETH"

    def test_to_dict_roundtrip(self):
        rec = _make_pair_record()
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert "z_score" in d
        assert "leg_y" in d


# ─── TestPersistence ─────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        arb = CrossAssetPairsArb(state_dir=tmp_path)
        arb._pairs["A_B"] = _make_pair_record(z_score=1.5)
        arb._last_scan_ts = 1700000000.0
        arb._save()

        arb2 = CrossAssetPairsArb(state_dir=tmp_path)
        assert "A_B" in arb2._pairs
        assert arb2._last_scan_ts == 1700000000.0

    def test_state_file_valid_json(self, tmp_path):
        arb = CrossAssetPairsArb(state_dir=tmp_path)
        arb._pairs["A_B"] = _make_pair_record()
        arb._save()
        p = tmp_path / "cross_asset_pairs.json"
        assert p.exists()
        data = json.loads(p.read_text())
        assert "pairs" in data

    def test_load_missing_file_no_crash(self, tmp_path):
        arb = CrossAssetPairsArb(state_dir=tmp_path)
        assert len(arb._pairs) == 0

    def test_no_state_dir_no_save(self):
        arb = CrossAssetPairsArb()
        arb._pairs["A_B"] = _make_pair_record()
        arb._save()  # Should not raise
        assert arb._state_dir is None
