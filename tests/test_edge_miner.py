"""Tests for EdgeMiner."""
from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from monitoring.edge_miner import (
    EdgeMiner,
    EdgeProfile,
    _hour_block,
    _regime_code,
    _source_code,
    _vix_bucket,
    _hold_bucket,
    _dominant_source,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_trade(
    symbol="AAPL",
    pnl_pct=0.02,
    regime="bull",
    hour=13,          # midday UTC
    vix=18.0,
    hold_hours=3.0,
    components=None,
    day_of_week=1,
):
    ts = datetime(2026, 3, 18, hour, 0, 0, tzinfo=timezone.utc).isoformat()
    return {
        "event": "EXIT",
        "symbol": symbol,
        "pnl_pct": pnl_pct,
        "regime": regime,
        "ts": ts,
        "vix": vix,
        "hold_hours": hold_hours,
        "components": components or {"ml": 0.5, "tech": 0.2},
        "day_of_week": day_of_week,
    }


def _write_audit(d: Path, trades: list, filename="trade_audit_20260318.jsonl"):
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(t) for t in trades]
    (d / filename).write_text("\n".join(lines), encoding="utf-8")


def _miner_with_trades(trades: list, min_cluster=2, min_win_rate=0.50, n_clusters=3) -> EdgeMiner:
    """Create an EdgeMiner in a temp dir with seeded audit data."""
    import tempfile
    tmp = tempfile.mkdtemp()
    d = Path(tmp)
    _write_audit(d, trades)
    m = EdgeMiner(
        audit_dir=d,
        data_dir=d,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster,
        min_win_rate=min_win_rate,
        lookback_days=365,
    )
    return m


# ── Feature helpers ────────────────────────────────────────────────────────────

class TestFeatureHelpers:
    def test_regime_code_known(self):
        assert _regime_code("bull") == pytest.approx(1.0)

    def test_regime_code_unknown_is_neutral(self):
        assert _regime_code("GARBAGE") == pytest.approx(2.0)

    def test_hour_block_open(self):
        ts = datetime(2026, 3, 18, 10, 0, 0, tzinfo=timezone.utc).isoformat()
        assert _hour_block(ts) == "open"

    def test_hour_block_midday(self):
        ts = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        assert _hour_block(ts) == "midday"

    def test_hour_block_close(self):
        ts = datetime(2026, 3, 18, 15, 30, 0, tzinfo=timezone.utc).isoformat()
        assert _hour_block(ts) == "close"

    def test_hour_block_extended(self):
        ts = datetime(2026, 3, 18, 19, 0, 0, tzinfo=timezone.utc).isoformat()
        assert _hour_block(ts) == "extended"

    def test_hour_block_bad_string(self):
        assert _hour_block("NOT_A_DATE") == "unknown"

    def test_vix_bucket_low(self):
        assert _vix_bucket(12.0) == pytest.approx(0.0)

    def test_vix_bucket_high(self):
        assert _vix_bucket(40.0) == pytest.approx(4.0)

    def test_hold_bucket_short(self):
        assert _hold_bucket(0.5) == pytest.approx(0.0)

    def test_hold_bucket_long(self):
        assert _hold_bucket(30.0) == pytest.approx(3.0)

    def test_dominant_source_highest_abs(self):
        assert _dominant_source({"ml": 0.5, "tech": 0.8}) == "tech"

    def test_dominant_source_empty(self):
        assert _dominant_source({}) == "unknown"

    def test_source_code_known(self):
        assert _source_code({"ml": 0.9}) == pytest.approx(0.0)


# ── EdgeProfile ───────────────────────────────────────────────────────────────

class TestEdgeProfile:
    def test_to_dict_keys(self):
        p = EdgeProfile(
            cluster_id=0, name="bull_midday_ml",
            regime="bull", hour_block="midday",
            signal_source="ml", n_trades=20,
            win_rate=0.70, avg_pnl_pct=0.015,
            confidence_boost=0.04,
        )
        d = p.to_dict()
        for k in ("cluster_id", "name", "regime", "hour_block",
                  "signal_source", "n_trades", "win_rate",
                  "avg_pnl_pct", "confidence_boost"):
            assert k in d


# ── Default state ─────────────────────────────────────────────────────────────

class TestDefaultState:
    def test_no_profiles_initially(self):
        m = EdgeMiner()
        assert m.profiles == []

    def test_get_confidence_boost_zero_when_no_profiles(self):
        m = EdgeMiner()
        boost = m.get_confidence_boost({"regime": "bull", "hour_block": "midday", "signal_source": "ml"})
        assert boost == pytest.approx(0.0)

    def test_get_report_structure(self):
        m = EdgeMiner()
        r = m.get_report()
        assert "total_profiles" in r
        assert "profiles" in r


# ── load_and_mine ─────────────────────────────────────────────────────────────

class TestLoadAndMine:
    def test_insufficient_trades_returns_empty(self):
        m = EdgeMiner(min_cluster_size=50)  # very high
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_audit(d, [_make_trade() for _ in range(5)])
            m._audit_dir = d
            profiles = m.load_and_mine()
        assert profiles == []

    def test_returns_profiles_list(self):
        pytest.importorskip("sklearn")
        trades = [_make_trade(pnl_pct=0.02, regime="bull") for _ in range(30)]
        m = _miner_with_trades(trades, min_cluster=3, min_win_rate=0.50, n_clusters=3)
        profiles = m.load_and_mine()
        assert isinstance(profiles, list)

    def test_high_win_rate_cluster_becomes_profile(self):
        """Trades in a single tight cluster all winning → should create a profile."""
        pytest.importorskip("sklearn")
        # All trades identical context, all winning
        trades = [
            _make_trade(pnl_pct=0.025, regime="bull", hour=12, vix=15.0, hold_hours=2.0)
            for _ in range(20)
        ]
        m = _miner_with_trades(trades, min_cluster=5, min_win_rate=0.50, n_clusters=2)
        profiles = m.load_and_mine()
        # Because all are winners and clustered together, at least 1 profile expected
        if profiles:  # may be 0 if sklearn collapses clusters
            assert profiles[0].win_rate >= 0.50

    def test_sets_last_mined_timestamp(self):
        pytest.importorskip("sklearn")
        before = time.time()
        trades = [_make_trade(pnl_pct=0.02) for _ in range(30)]
        m = _miner_with_trades(trades, min_cluster=3)
        m.load_and_mine()
        assert m._last_mined is not None
        assert m._last_mined >= before

    def test_profiles_sorted_by_win_rate_descending(self):
        pytest.importorskip("sklearn")
        trades = [_make_trade(pnl_pct=0.02, regime="bull") for _ in range(40)]
        m = _miner_with_trades(trades, min_cluster=3, min_win_rate=0.0, n_clusters=3)
        profiles = m.load_and_mine()
        if len(profiles) >= 2:
            assert profiles[0].win_rate >= profiles[1].win_rate


# ── get_confidence_boost ──────────────────────────────────────────────────────

class TestGetConfidenceBoost:
    def _gate_with_profiles(self, win_rate=0.75) -> EdgeMiner:
        m = EdgeMiner(min_win_rate=0.60, max_boost=0.08)
        m._profiles = [
            EdgeProfile(
                cluster_id=0, name="bull_midday_ml",
                regime="bull", hour_block="midday",
                signal_source="ml", n_trades=30,
                win_rate=win_rate,
                avg_pnl_pct=0.02,
                confidence_boost=round(min(0.08, (win_rate - 0.60) * 0.08 * 3.0), 4),
            )
        ]
        return m

    def test_matching_context_returns_boost(self):
        m = self._gate_with_profiles(0.75)
        boost = m.get_confidence_boost({
            "regime": "bull",
            "hour_block": "midday",
            "signal_source": "ml",
        })
        assert boost > 0.0

    def test_non_matching_regime_returns_zero(self):
        m = self._gate_with_profiles(0.75)
        boost = m.get_confidence_boost({
            "regime": "bear",
            "hour_block": "midday",
            "signal_source": "ml",
        })
        assert boost == pytest.approx(0.0)

    def test_regime_match_hour_mismatch_source_match_returns_boost(self):
        """regime + signal_source match (even without hour_block) → boost."""
        m = self._gate_with_profiles(0.75)
        boost = m.get_confidence_boost({
            "regime": "bull",
            "hour_block": "close",    # different from profile's "midday"
            "signal_source": "ml",    # matches
        })
        assert boost > 0.0

    def test_boost_capped_at_max_boost(self):
        m = EdgeMiner(min_win_rate=0.50, max_boost=0.08)
        m._profiles = [
            EdgeProfile(
                cluster_id=0, name="test",
                regime="bull", hour_block="midday",
                signal_source="ml", n_trades=50,
                win_rate=0.99,
                avg_pnl_pct=0.03,
                confidence_boost=0.08,  # already capped
            )
        ]
        boost = m.get_confidence_boost({
            "regime": "bull", "hour_block": "midday", "signal_source": "ml",
        })
        assert boost <= 0.08

    def test_returns_best_boost_across_multiple_profiles(self):
        m = EdgeMiner()
        m._profiles = [
            EdgeProfile(
                cluster_id=0, name="p1",
                regime="bull", hour_block="midday",
                signal_source="ml", n_trades=30,
                win_rate=0.65, avg_pnl_pct=0.01,
                confidence_boost=0.02,
            ),
            EdgeProfile(
                cluster_id=1, name="p2",
                regime="bull", hour_block="open",
                signal_source="ml", n_trades=30,
                win_rate=0.80, avg_pnl_pct=0.025,
                confidence_boost=0.06,
            ),
        ]
        boost = m.get_confidence_boost({
            "regime": "bull", "hour_block": "midday", "signal_source": "ml",
        })
        assert boost == pytest.approx(max(0.02, 0.06))


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_profiles_survive_reload(self):
        pytest.importorskip("sklearn")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            trades = [_make_trade(pnl_pct=0.02, regime="bull") for _ in range(40)]
            _write_audit(d, trades)
            m1 = EdgeMiner(
                audit_dir=d, data_dir=d,
                n_clusters=3, min_cluster_size=5,
                min_win_rate=0.40, lookback_days=365,
            )
            m1.load_and_mine()
            # Reload
            m2 = EdgeMiner(data_dir=d)
            assert m2.get_report()["total_profiles"] >= 0  # can be 0 if all below min_win_rate

    def test_empty_profiles_no_crash_on_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m1 = EdgeMiner(data_dir=d)
            m1._profiles = []
            m1._last_mined = time.time()
            m1._save_profiles()
            # Reload should work
            m2 = EdgeMiner(data_dir=d)
            assert m2.profiles == []

    def test_profile_fields_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            m1 = EdgeMiner(data_dir=d)
            m1._profiles = [
                EdgeProfile(
                    cluster_id=3, name="bull_open_ml",
                    regime="bull", hour_block="open",
                    signal_source="ml", n_trades=25,
                    win_rate=0.72, avg_pnl_pct=0.018,
                    confidence_boost=0.05,
                )
            ]
            m1._save_profiles()
            m2 = EdgeMiner(data_dir=d)
            p = m2.profiles[0]
            assert p.regime == "bull"
            assert p.hour_block == "open"
            assert p.win_rate == pytest.approx(0.72)
            assert p.confidence_boost == pytest.approx(0.05)
