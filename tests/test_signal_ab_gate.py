"""Tests for SignalABGate (Thompson Sampling A/B gate)."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from monitoring.signal_ab_gate import SignalABGate, VariantState


# ── helpers ───────────────────────────────────────────────────────────────────

_CTRL_W = {"ml": 0.50, "tech": 0.30, "sentiment": 0.20}
_CHALL_W = {"ml": 0.60, "tech": 0.25, "sentiment": 0.15}


def _gate(**kwargs) -> SignalABGate:
    return SignalABGate(**kwargs)


def _seeded_gate(ctrl=None, chall=None, **kwargs) -> SignalABGate:
    g = _gate(**kwargs)
    g.register_challenger(ctrl or _CTRL_W)          # becomes control
    if chall:
        g.register_challenger(chall)                # becomes challenger
    return g


# ── VariantState ──────────────────────────────────────────────────────────────

class TestVariantState:
    def test_initial_win_rate_is_half(self):
        v = VariantState(name="x", weights={})
        assert v.win_rate_mean == pytest.approx(0.5)

    def test_record_win_updates_alpha(self):
        v = VariantState(name="x", weights={}, alpha=1.0, beta_=1.0, n_trades=0)
        v.record(True)
        assert v.alpha == pytest.approx(2.0)
        assert v.n_trades == 1

    def test_record_loss_updates_beta(self):
        v = VariantState(name="x", weights={}, alpha=1.0, beta_=1.0, n_trades=0)
        v.record(False)
        assert v.beta_ == pytest.approx(2.0)
        assert v.n_trades == 1

    def test_win_rate_mean_after_trades(self):
        v = VariantState(name="x", weights={}, alpha=1.0, beta_=1.0, n_trades=0)
        for _ in range(7):
            v.record(True)
        for _ in range(3):
            v.record(False)
        # alpha=8, beta=4 → mean=8/12≈0.667
        assert v.win_rate_mean == pytest.approx(8 / 12)

    def test_sample_in_unit_range(self):
        v = VariantState(name="x", weights={})
        for _ in range(100):
            assert 0.0 <= v.sample() <= 1.0

    def test_to_dict_keys(self):
        v = VariantState(name="test", weights={"a": 0.5})
        d = v.to_dict()
        for k in ("name", "weights", "alpha", "beta_", "n_trades", "win_rate_mean"):
            assert k in d


# ── Default state ─────────────────────────────────────────────────────────────

class TestDefaultState:
    def test_no_control_active_weights_none(self):
        g = _gate()
        assert g.active_weights is None

    def test_should_use_challenger_false_when_no_challenger(self):
        g = _seeded_gate()
        assert g.should_use_challenger() is False

    def test_status_keys(self):
        g = _gate()
        s = g.get_status()
        assert "control" in s
        assert "challenger" in s
        assert "promotions" in s


# ── register_challenger ───────────────────────────────────────────────────────

class TestRegisterChallenger:
    def test_first_call_becomes_control(self):
        g = _gate()
        g.register_challenger(_CTRL_W, "control")
        assert g.active_weights == _CTRL_W
        status = g.get_status()
        assert status["challenger"] is None

    def test_second_call_becomes_challenger(self):
        g = _seeded_gate(_CTRL_W, _CHALL_W)
        assert g.get_status()["challenger"] is not None
        assert g.get_status()["control"]["weights"] == _CTRL_W

    def test_set_control_weights_overwrites(self):
        g = _gate()
        g.set_control_weights({"ml": 0.4, "tech": 0.6})
        assert g.active_weights == {"ml": 0.4, "tech": 0.6}

    def test_register_replaces_old_challenger(self):
        g = _seeded_gate(_CTRL_W, _CHALL_W)
        g.register_challenger({"ml": 0.70, "tech": 0.20, "sentiment": 0.10}, "v2")
        # new challenger replaces old
        assert g.get_status()["challenger"]["weights"]["ml"] == pytest.approx(0.70)


# ── should_use_challenger ─────────────────────────────────────────────────────

class TestShouldUseChallenger:
    def test_returns_bool(self):
        g = _seeded_gate(_CTRL_W, _CHALL_W)
        result = g.should_use_challenger()
        assert isinstance(result, bool)

    def test_no_challenger_always_false(self):
        g = _seeded_gate(_CTRL_W)
        for _ in range(20):
            assert g.should_use_challenger() is False

    def test_heavily_superior_challenger_wins_more_often(self):
        """Challenger with strong prior should be selected more than 50%."""
        g = _gate()
        g.set_control_weights(_CTRL_W)
        # Simulate a challenger that has won 90 of 100 trades
        from monitoring.signal_ab_gate import VariantState
        g._challenger = VariantState(
            name="strong_challenger",
            weights=_CHALL_W,
            alpha=91.0,   # 90 wins + prior
            beta_=11.0,   # 10 losses + prior
            n_trades=100,
        )
        wins = sum(1 for _ in range(500) if g.should_use_challenger())
        assert wins > 300  # expect clearly > 50%


# ── record_outcome ─────────────────────────────────────────────────────────────

class TestRecordOutcome:
    def test_records_control_win(self):
        g = _seeded_gate(_CTRL_W, _CHALL_W)
        before = g.get_status()["control"]["alpha"]
        g.record_outcome(used_challenger=False, win=True)
        after = g.get_status()["control"]["alpha"]
        assert after == pytest.approx(before + 1.0)

    def test_records_challenger_win(self):
        g = _seeded_gate(_CTRL_W, _CHALL_W)
        before = g.get_status()["challenger"]["alpha"]
        g.record_outcome(used_challenger=True, win=True)
        after = g.get_status()["challenger"]["alpha"]
        assert after == pytest.approx(before + 1.0)

    def test_records_challenger_loss(self):
        g = _seeded_gate(_CTRL_W, _CHALL_W)
        before = g.get_status()["challenger"]["beta_"]
        g.record_outcome(used_challenger=True, win=False)
        after = g.get_status()["challenger"]["beta_"]
        assert after == pytest.approx(before + 1.0)

    def test_no_challenger_outcome_no_crash(self):
        g = _seeded_gate(_CTRL_W)
        g.record_outcome(used_challenger=True, win=True)  # no crash


# ── maybe_rollback ────────────────────────────────────────────────────────────

class TestMaybeRollback:
    def test_rollback_when_challenger_far_below_control(self):
        g = _gate(rollback_ic_drop=0.05, min_trades=5)
        g.set_control_weights(_CTRL_W)
        from monitoring.signal_ab_gate import VariantState
        # control win rate ≈ 0.75, challenger ≈ 0.55 → drop = 0.20 > 0.05
        g._control = VariantState(
            name="control", weights=_CTRL_W,
            alpha=75.0, beta_=25.0, n_trades=100,
        )
        g._challenger = VariantState(
            name="challenger", weights=_CHALL_W,
            alpha=11.0, beta_=9.0, n_trades=20,
        )
        rolled = g.maybe_rollback()
        assert rolled is True
        assert g.get_status()["challenger"] is None

    def test_no_rollback_when_challenger_close_to_control(self):
        g = _gate(rollback_ic_drop=0.10, min_trades=5)
        g.set_control_weights(_CTRL_W)
        from monitoring.signal_ab_gate import VariantState
        # control ≈ 0.60, challenger ≈ 0.58 → drop < 0.10
        g._control = VariantState(
            name="control", weights=_CTRL_W,
            alpha=61.0, beta_=41.0, n_trades=100,
        )
        g._challenger = VariantState(
            name="challenger", weights=_CHALL_W,
            alpha=17.0, beta_=13.0, n_trades=30,
        )
        rolled = g.maybe_rollback()
        assert rolled is False

    def test_no_rollback_insufficient_trades(self):
        g = _gate(rollback_ic_drop=0.01, min_trades=30)
        g.set_control_weights(_CTRL_W)
        from monitoring.signal_ab_gate import VariantState
        g._control = VariantState(
            name="control", weights=_CTRL_W,
            alpha=75.0, beta_=25.0, n_trades=100,
        )
        g._challenger = VariantState(
            name="challenger", weights=_CHALL_W,
            alpha=2.0, beta_=8.0, n_trades=9,  # < min_trades//3
        )
        rolled = g.maybe_rollback()
        assert rolled is False


# ── maybe_promote ─────────────────────────────────────────────────────────────

class TestMaybePromote:
    def test_no_promote_insufficient_trades(self):
        g = _seeded_gate(_CTRL_W, _CHALL_W, min_trades=30)
        # challenger has 0 trades
        result = g.maybe_promote()
        assert result is None

    def test_no_promote_shadow_period_not_elapsed(self):
        g = _gate(min_trades=5, shadow_hours=48.0)
        g.set_control_weights(_CTRL_W)
        from monitoring.signal_ab_gate import VariantState
        g._challenger = VariantState(
            name="challenger", weights=_CHALL_W,
            alpha=91.0, beta_=11.0, n_trades=100,
            created_at=time.time() - 1,   # just 1 second ago
        )
        result = g.maybe_promote()
        assert result is None  # shadow period not met

    def test_promote_when_all_gates_pass(self):
        g = _gate(min_trades=5, shadow_hours=0.0, promotion_prob=0.50)
        g.set_control_weights(_CTRL_W)
        from monitoring.signal_ab_gate import VariantState
        g._challenger = VariantState(
            name="challenger", weights=_CHALL_W,
            alpha=90.0, beta_=10.0, n_trades=100,
            created_at=time.time() - 1,   # shadow_hours=0 → passes
        )
        result = g.maybe_promote()
        assert result == _CHALL_W
        assert g.get_status()["challenger"] is None
        assert g.active_weights == _CHALL_W

    def test_promotion_history_recorded(self):
        g = _gate(min_trades=5, shadow_hours=0.0, promotion_prob=0.50)
        g.set_control_weights(_CTRL_W)
        from monitoring.signal_ab_gate import VariantState
        g._challenger = VariantState(
            name="challenger", weights=_CHALL_W,
            alpha=90.0, beta_=10.0, n_trades=100,
            created_at=time.time() - 1,
        )
        g.maybe_promote()
        assert g.get_status()["promotions"] == 1


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_control_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            g1 = SignalABGate(data_dir=d)
            g1.set_control_weights(_CTRL_W)
            g2 = SignalABGate(data_dir=d)
            assert g2.active_weights == _CTRL_W

    def test_challenger_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            g1 = SignalABGate(data_dir=d)
            g1.set_control_weights(_CTRL_W)
            g1.register_challenger(_CHALL_W, "v2")
            g2 = SignalABGate(data_dir=d)
            status = g2.get_status()
            assert status["challenger"] is not None
            assert status["challenger"]["weights"] == _CHALL_W

    def test_outcome_counts_survive_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            g1 = SignalABGate(data_dir=d)
            g1.set_control_weights(_CTRL_W)
            g1.register_challenger(_CHALL_W)
            for _ in range(5):
                g1.record_outcome(used_challenger=True, win=True)
            g2 = SignalABGate(data_dir=d)
            assert g2.get_status()["challenger"]["n_trades"] == 5

    def test_promotion_history_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            g1 = SignalABGate(data_dir=d, min_trades=5, shadow_hours=0.0, promotion_prob=0.50)
            g1.set_control_weights(_CTRL_W)
            from monitoring.signal_ab_gate import VariantState
            g1._challenger = VariantState(
                name="challenger", weights=_CHALL_W,
                alpha=90.0, beta_=10.0, n_trades=100,
                created_at=time.time() - 1,
            )
            g1.maybe_promote()
            g2 = SignalABGate(data_dir=d)
            assert g2.get_status()["promotions"] == 1
