"""Tests for RLWeightGovernor enhancements."""
from __future__ import annotations

import tempfile
import pytest

from models.rl_weight_governor import (
    RLWeightGovernor,
    get_rl_weights,
    get_rl_confidence_mult,
    get_rl_governor_report,
    ACTIONS,
)


def _make_gov() -> RLWeightGovernor:
    return RLWeightGovernor(model_dir="/tmp/apex_rl_test")


class TestWeightRetrieval:

    def test_get_rl_weights_returns_dict(self):
        w = get_rl_weights("bull")
        assert isinstance(w, dict)

    def test_weights_contain_action_key(self):
        w = get_rl_weights("neutral")
        assert "__action__" in w

    def test_action_is_valid_int(self):
        w = get_rl_weights("bear")
        action = int(w["__action__"])
        assert action in ACTIONS

    def test_weights_for_unknown_regime(self):
        w = get_rl_weights("unknown_regime_xyz")
        assert isinstance(w, dict)
        assert "__action__" in w


class TestEpsilonDecay:

    def test_epsilon_decays_after_updates(self):
        gov = _make_gov()
        initial_eps = gov.epsilon
        for _ in range(50):
            gov.update_q_value("bull", False, 0, 0.01)
        assert gov.epsilon < initial_eps

    def test_epsilon_does_not_go_below_minimum(self):
        gov = _make_gov()
        for _ in range(2000):
            gov.update_q_value("bull", False, 0, 0.01)
        assert gov.epsilon >= 0.05

    def test_epsilon_bounded_above_initial(self):
        gov = _make_gov()
        assert gov.epsilon <= 0.20


class TestGetBestAction:

    def test_best_action_is_valid(self):
        gov = _make_gov()
        action = gov.get_best_action("bull")
        assert action in ACTIONS

    def test_best_action_after_consistent_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            gov = RLWeightGovernor(model_dir=tmp)
            # Train: action 3 always wins for bull
            for _ in range(30):
                gov.update_q_value("bull", False, 3, 0.05)
                gov.update_q_value("bull", False, 0, -0.01)
            best = gov.get_best_action("bull")
            assert best == 3

    def test_best_action_high_vol_separate(self):
        gov = _make_gov()
        gov.update_q_value("bull", True, 2, 0.05)
        gov.update_q_value("bull", False, 3, 0.05)
        # High-vol and normal-vol are separate states
        hv_best = gov.get_best_action("bull", is_high_volatility=True)
        nv_best = gov.get_best_action("bull", is_high_volatility=False)
        assert isinstance(hv_best, int)
        assert isinstance(nv_best, int)


class TestActionConfidence:

    def test_uninformed_returns_one(self):
        gov = _make_gov()
        # Fresh state: all Q=0, spread=0 → mult=1.0
        mult = gov.get_action_confidence("fresh_regime_xyz")
        assert mult == pytest.approx(1.0, abs=0.01)

    def test_confident_state_boosts_multiplier(self):
        gov = _make_gov()
        for _ in range(20):
            gov.update_q_value("bull", False, 1, 0.05)
            gov.update_q_value("bull", False, 0, -0.05)
        mult = gov.get_action_confidence("bull", False)
        assert mult > 1.0

    def test_negative_q_penalises_multiplier(self):
        gov = _make_gov()
        # Train ALL actions with negative rewards so q_max < 0
        for _ in range(20):
            gov.update_q_value("crisis2", False, 0, -0.05)
            gov.update_q_value("crisis2", False, 1, -0.04)
            gov.update_q_value("crisis2", False, 2, -0.03)
            gov.update_q_value("crisis2", False, 3, -0.02)
        mult = gov.get_action_confidence("crisis2", False)
        assert mult <= 1.0

    def test_confidence_mult_bounded(self):
        gov = _make_gov()
        for _ in range(50):
            gov.update_q_value("bull", False, 1, 0.10)
        mult = gov.get_action_confidence("bull")
        assert 0.85 <= mult <= 1.10


class TestGovernorReport:

    def test_report_structure(self):
        gov = _make_gov()
        gov.update_q_value("bull", False, 0, 0.01)
        report = gov.get_governor_report()
        assert "states" in report
        assert "epsilon" in report
        assert "total_updates" in report

    def test_report_has_state_details(self):
        gov = _make_gov()
        gov.update_q_value("bear", True, 2, -0.02)
        report = gov.get_governor_report()
        # At least one state should exist
        assert len(report["states"]) >= 1
        for state_data in report["states"].values():
            assert "best_action" in state_data
            assert "best_q" in state_data
            assert "q_spread" in state_data
            assert "all_q" in state_data

    def test_epsilon_in_report_decays(self):
        gov = _make_gov()
        for _ in range(100):
            gov.update_q_value("bull", False, 0, 0.01)
        report = gov.get_governor_report()
        assert report["epsilon"] < 0.20

    def test_total_updates_tracked(self):
        gov = _make_gov()
        for _ in range(5):
            gov.update_q_value("neutral", False, 1, 0.02)
        report = gov.get_governor_report()
        assert report["total_updates"] == 5


class TestGlobalFunctions:

    def test_get_rl_confidence_mult_returns_float(self):
        mult = get_rl_confidence_mult("bull")
        assert isinstance(mult, float)

    def test_get_rl_confidence_mult_bounded(self):
        mult = get_rl_confidence_mult("bear", is_high_volatility=True)
        assert 0.85 <= mult <= 1.10

    def test_get_rl_governor_report_returns_dict(self):
        report = get_rl_governor_report()
        assert isinstance(report, dict)
        assert "epsilon" in report
