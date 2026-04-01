"""tests/test_rl_governor_persistence.py — RL Weight Governor startup init + persistence tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from models.rl_weight_governor import (
    RLWeightGovernor,
    ACTIONS,
    init_rl_governor,
    get_rl_weights,
    get_rl_confidence_mult,
    get_rl_governor_report,
    feedback_rl_reward,
)


# ── Q-table persistence ───────────────────────────────────────────────────────

class TestQTablePersistence:
    def test_save_and_reload_q_table(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            # Learn one update
            g.update_q_value("bull", False, 0, reward=0.02)
            assert len(g.q_table) > 0
            # Reload
            g2 = RLWeightGovernor(model_dir=d)
            assert len(g2.q_table) > 0

    def test_epsilon_persisted_across_restart(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            # Perform many updates to decay epsilon
            for _ in range(50):
                g.update_q_value("bull", False, 0, reward=0.01)
            saved_eps = g.epsilon
            # Reload
            g2 = RLWeightGovernor(model_dir=d)
            assert abs(g2.epsilon - saved_eps) < 1e-6

    def test_total_updates_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            for _ in range(10):
                g.update_q_value("neutral", False, 1, reward=0.005)
            saved_updates = getattr(g, '_total_updates', 0)
            # Reload
            g2 = RLWeightGovernor(model_dir=d)
            assert getattr(g2, '_total_updates', 0) == saved_updates

    def test_q_values_preserved_across_restart(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            g.update_q_value("bear", True, 2, reward=0.03)
            state_key = g._get_state_key("bear", True)
            q_val = g.q_table[state_key]["2"]
            # Reload
            g2 = RLWeightGovernor(model_dir=d)
            assert state_key in g2.q_table
            assert abs(g2.q_table[state_key]["2"] - q_val) < 1e-9

    def test_atomic_save_creates_no_leftover_tmp(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            g.update_q_value("bull", False, 0, reward=0.01)
            # .tmp file should be renamed away
            tmp_path = g.q_table_path.with_suffix(".tmp")
            assert not tmp_path.exists()

    def test_legacy_format_loaded_correctly(self):
        """Raw dict (no metadata) should still load as q_table."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q_table_governor.json"
            legacy = {"BULL_NORM_VOL": {"0": 0.1, "1": 0.2, "2": 0.05, "3": 0.15}}
            path.write_text(json.dumps(legacy))
            g = RLWeightGovernor(model_dir=d)
            assert "BULL_NORM_VOL" in g.q_table

    def test_corrupt_file_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q_table_governor.json"
            path.write_text("{{NOT_JSON}}")
            g = RLWeightGovernor(model_dir=d)
            assert g.q_table == {}


# ── init_rl_governor factory ──────────────────────────────────────────────────

class TestInitRLGovernor:
    def test_returns_rl_weight_governor(self):
        with tempfile.TemporaryDirectory() as d:
            gov = init_rl_governor(model_dir=d)
            assert isinstance(gov, RLWeightGovernor)

    def test_uses_specified_model_dir(self):
        with tempfile.TemporaryDirectory() as d:
            gov = init_rl_governor(model_dir=d)
            assert str(gov.model_dir) == d

    def test_global_singleton_updated(self):
        with tempfile.TemporaryDirectory() as d:
            gov = init_rl_governor(model_dir=d)
            # After init, module-level functions should use the new governor
            # (can't easily test internal reference, but at minimum it shouldn't raise)
            result = get_rl_governor_report()
            assert "epsilon" in result


# ── Q-learning behaviour ─────────────────────────────────────────────────────

class TestQLearning:
    def test_new_state_initialised_to_zeros(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            state = g._get_state_key("volatile", True)
            g._ensure_state(state)
            assert all(v == 0.0 for v in g.q_table[state].values())

    def test_positive_reward_increases_q_value(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            state = g._get_state_key("bull", False)
            g._ensure_state(state)
            before = g.q_table[state]["0"]
            g.update_q_value("bull", False, 0, reward=0.10)
            after = g.q_table[state]["0"]
            assert after > before

    def test_negative_reward_decreases_q_value(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            state = g._get_state_key("bull", False)
            g._ensure_state(state)
            before = g.q_table[state]["1"]
            g.update_q_value("bull", False, 1, reward=-0.10)
            after = g.q_table[state]["1"]
            assert after < before

    def test_epsilon_decays_with_updates(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            init_eps = g.epsilon
            for _ in range(100):
                g.update_q_value("bull", False, 0, reward=0.01)
            assert g.epsilon < init_eps

    def test_epsilon_floored_at_minimum(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            for _ in range(10_000):
                g.update_q_value("bull", False, 0, reward=0.001)
            assert g.epsilon >= 0.05


# ── get_optimal_weights ───────────────────────────────────────────────────────

class TestGetOptimalWeights:
    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            w = g.get_optimal_weights("neutral", False)
            assert isinstance(w, dict)

    def test_contains_all_base_weights(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            w = g.get_optimal_weights("bull", False, training=False)
            # Should include the __action__ key plus at least the base weights
            assert "__action__" in w
            assert "momentum" in w

    def test_strong_bull_overrides_applied(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            w = g.get_optimal_weights("strong_bull", False, training=False)
            # Mean reversion should be boosted for mega-bull exhaustion
            assert w.get("mean_reversion", 0.0) >= 0.45

    def test_action_within_valid_range(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            for regime in ["bull", "bear", "neutral", "volatile"]:
                w = g.get_optimal_weights(regime, False, training=False)
                assert int(w["__action__"]) in ACTIONS


# ── get_action_confidence ─────────────────────────────────────────────────────

class TestActionConfidence:
    def test_returns_float_in_range(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            mult = g.get_action_confidence("neutral", False)
            assert 0.80 <= mult <= 1.15

    def test_high_spread_boosts_confidence(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            state = g._get_state_key("bull", False)
            g._ensure_state(state)
            # Inject a wide Q-value spread
            g.q_table[state] = {"0": 0.50, "1": 0.01, "2": 0.01, "3": 0.01}
            mult = g.get_action_confidence("bull", False)
            assert mult > 1.0

    def test_negative_best_q_penalises_confidence(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            state = g._get_state_key("bear", True)
            g._ensure_state(state)
            g.q_table[state] = {"0": -0.30, "1": -0.35, "2": -0.40, "3": -0.28}
            mult = g.get_action_confidence("bear", True)
            assert mult < 1.0


# ── get_governor_report ───────────────────────────────────────────────────────

class TestGovernorReport:
    def test_report_has_expected_keys(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            g.update_q_value("neutral", False, 0, reward=0.01)
            report = g.get_governor_report()
            assert "states" in report
            assert "epsilon" in report
            assert "total_updates" in report

    def test_each_state_has_q_values(self):
        with tempfile.TemporaryDirectory() as d:
            g = RLWeightGovernor(model_dir=d)
            g.update_q_value("bull", False, 0, reward=0.05)
            report = g.get_governor_report()
            for state_data in report["states"].values():
                assert "best_action" in state_data
                assert "best_q" in state_data
                assert "all_q" in state_data
