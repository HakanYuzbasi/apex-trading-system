"""Tests for StrategyRotationController."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from monitoring.strategy_rotation import StrategyRotationController, RegimeWeightSet


def _make_ctrl(**kwargs) -> StrategyRotationController:
    defaults = dict(min_records=5, temperature=2.0, significance_alpha=0.10)
    defaults.update(kwargs)
    return StrategyRotationController(**defaults)


def _record(ctrl, regime="bull", pnl=0.01, comps=None):
    if comps is None:
        comps = {"ml": 0.55, "tech": 0.45, "sentiment": 0.0, "momentum": 0.0, "pairs": 0.0}
    ctrl.record_outcome(regime=regime, pnl_pct=pnl, components=comps)


class TestDefaultWeights:

    def test_equal_weight_before_data(self):
        ctrl = _make_ctrl()
        weights = ctrl.get_blend_weights("bull")
        for c in ("ml", "tech", "sentiment", "momentum", "pairs"):
            assert weights[c] == pytest.approx(1.0 / 5, abs=0.001)

    def test_equal_weight_insufficient_data(self):
        ctrl = _make_ctrl(min_records=10)
        for _ in range(3):
            _record(ctrl)
        weights = ctrl.get_blend_weights("bull")
        assert all(abs(w - 0.2) < 0.001 for w in weights.values())

    def test_weights_sum_to_one(self):
        ctrl = _make_ctrl()
        for _ in range(20):
            _record(ctrl)
        weights = ctrl.get_blend_weights("bull")
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)


class TestAlphaComputation:

    def test_winning_ml_component_gets_higher_weight(self):
        ctrl = _make_ctrl(min_records=5, significance_alpha=0.30)
        # ML contributes to every winning trade, tech contributes to losers
        for _ in range(15):
            _record(ctrl, pnl=0.02, comps={"ml": 1.0, "tech": 0.0, "sentiment": 0.0, "momentum": 0.0, "pairs": 0.0})
        for _ in range(5):
            _record(ctrl, pnl=-0.01, comps={"ml": 0.0, "tech": 1.0, "sentiment": 0.0, "momentum": 0.0, "pairs": 0.0})
        weights = ctrl.get_blend_weights("bull")
        # ML should get higher weight than tech
        assert weights.get("ml", 0) > weights.get("tech", 0)

    def test_regime_isolation(self):
        ctrl = _make_ctrl(min_records=5)
        # Bull: ML dominant; Bear: tech dominant
        for _ in range(10):
            _record(ctrl, regime="bull", pnl=0.02, comps={"ml": 1.0, "tech": 0.0, "sentiment": 0.0, "momentum": 0.0, "pairs": 0.0})
        for _ in range(10):
            _record(ctrl, regime="bear", pnl=0.02, comps={"ml": 0.0, "tech": 1.0, "sentiment": 0.0, "momentum": 0.0, "pairs": 0.0})
        bull_w = ctrl.get_blend_weights("bull")
        bear_w = ctrl.get_blend_weights("bear")
        assert bull_w.get("ml", 0) != bear_w.get("ml", 0)

    def test_negative_pnl_component_penalised(self):
        ctrl = _make_ctrl(min_records=5, significance_alpha=0.30)
        # Pairs always contributes to losing trades
        for _ in range(15):
            _record(ctrl, pnl=-0.02, comps={"ml": 0.0, "tech": 0.0, "sentiment": 0.0, "momentum": 0.0, "pairs": 1.0})
        for _ in range(5):
            _record(ctrl, pnl=0.03, comps={"ml": 0.5, "tech": 0.5, "sentiment": 0.0, "momentum": 0.0, "pairs": 0.0})
        weights = ctrl.get_blend_weights("bull")
        assert weights.get("pairs", 1.0) < weights.get("ml", 0)


class TestSoftmax:

    def test_equal_scores_equal_weights(self):
        import numpy as np
        ctrl = _make_ctrl()
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        w = ctrl._softmax(x)
        assert all(abs(wi - 0.2) < 0.001 for wi in w)

    def test_highest_score_gets_highest_weight(self):
        import numpy as np
        ctrl = _make_ctrl()
        x = np.array([10.0, 1.0, 0.0, 0.0, 0.0])
        w = ctrl._softmax(x)
        assert w[0] > w[1] > w[2]

    def test_softmax_sums_to_one(self):
        import numpy as np
        ctrl = _make_ctrl()
        x = np.array([0.5, -0.3, 0.1, -0.2, 0.4])
        w = ctrl._softmax(x)
        assert sum(w) == pytest.approx(1.0, abs=1e-6)


class TestRecordAndQuery:

    def test_record_adds_to_history(self):
        ctrl = _make_ctrl()
        _record(ctrl)
        assert len(ctrl._records["bull"]) == 1

    def test_max_records_cap(self):
        ctrl = _make_ctrl(max_records=10)
        for _ in range(20):
            _record(ctrl)
        assert len(ctrl._records["bull"]) == 10

    def test_dirty_regime_recomputed_on_query(self):
        ctrl = _make_ctrl(min_records=5)
        for _ in range(10):
            _record(ctrl)
        assert "bull" in ctrl._dirty_regimes
        ctrl.get_blend_weights("bull")
        assert "bull" not in ctrl._dirty_regimes

    def test_get_all_regimes(self):
        ctrl = _make_ctrl()
        _record(ctrl, regime="bull")
        _record(ctrl, regime="bear")
        regimes = ctrl.get_all_regimes()
        assert "bull" in regimes
        assert "bear" in regimes


class TestReport:

    def test_report_structure(self):
        ctrl = _make_ctrl()
        for _ in range(5):
            _record(ctrl)
        report = ctrl.get_report()
        assert "regimes" in report
        assert "record_counts" in report
        assert "generated_at" in report

    def test_report_contains_regime_weights(self):
        ctrl = _make_ctrl(min_records=5)
        for _ in range(10):
            _record(ctrl)
        report = ctrl.get_report()
        assert "bull" in report["regimes"]
        regime_data = report["regimes"]["bull"]
        assert "weights" in regime_data
        assert "alpha_scores" in regime_data
        assert "record_count" in regime_data

    def test_report_record_counts(self):
        ctrl = _make_ctrl()
        for _ in range(7):
            _record(ctrl)
        report = ctrl.get_report()
        assert report["record_counts"]["bull"] == 7


class TestPersistence:

    def test_weight_cache_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = StrategyRotationController(data_dir=Path(tmp), min_records=5)
            for _ in range(10):
                _record(ctrl)
            ctrl.get_blend_weights("bull")  # triggers persist

            path = Path(tmp) / "strategy_rotation.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert "weight_cache" in data

    def test_state_loaded_on_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl1 = StrategyRotationController(data_dir=Path(tmp), min_records=5)
            for _ in range(10):
                _record(ctrl1)
            ctrl1.get_report()  # flush

            ctrl2 = StrategyRotationController(data_dir=Path(tmp))
            assert "bull" in ctrl2._weight_cache

    def test_no_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = StrategyRotationController(data_dir=Path(tmp))
            assert len(ctrl._weight_cache) == 0
