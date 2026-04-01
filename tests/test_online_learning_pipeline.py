"""
Tests for OnlineLearningPipeline.

Covers: dataset harvesting, champion evaluation, stat gate, state persistence,
cooldown logic, and promotion decisions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from core.online_learning_pipeline import (
    MIN_LABELED_SAMPLES,
    OnlineLearningPipeline,
    PipelineRun,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeSignal:
    symbol: str = "AAPL"
    signal_value: float = 0.6
    confidence: float = 0.7
    regime: str = "bull"
    entry_price: float = 180.0
    generator_signals: Dict[str, float] = field(default_factory=lambda: {"ml": 0.6, "tech": 0.4})
    timestamp: datetime = field(default_factory=datetime.now)
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    completed: bool = False


def _make_outcome_loop(n_correct: int = 40, n_wrong: int = 20):
    """Fake OutcomeFeedbackLoop with labeled completed signals."""
    loop = MagicMock()
    sigs: List[_FakeSignal] = []
    for i in range(n_correct):
        s = _FakeSignal(signal_value=0.5 + 0.1 * (i % 3), return_5d=0.02)
        sigs.append(s)
    for i in range(n_wrong):
        s = _FakeSignal(signal_value=0.5 + 0.1 * (i % 3), return_5d=-0.02)
        sigs.append(s)
    loop._completed_signals = sigs
    return loop


def _make_pipeline(tmp_path: Path, **kwargs) -> OnlineLearningPipeline:
    inst_gen = MagicMock()
    inst_gen.train = MagicMock(return_value=None)
    outcome_loop = _make_outcome_loop()
    return OnlineLearningPipeline(
        inst_generator=inst_gen,
        outcome_loop=outcome_loop,
        state_dir=tmp_path / "olp",
        n_trials=3,
        min_run_interval_hours=0.0,  # no cooldown in tests
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def test_returns_correct_shape(self, tmp_path):
        p = _make_pipeline(tmp_path)
        X, y = p._build_dataset()
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

    def test_label_is_binary(self, tmp_path):
        p = _make_pipeline(tmp_path)
        _, y = p._build_dataset()
        assert set(y).issubset({0.0, 1.0})

    def test_ignores_signals_without_return_5d(self, tmp_path):
        loop = MagicMock()
        loop._completed_signals = [_FakeSignal(return_5d=None)]
        p = OnlineLearningPipeline(
            inst_generator=MagicMock(), outcome_loop=loop,
            state_dir=tmp_path / "olp", n_trials=1, min_run_interval_hours=0,
        )
        X, y = p._build_dataset()
        assert len(y) == 0

    def test_correct_signal_correctly_labeled(self, tmp_path):
        loop = MagicMock()
        loop._completed_signals = [_FakeSignal(signal_value=0.7, return_5d=0.03)]
        p = OnlineLearningPipeline(
            inst_generator=MagicMock(), outcome_loop=loop,
            state_dir=tmp_path / "olp", n_trials=1, min_run_interval_hours=0,
        )
        _, y = p._build_dataset()
        assert y[0] == 1.0

    def test_wrong_signal_labeled_zero(self, tmp_path):
        loop = MagicMock()
        loop._completed_signals = [_FakeSignal(signal_value=0.7, return_5d=-0.03)]
        p = OnlineLearningPipeline(
            inst_generator=MagicMock(), outcome_loop=loop,
            state_dir=tmp_path / "olp", n_trials=1, min_run_interval_hours=0,
        )
        _, y = p._build_dataset()
        assert y[0] == 0.0

    def test_empty_generator_signals_uses_signal_value(self, tmp_path):
        loop = MagicMock()
        loop._completed_signals = [
            _FakeSignal(signal_value=0.5, generator_signals={}, return_5d=0.01)
        ]
        p = OnlineLearningPipeline(
            inst_generator=MagicMock(), outcome_loop=loop,
            state_dir=tmp_path / "olp", n_trials=1, min_run_interval_hours=0,
        )
        X, y = p._build_dataset()
        assert X.shape[1] == 1  # single-feature fallback


# ---------------------------------------------------------------------------
# Count labeled
# ---------------------------------------------------------------------------

class TestCountLabeled:
    def test_counts_only_completed_with_return(self, tmp_path):
        loop = MagicMock()
        loop._completed_signals = [
            _FakeSignal(return_5d=0.01),
            _FakeSignal(return_5d=None),
            _FakeSignal(return_5d=-0.02),
        ]
        p = OnlineLearningPipeline(
            inst_generator=MagicMock(), outcome_loop=loop,
            state_dir=tmp_path / "olp", n_trials=1, min_run_interval_hours=0,
        )
        assert p._count_labeled() == 2


# ---------------------------------------------------------------------------
# Statistical gate
# ---------------------------------------------------------------------------

class TestStatSignificance:
    def test_high_wins_low_pvalue(self, tmp_path):
        p = _make_pipeline(tmp_path)
        # 80% accuracy vs null 55%: should be significant
        pv = p._stat_significance_test(wins=80, n=100, null_p=0.55)
        assert pv < 0.05

    def test_marginal_wins_high_pvalue(self, tmp_path):
        p = _make_pipeline(tmp_path)
        # 52 vs 50%: not significant at n=20
        pv = p._stat_significance_test(wins=11, n=20, null_p=0.50)
        assert pv > 0.05

    def test_zero_samples_returns_one(self, tmp_path):
        p = _make_pipeline(tmp_path)
        pv = p._stat_significance_test(wins=0, n=0, null_p=0.5)
        assert pv == 1.0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

class TestCooldown:
    @pytest.mark.asyncio
    async def test_cooldown_blocks_run(self, tmp_path):
        inst_gen = MagicMock()
        p = OnlineLearningPipeline(
            inst_generator=inst_gen,
            outcome_loop=_make_outcome_loop(),
            state_dir=tmp_path / "olp",
            n_trials=1,
            min_run_interval_hours=100.0,  # long cooldown
        )
        p._state.last_run_ts = time.time()
        launched = await p.maybe_run()
        assert launched is False

    @pytest.mark.asyncio
    async def test_no_cooldown_allows_run(self, tmp_path):
        p = _make_pipeline(tmp_path)  # already has min_run_interval_hours=0.0
        p._state.last_run_ts = 0.0
        if p._count_labeled() >= MIN_LABELED_SAMPLES:
            launched = await p.maybe_run()
            assert launched is True


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_and_reload(self, tmp_path):
        p = _make_pipeline(tmp_path)
        p._state.total_runs = 5
        p._state.total_promotions = 2
        p._state.champion_accuracy = 0.63
        p._save_state()

        # Load fresh instance from same dir
        p2 = _make_pipeline(tmp_path)
        assert p2._state.total_runs == 5
        assert p2._state.total_promotions == 2
        assert p2._state.champion_accuracy == pytest.approx(0.63)

    def test_corrupted_state_file_falls_back_to_fresh(self, tmp_path):
        state_path = tmp_path / "olp" / "pipeline_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("NOT_JSON", encoding="utf-8")
        p = _make_pipeline(tmp_path)
        assert p._state.total_runs == 0

    def test_runs_capped_at_50(self, tmp_path):
        p = _make_pipeline(tmp_path)
        for i in range(60):
            run = PipelineRun(
                run_id=f"r{i}", started_at="t", finished_at="t",
                n_samples=100, n_train=80, n_holdout=20,
                champion_accuracy=0.55, challenger_accuracy=0.58,
                accuracy_gain=0.03, p_value=0.02, promoted=False, reason="x"
            )
            p._state.runs.append(run)
        p._save_state()
        p2 = _make_pipeline(tmp_path)
        assert len(p2._state.runs) <= 50


# ---------------------------------------------------------------------------
# get_state API
# ---------------------------------------------------------------------------

class TestGetState:
    def test_returns_available_true(self, tmp_path):
        p = _make_pipeline(tmp_path)
        s = p.get_state()
        assert s["available"] is True

    def test_contains_required_keys(self, tmp_path):
        p = _make_pipeline(tmp_path)
        s = p.get_state()
        for k in ["champion_accuracy", "total_runs", "total_promotions", "recent_runs"]:
            assert k in s

    def test_recent_runs_at_most_5(self, tmp_path):
        p = _make_pipeline(tmp_path)
        for i in range(10):
            run = PipelineRun(
                run_id=f"r{i}", started_at="t", finished_at="t",
                n_samples=100, n_train=80, n_holdout=20,
                champion_accuracy=0.55, challenger_accuracy=0.58,
                accuracy_gain=0.03, p_value=0.02, promoted=False, reason="x"
            )
            p._state.runs.append(run)
        s = p.get_state()
        assert len(s["recent_runs"]) <= 5

    def test_no_runs_returns_empty_list(self, tmp_path):
        p = _make_pipeline(tmp_path)
        s = p.get_state()
        assert s["recent_runs"] == []


# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------

class TestPromotionDecision:
    def test_record_run_increments_total(self, tmp_path):
        p = _make_pipeline(tmp_path)
        run = PipelineRun(
            run_id="r1", started_at="t", finished_at="t",
            n_samples=100, n_train=80, n_holdout=20,
            champion_accuracy=0.55, challenger_accuracy=0.60,
            accuracy_gain=0.05, p_value=0.01, promoted=True, reason="ok"
        )
        p._record_run(run, promoted=True, challenger_acc=0.60, challenger_params={"n": 100})
        assert p._state.total_runs == 1
        assert p._state.total_promotions == 1

    def test_no_promotion_does_not_update_champion(self, tmp_path):
        p = _make_pipeline(tmp_path)
        p._state.champion_accuracy = 0.60
        run = PipelineRun(
            run_id="r1", started_at="t", finished_at="t",
            n_samples=100, n_train=80, n_holdout=20,
            champion_accuracy=0.60, challenger_accuracy=0.59,
            accuracy_gain=-0.01, p_value=0.8, promoted=False, reason="no gain"
        )
        p._record_run(run, promoted=False, challenger_acc=0.59, challenger_params={})
        assert p._state.champion_accuracy == pytest.approx(0.60)

    def test_promotion_updates_champion_accuracy(self, tmp_path):
        p = _make_pipeline(tmp_path)
        p._state.champion_accuracy = 0.55
        run = PipelineRun(
            run_id="r1", started_at="t", finished_at="t",
            n_samples=100, n_train=80, n_holdout=20,
            champion_accuracy=0.55, challenger_accuracy=0.65,
            accuracy_gain=0.10, p_value=0.001, promoted=True, reason="promoted"
        )
        p._record_run(run, promoted=True, challenger_acc=0.65, challenger_params={"lr": 0.01})
        assert p._state.champion_accuracy == pytest.approx(0.65)
        assert p._state.champion_params == {"lr": 0.01}
