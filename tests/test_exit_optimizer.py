"""tests/test_exit_optimizer.py — Self-Calibrating Exit Optimizer tests."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from monitoring.exit_optimizer import (
    ExitOptimizer,
    _GBModel,
    _audit_rows_to_xy,
    _make_features,
    _regime_code,
    _vix_bucket,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_data(tmp_path):
    """Provides a temporary data directory."""
    return tmp_path


@pytest.fixture()
def optimizer(tmp_data):
    return ExitOptimizer(data_dir=tmp_data, score_threshold=0.65, min_samples=5)


def _make_exit_row(
    pnl_pct: float = 0.02,
    regime: str = "bull",
    hold_hours: float = 2.0,
    entry_signal: float = 0.15,
    exit_signal: float = 0.10,
    vix: float = 18.0,
    ts: str | None = None,
) -> dict:
    if ts is None:
        ts = "2025-01-15T10:00:00+00:00"
    return {
        "event_type": "EXIT",
        "symbol": "AAPL",
        "regime": regime,
        "hold_hours": hold_hours,
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
        "pnl_pct": pnl_pct,
        "vix": vix,
        "timestamp": ts,
    }


# ── _regime_code ──────────────────────────────────────────────────────────────

class TestRegimeCode:
    def test_known_regimes(self):
        assert _regime_code("bull") == 1
        assert _regime_code("neutral") == 2
        assert _regime_code("bear") == 3
        assert _regime_code("strong_bear") == 4
        assert _regime_code("volatile") == 5
        assert _regime_code("crisis") == 5
        assert _regime_code("strong_bull") == 0

    def test_unknown_falls_back(self):
        assert _regime_code("unknown_regime") == 2

    def test_case_insensitive(self):
        assert _regime_code("BULL") == 1
        assert _regime_code("Bear") == 3


# ── _vix_bucket ───────────────────────────────────────────────────────────────

class TestVixBucket:
    def test_low(self):
        assert _vix_bucket(12.0) == 0
        assert _vix_bucket(14.9) == 0

    def test_normal(self):
        assert _vix_bucket(15.0) == 1
        assert _vix_bucket(19.9) == 1

    def test_high(self):
        assert _vix_bucket(20.0) == 2
        assert _vix_bucket(29.9) == 2

    def test_extreme(self):
        assert _vix_bucket(30.0) == 3
        assert _vix_bucket(80.0) == 3


# ── _make_features ────────────────────────────────────────────────────────────

class TestMakeFeatures:
    def test_shape(self):
        feat = _make_features(2.0, "bull", 18.0, 0.15, 0.12, 0.01, 2, 10)
        assert feat.shape == (8,)

    def test_decay_ratio(self):
        feat = _make_features(2.0, "neutral", 18.0, 0.20, 0.10, 0.0, 0, 9)
        # decay = current / entry = 0.10 / 0.20 = 0.5
        assert abs(feat[4] - 0.5) < 1e-9

    def test_zero_entry_signal_no_div_zero(self):
        # entry_signal=0 → uses 1e-6 denominator — should not raise
        feat = _make_features(1.0, "bull", 15.0, 0.0, 0.05, 0.0, 1, 11)
        assert np.isfinite(feat).all()

    def test_dtype_float(self):
        feat = _make_features(3.0, "bear", 25.0, 0.12, 0.09, -0.01, 3, 14)
        assert feat.dtype == float


# ── _audit_rows_to_xy ─────────────────────────────────────────────────────────

class TestAuditRowsToXy:
    def test_empty_returns_none(self):
        X, y = _audit_rows_to_xy([])
        assert X is None and y is None

    def test_insufficient_rows_returns_none(self):
        rows = [_make_exit_row(pnl_pct=0.01) for _ in range(5)]
        X, y = _audit_rows_to_xy(rows)
        assert X is None and y is None

    def test_good_exit_label_1(self):
        rows = [_make_exit_row(pnl_pct=0.03) for _ in range(40)]
        X, y = _audit_rows_to_xy(rows)
        assert y is not None
        assert (y == 1).all()

    def test_bad_exit_label_0(self):
        rows = [_make_exit_row(pnl_pct=-0.03) for _ in range(40)]
        X, y = _audit_rows_to_xy(rows)
        assert y is not None
        assert (y == 0).all()

    def test_neutral_zone_skipped(self):
        # pnl in (-0.02, 0) → neutral → skipped
        rows = [_make_exit_row(pnl_pct=-0.01) for _ in range(40)]
        X, y = _audit_rows_to_xy(rows)
        # All rows land in neutral zone → insufficient
        assert X is None and y is None

    def test_mixed_labels(self):
        good_rows = [_make_exit_row(pnl_pct=0.02) for _ in range(20)]
        bad_rows  = [_make_exit_row(pnl_pct=-0.05) for _ in range(20)]
        X, y = _audit_rows_to_xy(good_rows + bad_rows)
        assert X is not None
        assert set(y.tolist()) == {0, 1}

    def test_feature_shape(self):
        rows = [_make_exit_row(pnl_pct=0.02 if i % 2 == 0 else -0.03)
                for i in range(40)]
        X, y = _audit_rows_to_xy(rows)
        assert X is not None
        assert X.shape[1] == 8


# ── _GBModel ──────────────────────────────────────────────────────────────────

class TestGBModel:
    def _make_data(self, n=100):
        np.random.seed(42)
        X = np.random.randn(n, 8)
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_unfitted_returns_05(self):
        m = _GBModel()
        x = np.zeros(8)
        assert m.predict_proba_pos(x) == 0.5

    def test_fit_and_predict_range(self):
        X, y = self._make_data()
        m = _GBModel()
        m.fit(X, y)
        for _ in range(10):
            x = np.random.randn(8)
            p = m.predict_proba_pos(x)
            assert 0.0 <= p <= 1.0

    def test_logistic_fallback(self):
        """If sklearn is unavailable, should use logistic fallback without error."""
        with patch.dict(sys.modules, {"sklearn": None, "sklearn.ensemble": None}):
            # Force fallback path
            m = _GBModel()
            X, y = self._make_data(80)
            # Manually call fallback
            m._logistic_fit(X, y)
            p = m.predict_proba_pos(X[0])
            assert 0.0 <= p <= 1.0


# ── ExitOptimizer.get_exit_score ─────────────────────────────────────────────

class TestGetExitScore:
    def test_unfitted_returns_05(self, optimizer):
        score = optimizer.get_exit_score(
            "AAPL", "bull", 2.0, 0.01, 0.15, 0.12, 18.0
        )
        assert score == 0.5

    def test_returns_in_range(self, optimizer, tmp_data):
        """After fitting with synthetic data, score should be in [0, 1]."""
        # Build enough rows
        rows = (
            [_make_exit_row(pnl_pct=0.03) for _ in range(30)] +
            [_make_exit_row(pnl_pct=-0.04) for _ in range(30)]
        )
        for r in rows:
            optimizer.record_outcome(
                r["symbol"], r["regime"], r["hold_hours"],
                r["entry_signal"], r["exit_signal"], r["pnl_pct"],
            )
        optimizer.fit()
        score = optimizer.get_exit_score(
            "AAPL", "bull", 2.0, 0.01, 0.15, 0.12, 18.0
        )
        assert 0.0 <= score <= 1.0

    def test_error_returns_05(self, optimizer):
        """Feature calculation error should return 0.5 gracefully."""
        optimizer._fitted = True
        optimizer._model  = MagicMock()
        optimizer._model.predict_proba_pos.side_effect = Exception("boom")
        score = optimizer.get_exit_score(
            "AAPL", "bull", 2.0, 0.01, 0.15, 0.12, 18.0
        )
        assert score == 0.5


# ── ExitOptimizer.should_exit ────────────────────────────────────────────────

class TestShouldExit:
    def test_unfitted_no_exit(self, optimizer):
        rec_exit, reason, score = optimizer.should_exit(
            "AAPL", "bull", 2.0, 0.01, 0.15, 0.12, 18.0
        )
        assert rec_exit is False
        assert score == 0.5

    def test_high_score_triggers_exit(self, optimizer):
        optimizer._fitted = True
        optimizer._model  = MagicMock()
        optimizer._model.predict_proba_pos = MagicMock(return_value=0.80)
        rec_exit, reason, score = optimizer.should_exit(
            "AAPL", "bull", 2.0, 0.01, 0.15, 0.12, 18.0
        )
        assert rec_exit is True
        assert "exit_optimizer_score" in reason
        assert abs(score - 0.80) < 1e-9

    def test_low_score_no_exit(self, optimizer):
        optimizer._fitted = True
        optimizer._model  = MagicMock()
        optimizer._model.predict_proba_pos = MagicMock(return_value=0.40)
        rec_exit, reason, score = optimizer.should_exit(
            "AAPL", "bull", 2.0, 0.01, 0.15, 0.12, 18.0
        )
        assert rec_exit is False
        assert reason == ""


# ── ExitOptimizer.record_outcome ─────────────────────────────────────────────

class TestRecordOutcome:
    def test_buffer_grows(self, optimizer):
        assert len(optimizer._buffer) == 0
        optimizer.record_outcome("AAPL", "bull", 2.0, 0.15, 0.10, 0.02)
        assert len(optimizer._buffer) == 1

    def test_buffer_bounded(self, optimizer):
        for _ in range(2100):
            optimizer.record_outcome("AAPL", "bull", 1.0, 0.15, 0.10, 0.01)
        assert len(optimizer._buffer) <= 2000

    def test_record_contains_exit_event_type(self, optimizer):
        optimizer.record_outcome("TSLA", "bear", 3.0, 0.12, 0.08, -0.03)
        assert optimizer._buffer[-1]["event_type"] == "EXIT"
        assert optimizer._buffer[-1]["symbol"] == "TSLA"
        assert optimizer._buffer[-1]["pnl_pct"] == -0.03


# ── ExitOptimizer.fit ─────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_false_insufficient_data(self, optimizer):
        # Empty data → fit returns False
        result = optimizer.fit()
        assert result is False

    def test_fit_returns_true_with_enough_buffer(self, optimizer):
        # Seed buffer with enough data
        for i in range(60):
            pnl = 0.03 if i % 2 == 0 else -0.05
            optimizer.record_outcome("AAPL", "bull", float(i % 5 + 1),
                                     0.15, 0.10, pnl)
        result = optimizer.fit()
        assert result is True
        assert optimizer._fitted is True
        assert optimizer._n_train >= 30

    def test_fit_updates_last_fit_ts(self, optimizer):
        for i in range(60):
            pnl = 0.03 if i % 2 == 0 else -0.05
            optimizer.record_outcome("AAPL", "bull", 1.0, 0.15, 0.10, pnl)
        optimizer.fit()
        assert optimizer._last_fit_ts > 0.0

    def test_fit_saves_model_file(self, optimizer):
        tmp_data = optimizer._data_dir
        for i in range(60):
            pnl = 0.03 if i % 2 == 0 else -0.05
            optimizer.record_outcome("AAPL", "bull", 1.0, 0.15, 0.10, pnl)
        optimizer.fit()
        assert (tmp_data / "exit_optimizer_model.pkl").exists()


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def _fill_and_fit(self, opt):
        for i in range(60):
            pnl = 0.03 if i % 2 == 0 else -0.05
            opt.record_outcome("AAPL", "neutral", 1.0, 0.15, 0.10, pnl)
        opt.fit()

    def test_reload_restores_fitted(self, tmp_data):
        opt1 = ExitOptimizer(data_dir=tmp_data, min_samples=5)
        self._fill_and_fit(opt1)
        assert opt1._fitted

        opt2 = ExitOptimizer(data_dir=tmp_data, min_samples=5)
        assert opt2._fitted

    def test_reload_score_consistent(self, tmp_data):
        opt1 = ExitOptimizer(data_dir=tmp_data, min_samples=5)
        self._fill_and_fit(opt1)
        s1 = opt1.get_exit_score("AAPL", "bull", 2.0, 0.01, 0.15, 0.12, 18.0)

        opt2 = ExitOptimizer(data_dir=tmp_data, min_samples=5)
        s2 = opt2.get_exit_score("AAPL", "bull", 2.0, 0.01, 0.15, 0.12, 18.0)
        assert abs(s1 - s2) < 1e-9

    def test_missing_model_file_no_crash(self, tmp_data):
        # Create state file but no model file — should not crash
        state_path = tmp_data / "exit_optimizer_state.json"
        state_path.write_text(json.dumps({"fitted": True, "n_train": 50, "last_fit_ts": 0.0}))
        opt = ExitOptimizer(data_dir=tmp_data)
        assert not opt._fitted  # model file missing → unfitted


# ── get_status ────────────────────────────────────────────────────────────────

class TestGetStatus:
    def test_status_keys(self, optimizer):
        status = optimizer.get_status()
        assert "fitted" in status
        assert "n_train" in status
        assert "buffer_size" in status
        assert "last_fit_ts" in status
        assert "score_threshold" in status
        assert "model_type" in status

    def test_status_unfitted(self, optimizer):
        assert optimizer.get_status()["model_type"] == "none"
