"""tests/test_param_optimizer.py — Optuna Bayesian parameter optimizer tests."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from monitoring.param_optimizer import (
    ParamOptimizer,
    TradeRecord,
    OptimizeResult,
    _compute_sharpe,
    _simulate_sharpe,
    _PARAM_SPACE,
    _HARD_BOUNDS,
    load_trade_records,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trades(n: int = 30, win_rate: float = 0.6) -> list:
    trades = []
    for i in range(n):
        pnl = 0.015 if (i % 10) < round(win_rate * 10) else -0.010
        trades.append(TradeRecord(
            signal=0.18,
            pnl_pct=pnl,
            hold_hours=4.0,
            regime="neutral",
            entry_confidence=0.65,
        ))
    return trades


# ── _compute_sharpe ───────────────────────────────────────────────────────────

class TestComputeSharpe:
    def test_positive_win_rate(self):
        pnls = [0.01] * 20 + [-0.005] * 5
        s = _compute_sharpe(pnls)
        assert s > 0

    def test_all_losses(self):
        pnls = [-0.01] * 20
        s = _compute_sharpe(pnls)
        assert s < 0

    def test_too_few_trades(self):
        assert _compute_sharpe([0.01, 0.02]) == 0.0

    def test_zero_std_handled(self):
        # All identical → std=0 → should not raise
        s = _compute_sharpe([0.01] * 10)
        assert math.isfinite(s)


# ── _simulate_sharpe ──────────────────────────────────────────────────────────

class TestSimulateSharpe:
    def _default_params(self):
        return {name: (lo + hi) / 2 for name, lo, hi, _ in _PARAM_SPACE}

    def test_reasonable_sharpe_with_good_trades(self):
        trades = _make_trades(50, win_rate=0.65)
        params = self._default_params()
        s = _simulate_sharpe(trades, params)
        assert s > 0

    def test_high_threshold_filters_all(self):
        """If threshold too high, no trades pass → returns -99."""
        trades = _make_trades(30, win_rate=0.60)
        params = self._default_params()
        params["threshold_bull"] = 0.99  # no trade has signal >= 0.99
        params["threshold_neutral"] = 0.99
        params["threshold_bear"] = 0.99
        s = _simulate_sharpe(trades, params)
        assert s == -99.0

    def test_strict_confidence_filters_trades(self):
        trades = _make_trades(30, win_rate=0.60)
        params = self._default_params()
        params["entry_confidence_moderate"] = 0.99  # block all
        s = _simulate_sharpe(trades, params)
        assert s == -99.0

    def test_loose_params_keeps_all_trades(self):
        trades = _make_trades(30, win_rate=0.70)
        params = self._default_params()
        params["threshold_bull"] = 0.01  # keep everything
        params["threshold_neutral"] = 0.01
        params["threshold_bear"] = 0.01
        params["entry_confidence_moderate"] = 0.01
        s = _simulate_sharpe(trades, params)
        assert s > 0


# ── ParamOptimizer ────────────────────────────────────────────────────────────

class TestParamOptimizer:
    def test_init_no_data_dir(self):
        with tempfile.TemporaryDirectory() as d:
            po = ParamOptimizer(data_dir=Path(d))
            assert po.get_best_params() == {}

    def test_run_study_too_few_trades(self):
        with tempfile.TemporaryDirectory() as d:
            po = ParamOptimizer(data_dir=Path(d))
            result = po.run_study(trades=_make_trades(5), n_trials=5)
            assert not result.improved
            assert result.n_trades == 5

    def test_run_study_returns_result(self):
        with tempfile.TemporaryDirectory() as d:
            po = ParamOptimizer(data_dir=Path(d))
            trades = _make_trades(40, win_rate=0.65)
            result = po.run_study(trades=trades, n_trials=5)
            assert isinstance(result, OptimizeResult)
            assert result.n_trades == 40
            assert result.n_trials > 0

    def test_params_within_hard_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            po = ParamOptimizer(data_dir=Path(d))
            trades = _make_trades(40, win_rate=0.70)
            result = po.run_study(trades=trades, n_trials=10)
            for name, value in result.best_params.items():
                lo, hi = _HARD_BOUNDS.get(name, (-1e9, 1e9))
                assert lo <= value <= hi, f"{name}={value} outside [{lo}, {hi}]"

    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            po1 = ParamOptimizer(data_dir=Path(d))
            # Manually inject best params and save
            po1._best_params = {"threshold_bull": 0.13}
            po1._save_state()

            po2 = ParamOptimizer(data_dir=Path(d))
            assert po2.get_best_params().get("threshold_bull") == 0.13

    def test_corrupt_state_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as d:
            state_path = Path(d) / "optimized_params.json"
            state_path.write_text("NOT JSON {{{")
            po = ParamOptimizer(data_dir=Path(d))
            # Should not raise; falls back to empty
            assert po.get_best_params() == {}

    def test_apply_best_params_no_params(self):
        with tempfile.TemporaryDirectory() as d:
            po = ParamOptimizer(data_dir=Path(d))
            assert not po.apply_best_params()  # nothing to apply

    def test_get_last_result_none_initially(self):
        with tempfile.TemporaryDirectory() as d:
            po = ParamOptimizer(data_dir=Path(d))
            assert po.get_last_result() is None

    def test_get_last_result_after_study(self):
        with tempfile.TemporaryDirectory() as d:
            po = ParamOptimizer(data_dir=Path(d))
            trades = _make_trades(30, win_rate=0.65)
            po.run_study(trades=trades, n_trials=3)
            r = po.get_last_result()
            assert r is not None
            assert "baseline_sharpe" in r
            assert "best_sharpe" in r
            assert "n_trials" in r


# ── OptimizeResult ────────────────────────────────────────────────────────────

class TestOptimizeResult:
    def test_to_dict_keys(self):
        r = OptimizeResult(
            improved=True,
            baseline_sharpe=0.5,
            best_sharpe=0.8,
            best_params={"threshold_bull": 0.13},
            n_trials=50,
            n_trades=40,
        )
        d = r.to_dict()
        assert d["improved"] is True
        assert d["improvement"] == round(0.8 - 0.5, 4)
        assert "ran_at" in d


# ── load_trade_records ────────────────────────────────────────────────────────

class TestLoadTradeRecords:
    def test_empty_dir_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as d:
            records = load_trade_records(Path(d), lookback_days=30)
            assert records == []

    def test_loads_exit_records(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as d:
            audit_dir = Path(d) / "users" / "admin" / "audit"
            audit_dir.mkdir(parents=True)
            ts = datetime.now(timezone.utc).isoformat()
            record = {
                "action": "EXIT",
                "timestamp": ts,
                "signal": 0.18,
                "pnl_pct": 0.012,
                "hold_hours": 4.0,
                "regime": "bull",
                "confidence": 0.65,
            }
            (audit_dir / "trade_audit_test.jsonl").write_text(
                json.dumps(record) + "\n"
            )
            records = load_trade_records(Path(d), lookback_days=30)
        assert len(records) == 1
        assert records[0].regime == "bull"
        assert records[0].signal == 0.18

    def test_ignores_entry_records(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as d:
            audit_dir = Path(d) / "users" / "admin" / "audit"
            audit_dir.mkdir(parents=True)
            ts = datetime.now(timezone.utc).isoformat()
            (audit_dir / "trade_audit_test.jsonl").write_text(
                json.dumps({"action": "ENTRY", "timestamp": ts}) + "\n"
            )
            records = load_trade_records(Path(d), lookback_days=30)
        assert len(records) == 0

    def test_ignores_old_records(self):
        with tempfile.TemporaryDirectory() as d:
            audit_dir = Path(d) / "users" / "admin" / "audit"
            audit_dir.mkdir(parents=True)
            old_ts = "2020-01-01T00:00:00+00:00"
            record = {"action": "EXIT", "timestamp": old_ts, "signal": 0.18,
                      "pnl_pct": 0.01, "regime": "bull", "confidence": 0.6}
            (audit_dir / "trade_audit_test.jsonl").write_text(
                json.dumps(record) + "\n"
            )
            records = load_trade_records(Path(d), lookback_days=30)
        assert len(records) == 0

    def test_param_space_coverage(self):
        """All param space entries have matching hard bounds."""
        for name, _, _, _ in _PARAM_SPACE:
            assert name in _HARD_BOUNDS, f"{name} missing from _HARD_BOUNDS"
