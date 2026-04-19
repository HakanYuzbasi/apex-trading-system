"""
tests/test_integration_smoke.py — Round 8 / GAP-13B

Thin, fast integration smoke tests that exercise seven critical code paths
end-to-end (no mocks on the SUT, only lightweight fixtures that replace
network / IO boundaries).  Each test is intentionally narrow — it confirms
that the module loads, its public contract is honoured, and the observable
side-effects match expectations.  Deeper behavioural tests live in the
per-module files already in ``tests/``.

Critical paths exercised:

1. Config load + ``validate_startup_schema()`` returns an empty error list.
2. :class:`AdaptiveMetaController` persists a recorded outcome to disk and a
   freshly-constructed controller reloads the state.
3. :meth:`SignalAggregator.record_source_outcome` accumulates samples and
   refits per-source weights after ``ML_WEIGHT_UPDATE_BARS`` observations.
4. :func:`execution.cost_model.expected_cost_bps` returns a finite positive
   float for a normal-size US equity taker order.
5. :meth:`AdvancedBacktester.run_walk_forward` returns a dict with the
   contract-required ``folds`` and ``aggregate`` keys.
6. :class:`CircuitBreaker` trips after ``CIRCUIT_BREAKER_CONSECUTIVE_LOSSES``
   consecutive losing trades.
7. :func:`leakage_check` raises :class:`LabelLeakageError` when a feature
   column equals ``label.shift(-1)``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from config import ApexConfig, validate_startup_schema


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config load + schema validator
# ─────────────────────────────────────────────────────────────────────────────

def test_config_load_and_schema_validator() -> None:
    """Config loads cleanly and the schema validator finds no errors."""
    # A representative subset of Round 8 keys must resolve to concrete values.
    for key in (
        "CIRCUIT_BREAKER_CONSECUTIVE_LOSSES",
        "CIRCUIT_BREAKER_COOLDOWN_HOURS",
        "CIRCUIT_BREAKER_COOLDOWN_BARS",
        "CIRCUIT_BREAKER_AUTO_RESET",
        "LEAKAGE_MAX_FUTURE_SHIFT",
        "LEAKAGE_CORR_THRESHOLD",
    ):
        assert hasattr(ApexConfig, key), f"missing {key}"

    errors: List[str] = validate_startup_schema()
    assert errors == [], f"schema validation errors: {errors}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. AdaptiveMetaController persist/reload
# ─────────────────────────────────────────────────────────────────────────────

def test_adaptive_meta_controller_persist_and_reload(tmp_path: Path) -> None:
    """record_outcome persists to disk; a new controller reloads that state."""
    from risk.adaptive_meta_controller import (
        AdaptiveMetaController,
        TradeContext,
    )

    state_file = tmp_path / "meta_controller_state.json"
    ctrl_a = AdaptiveMetaController(persist_path=str(state_file))

    ctx = TradeContext(
        symbol="AAPL",
        signal=0.42,
        confidence=0.70,
        asset_class="EQUITY",
        regime="bull",
    )
    # _save_state fires every 5 outcomes — feed exactly 5 so the file
    # materialises deterministically.
    for i in range(5):
        ctrl_a.record_outcome(ctx, pnl_pct=0.01 + 0.001 * i)

    assert state_file.exists(), "controller did not persist state to disk"
    # The on-disk payload must round-trip as JSON and include an ``n_obs``
    # entry for the bucket we just exercised.
    payload: Dict[str, Any] = json.loads(state_file.read_text())
    assert "n_obs" in payload, f"missing 'n_obs' in persisted state: {payload}"
    assert "EQUITY:bull" in payload["n_obs"]
    assert int(payload["n_obs"]["EQUITY:bull"]) >= 5

    # Reload path: a freshly-constructed controller must pick the state up.
    ctrl_b = AdaptiveMetaController(persist_path=str(state_file))
    ctrl_b._load_state()
    assert ctrl_b._n_obs.get("EQUITY:bull", 0) >= 5


# ─────────────────────────────────────────────────────────────────────────────
# 3. SignalAggregator.record_source_outcome
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_aggregator_records_and_refits_weights() -> None:
    """
    Feeding ``ML_WEIGHT_UPDATE_BARS`` outcomes triggers the softmax weight
    refit so the internal ``_source_weight`` dict becomes non-empty.
    """
    from signals.signal_aggregator import SignalAggregator

    agg = SignalAggregator()
    n_bars = int(ApexConfig.ML_WEIGHT_UPDATE_BARS)
    assert n_bars > 0, "ML_WEIGHT_UPDATE_BARS must be positive"

    # Two sources with clearly different PnL distributions so softmax has
    # a non-trivial ordering to learn.
    for i in range(n_bars):
        agg.record_source_outcome("funding_rate", pnl_pct=0.005)
        agg.record_source_outcome("pattern", pnl_pct=-0.003)

    # Refit fires on len(buf) % update_bars == 0 AND len(buf) >= update_bars,
    # so after exactly ``n_bars`` outcomes per source the weights must exist.
    assert "funding_rate" in agg._source_weight
    assert "pattern" in agg._source_weight
    w_fr = agg._weight_for("funding_rate")
    w_pt = agg._weight_for("pattern")
    # Both must be finite, positive, and floored at ML_WEIGHT_FLOOR.
    floor = float(ApexConfig.ML_WEIGHT_FLOOR)
    assert np.isfinite(w_fr) and w_fr >= floor
    assert np.isfinite(w_pt) and w_pt >= floor
    # The profitable source must outweigh the losing one (temperature > 0).
    if float(ApexConfig.ML_WEIGHT_TEMPERATURE) > 0.0:
        assert w_fr > w_pt, (
            f"profitable source underweighted: funding_rate={w_fr} pattern={w_pt}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. cost_model.expected_cost_bps
# ─────────────────────────────────────────────────────────────────────────────

def test_expected_cost_bps_positive_float_for_equity_taker() -> None:
    """A $10k aggressive US-equity order returns a finite positive bps cost."""
    from execution.cost_model import expected_cost_bps

    cost = expected_cost_bps(
        asset_class="EQUITY",
        is_maker=False,
        notional_usd=10_000.0,
    )
    assert isinstance(cost, float)
    assert np.isfinite(cost)
    assert cost > 0.0, f"expected positive cost in bps, got {cost}"
    # Sanity upper bound — a normal-size equity taker should never cost
    # more than 1% (100 bps) round-trip by itself.
    assert cost < 100.0, f"implausibly high equity taker cost: {cost} bps"


# ─────────────────────────────────────────────────────────────────────────────
# 5. AdvancedBacktester.run_walk_forward contract
# ─────────────────────────────────────────────────────────────────────────────

def test_run_walk_forward_returns_folds_and_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    ``run_walk_forward`` returns ``{'folds': [...], 'aggregate': {...}}``.

    The inner ``run_backtest`` is monkey-patched to a trivial stub so the
    smoke test runs in milliseconds rather than minutes — we are only
    validating the walk-forward *contract*, not the backtest engine itself.
    """
    from backtesting.advanced_backtester import AdvancedBacktester

    bt = AdvancedBacktester(initial_capital=100_000.0)

    def _fake_run_backtest(self, **kwargs: Any) -> Dict[str, float]:
        return {
            "sharpe_ratio": 1.25,
            "total_return": 0.012,
            "max_drawdown": -0.03,
            "win_rate": 0.58,
            "profit_factor": 1.4,
            "total_trades": 10,
            "final_value": 101_200.0,
        }

    monkeypatch.setattr(
        AdvancedBacktester, "run_backtest", _fake_run_backtest, raising=True
    )

    # Cover enough calendar days for at least one fold regardless of the
    # configured WF_IS_BARS/WF_OOS_BARS/WF_STEP_BARS values.
    is_n = int(ApexConfig.WF_IS_BARS)
    oos_n = int(ApexConfig.WF_OOS_BARS)
    span_days = is_n + oos_n + 5

    idx = pd.date_range("2024-01-01", periods=span_days, freq="D")
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
            },
            index=idx,
        )
    }

    out = bt.run_walk_forward(
        data=data,
        signal_generator=None,  # stubbed run_backtest never dereferences it
        start_date=str(idx[0].date()),
        end_date=str(idx[-1].date()),
    )
    assert isinstance(out, dict)
    assert "folds" in out and "aggregate" in out
    assert isinstance(out["folds"], list)
    assert isinstance(out["aggregate"], dict)
    # At least one fold must have been produced given span_days > is_n + oos_n.
    assert out["aggregate"].get("folds_run", 0) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Circuit breaker halts after N consecutive losses
# ─────────────────────────────────────────────────────────────────────────────

def test_circuit_breaker_trips_after_consecutive_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Recording ``CIRCUIT_BREAKER_CONSECUTIVE_LOSSES`` losses whose magnitude
    exceeds ``CIRCUIT_BREAKER_MIN_LOSS_USD`` trips the breaker.
    """
    from risk.risk_session import CircuitBreaker

    # Force breaker enabled regardless of deployment env for this smoke test.
    monkeypatch.setattr(ApexConfig, "CIRCUIT_BREAKER_ENABLED", True, raising=False)

    breaker = CircuitBreaker()
    assert breaker.is_tripped is False

    n_losses = int(ApexConfig.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES)
    min_loss = float(ApexConfig.CIRCUIT_BREAKER_MIN_LOSS_USD)
    loss_amount = -(min_loss + 10.0)  # clearly below the micro-loss threshold

    for _ in range(n_losses):
        breaker.record_trade(loss_amount)

    assert breaker.is_tripped is True
    assert breaker.consecutive_losses >= n_losses
    assert breaker.trip_reason is not None
    assert breaker.bars_since_trip == 0

    # tick_bar() advances the bar-cooldown counter while tripped and does
    # not auto-reset until ``CIRCUIT_BREAKER_COOLDOWN_BARS`` has elapsed.
    cooldown_bars = int(ApexConfig.CIRCUIT_BREAKER_COOLDOWN_BARS)
    if cooldown_bars > 1:
        assert breaker.tick_bar() is False
        assert breaker.bars_since_trip == 1
        assert breaker.is_tripped is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. leakage_check raises on forward-shifted feature
# ─────────────────────────────────────────────────────────────────────────────

def test_leakage_check_raises_on_forward_shifted_feature() -> None:
    """
    A feature column identical to ``label.shift(-1)`` leaks the next-bar
    label into training — ``leakage_check`` must raise ``LabelLeakageError``.
    """
    from models.ml_validator import LabelLeakageError, leakage_check

    rng = np.random.default_rng(seed=42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    label = pd.Series(rng.normal(size=n), index=idx)
    clean_feature = pd.Series(rng.normal(size=n), index=idx)
    # The leaky column literally equals the next-bar label — a textbook
    # look-ahead bug.
    leaky_feature = label.shift(-1)

    df = pd.DataFrame(
        {
            "clean": clean_feature,
            "leak": leaky_feature,
            "target": label,
        },
        index=idx,
    )

    with pytest.raises(LabelLeakageError):
        leakage_check(
            df,
            label_col="target",
            feature_cols=["clean", "leak"],
            max_future_shift=3,
            leak_corr_threshold=0.98,
            raise_on_fail=True,
        )

    # Non-raising mode must still flag the leaky column explicitly.
    report = leakage_check(
        df,
        label_col="target",
        feature_cols=["clean", "leak"],
        max_future_shift=3,
        leak_corr_threshold=0.98,
        raise_on_fail=False,
    )
    assert report["ok"] is False
    assert "leak" in report["leaky_features"]
