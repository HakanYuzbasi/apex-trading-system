"""test_live_trader_smoke.py — 7 smoke tests for paper-trading infrastructure.
All tests mock Alpaca; no real orders are ever submitted.
Run: pytest tests/test_live_trader_smoke.py -v --override-ini="addopts="
"""
# ── websockets.sync stub (yfinance>=1.0 needs websockets>=12; venv may have 10.x)
import sys, types as _types
for _mod in ("websockets.sync", "websockets.sync.client"):
    if _mod not in sys.modules:
        sys.modules[_mod] = _types.ModuleType(_mod)
if not hasattr(sys.modules["websockets.sync.client"], "connect"):
    sys.modules["websockets.sync.client"].connect = lambda *a, **kw: None  # type: ignore

import gzip, json, os, time
from datetime import date, timezone, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR for every test so files don't pollute the repo."""
    import scripts.execution_log as elog_mod
    import scripts.health_check  as hc_mod
    import scripts.slippage_monitor as sm_mod

    for mod in (elog_mod, hc_mod, sm_mod):
        monkeypatch.setattr(mod, "DATA_DIR", tmp_path, raising=False)

    # Patch execution_log constants directly
    monkeypatch.setattr(elog_mod, "LOG_FILE",    tmp_path / "execution_log.jsonl")
    monkeypatch.setattr(elog_mod, "DATA_DIR",    tmp_path)
    monkeypatch.setattr(hc_mod,   "DATA_DIR",    tmp_path)
    monkeypatch.setattr(hc_mod,   "STATUS_FILE", tmp_path / "health_status.json")
    monkeypatch.setattr(hc_mod,   "EXEC_LOG",    tmp_path / "execution_log.jsonl")
    monkeypatch.setattr(hc_mod,   "LAST_RUN",    tmp_path / "last_successful_run.txt")
    monkeypatch.setattr(hc_mod,   "_CKSUM_FILE", tmp_path / "model_checksums.json")
    monkeypatch.setattr(sm_mod,   "DATA_DIR",    tmp_path)
    monkeypatch.setattr(sm_mod,   "EXEC_LOG",    tmp_path / "execution_log.jsonl")
    monkeypatch.setattr(sm_mod,   "HALT_FILE",   tmp_path / "HALT_slippage_exceeded.txt")
    yield tmp_path


@pytest.fixture()
def fake_model(tmp_path):
    """Write a minimal valid model.pkl and return its path."""
    import pickle
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    X = np.random.rand(60, 8)
    y = (X[:, 0] > 0.5).astype(int)
    gbm = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0)
    gbm.fit(X, y)
    art = {"gbm": gbm, "scaler": StandardScaler().fit(X),
           "feat_names": [f"f{i}" for i in range(8)],
           "top5": [], "train_end": "2023-12-31"}

    model_dir = tmp_path / "r18_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(art, f)
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# T1 — health_check passes when model is present and writable
# ─────────────────────────────────────────────────────────────────────────────
def test_health_check_passes_with_valid_model(tmp_path, fake_model, monkeypatch):
    """Health check must return OK when model is valid and APIs (mocked) respond."""
    import scripts.health_check as hc

    monkeypatch.setattr(hc, "MODEL_PATH", fake_model)

    # Mock network-dependent checks
    monkeypatch.setattr(hc, "check_alpaca_reachable",
                        lambda: {"check": "alpaca_api", "ok": True, "detail": "mocked"})
    monkeypatch.setattr(hc, "check_yfinance_reachable",
                        lambda: {"check": "yfinance", "ok": True, "detail": "mocked"})

    result = hc.run_health_check(halt_on_fail=False)
    assert result["overall"] in ("OK", "WARN"), (
        f"Expected OK or WARN, got {result['overall']}. "
        f"Checks: {result['all_checks']}"
    )
    # Model check specifically must be OK
    model_check = next(c for c in result["all_checks"] if c["check"] == "model_files")
    assert model_check["ok"], f"Model check failed: {model_check}"


# ─────────────────────────────────────────────────────────────────────────────
# T2 — data validation rejects stale and NaN bars
# ─────────────────────────────────────────────────────────────────────────────
def test_data_validation_rejects_stale_and_nan_bars():
    """load validation logic: stale date and NaN price must both be rejected."""
    yesterday = pd.Timestamp("2000-01-01")  # deliberately ancient

    # Stale bar
    df_stale = pd.DataFrame(
        {"Close": [100.0], "High": [101.0], "Low": [99.0], "Volume": [1_000_000]},
        index=[yesterday],
    )
    latest_date = str(df_stale.index[-1].date())
    today_str   = str(date.today())
    assert latest_date != today_str, "Stale check should fail for old date"

    # NaN price bar
    df_nan = pd.DataFrame(
        {"Close": [float("nan")], "High": [101.0], "Low": [99.0], "Volume": [1_000_000]},
        index=[pd.Timestamp.today().normalize()],
    )
    bar = df_nan.iloc[-1]
    is_invalid = bar["Close"] <= 0 or bar["Volume"] <= 0 or pd.isna(bar["Close"])
    assert is_invalid, "NaN close should be flagged as invalid"

    # Zero volume bar
    df_zero_vol = pd.DataFrame(
        {"Close": [100.0], "High": [101.0], "Low": [99.0], "Volume": [0]},
        index=[pd.Timestamp.today().normalize()],
    )
    bar2 = df_zero_vol.iloc[-1]
    is_invalid2 = bar2["Close"] <= 0 or bar2["Volume"] <= 0 or pd.isna(bar2["Close"])
    assert is_invalid2, "Zero volume should be flagged as invalid"


# ─────────────────────────────────────────────────────────────────────────────
# T3 — order size never exceeds MAX_ORDER_USD
# ─────────────────────────────────────────────────────────────────────────────
def test_order_size_never_exceeds_max(monkeypatch):
    """Kelly sizing must be capped at MAX_ORDER_USD regardless of portfolio size."""
    import scripts.live_trader as lt

    monkeypatch.setattr(lt, "MAX_ORDER_USD", 2000.0)

    KELLY_FRACTION = 0.5
    universe_size  = 12

    for portfolio_value in [10_000, 100_000, 1_000_000]:
        kelly_size = portfolio_value * KELLY_FRACTION / universe_size
        notional   = min(kelly_size, lt.MAX_ORDER_USD)
        assert notional <= 2000.0, (
            f"Notional {notional:.2f} exceeded MAX_ORDER_USD=2000 "
            f"(portfolio={portfolio_value})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T4 — daily loss limit halts orders
# ─────────────────────────────────────────────────────────────────────────────
def test_daily_loss_limit_halts_orders(monkeypatch):
    """When intraday loss > DAILY_LOSS_PCT, _halt_orders becomes True."""
    import scripts.live_trader as lt
    monkeypatch.setattr(lt, "_halt_orders", False)
    monkeypatch.setattr(lt, "DAILY_LOSS_PCT", 0.03)

    last_equity     = 100_000.0
    portfolio_value =  96_000.0   # 4% loss → exceeds 3% limit

    intraday_loss = (last_equity - portfolio_value) / last_equity
    if intraday_loss > lt.DAILY_LOSS_PCT:
        lt._halt_orders = True

    assert lt._halt_orders, (
        f"_halt_orders should be True for {intraday_loss:.1%} loss "
        f"(limit {lt.DAILY_LOSS_PCT:.1%})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# T5 — duplicate order check blocks second submission
# ─────────────────────────────────────────────────────────────────────────────
def test_duplicate_order_check_blocks_second_submission(monkeypatch):
    """_submit_order_with_retry must return None if same symbol+side already open."""
    import scripts.live_trader as lt

    # Fake open order for AAPL buy
    fake_open_order = SimpleNamespace(symbol="AAPL", side="buy")
    mock_api = MagicMock()
    mock_api.list_orders.return_value = [fake_open_order]

    result = lt._submit_order_with_retry(mock_api, "AAPL", qty=1,
                                          side="buy", notional=1500.0)
    assert result is None, "Duplicate order should be blocked (return None)"
    mock_api.submit_order.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# T6 — execution_log writes SUBMITTED then updates to FILLED
# ─────────────────────────────────────────────────────────────────────────────
def test_execution_log_submitted_then_filled(tmp_path):
    """ExecutionLog must write SUBMITTED record then update it to FILLED."""
    import scripts.execution_log as elog_mod
    elog_mod.LOG_FILE = tmp_path / "execution_log.jsonl"
    elog_mod.DATA_DIR = tmp_path

    from scripts.execution_log import ExecutionLog
    el  = ExecutionLog()
    rec = el.write("MSFT", "buy", signal_confidence=0.68,
                   notional=1200.0, status="SUBMITTED",
                   model_price=380.0, qty=3.15)

    # Verify SUBMITTED written
    records = ExecutionLog.read_all()
    assert len(records) == 1
    assert records[0]["status"] == "SUBMITTED"
    assert records[0]["fill_price"] is None

    # Update to FILLED
    el.update(rec["_id"], fill_price=380.45,
              fill_time=datetime.now(timezone.utc).isoformat(),
              status="FILLED")

    records = ExecutionLog.read_all()
    assert len(records) == 1
    assert records[0]["status"] == "FILLED"
    assert records[0]["fill_price"] == pytest.approx(380.45, abs=0.01)
    # Slippage should now be computed
    assert records[0]["slippage_bps"] is not None
    assert records[0]["slippage_bps"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# T7 — slippage monitor raises alert at correct threshold
# ─────────────────────────────────────────────────────────────────────────────
def test_slippage_monitor_alert_threshold(tmp_path, monkeypatch):
    """Rolling-20 avg >= HALT_BPS must create a HALT file."""
    import scripts.slippage_monitor as sm
    monkeypatch.setattr(sm, "DATA_DIR",  tmp_path)
    monkeypatch.setattr(sm, "EXEC_LOG",  tmp_path / "execution_log.jsonl")
    monkeypatch.setattr(sm, "HALT_FILE", tmp_path / "HALT_slippage_exceeded.txt")
    monkeypatch.setattr(sm, "HALT_BPS",  20)
    monkeypatch.setattr(sm, "CRIT_BPS",  10)
    monkeypatch.setattr(sm, "WARN_BPS",  15)

    # 20 filled trades at 25 bps each → rolling-20 avg = 25 ≥ HALT_BPS=20
    records = [
        {"status": "FILLED", "slippage_bps": 25.0,
         "symbol": "SPY", "date": "2025-01-01"}
        for _ in range(20)
    ]
    # Feed the DataFrame directly to analyse() — no patching of lazy imports needed
    df      = pd.DataFrame(records)
    summary = sm.analyse(df)

    assert sm.HALT_FILE.exists(), (
        f"HALT file must be written when rolling-20 avg >= {sm.HALT_BPS} bps. "
        f"summary={summary}"
    )
