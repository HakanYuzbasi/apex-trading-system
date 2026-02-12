"""Tests for governor walk-forward tuning."""

from datetime import datetime, timedelta

import pandas as pd

from backtesting.governor_walkforward import WalkForwardTuningConfig, tune_policy_for_key


def test_walkforward_tuner_generates_policy():
    rows = []
    start = datetime(2025, 1, 1)
    for i in range(180):
        ts = start + timedelta(days=i)
        rows.append(
            {
                "signal_time": ts,
                "symbol": "AAPL",
                "regime": "default",
                "signal_value": 0.35 + ((i % 5) * 0.02),
                "confidence": 0.45 + ((i % 7) * 0.03),
                "signal_direction": "BUY" if i % 3 else "SELL",
                "return_10d": 0.01 if i % 3 else -0.008,
            }
        )

    df = pd.DataFrame(rows)
    config = WalkForwardTuningConfig(cadence="weekly", min_samples_per_fold=5)
    policy = tune_policy_for_key(df, asset_class="EQUITY", regime="default", config=config)

    assert policy is not None
    assert policy.asset_class == "EQUITY"
    assert policy.regime == "default"
    assert "red" in policy.tier_controls
    assert policy.tier_controls["red"].halt_new_entries is True
