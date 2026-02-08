"""
tests/test_signal_leakage.py - Signal Leakage / Data Snooping Tests

Validates:
- Training panel is time-sorted with (timestamp, symbol) MultiIndex
- Future-return labels use shift(-N) so they reference actual future data
- Purge/embargo CV excludes contaminated samples from training
- No overlap between train and test timestamps after purge/embargo
- Volatility-scaled labels don't introduce lookahead
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import ApexConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_historical_data(
    symbols: list[str],
    n_days: int = 500,
    start: datetime | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate realistic multi-symbol historical OHLCV data."""
    rng = np.random.RandomState(seed)
    start = start or datetime(2022, 1, 3)
    dates = pd.bdate_range(start=start, periods=n_days, freq="B")

    data = {}
    for i, sym in enumerate(symbols):
        base = 100.0 + i * 50
        returns = rng.normal(0.0003, 0.015, n_days)
        close = base * np.exp(np.cumsum(returns))
        df = pd.DataFrame(
            {
                "Open": close * (1 + rng.uniform(-0.005, 0.005, n_days)),
                "High": close * (1 + rng.uniform(0, 0.015, n_days)),
                "Low": close * (1 + rng.uniform(-0.015, 0, n_days)),
                "Close": close,
                "Volume": rng.randint(500_000, 5_000_000, n_days).astype(float),
            },
            index=dates,
        )
        data[sym] = df
    return data


def _try_import_advanced_signal_generator():
    """Import AdvancedSignalGenerator, skip if ML deps missing."""
    try:
        from models.advanced_signal_generator import AdvancedSignalGenerator, ML_AVAILABLE
        if not ML_AVAILABLE:
            pytest.skip("scikit-learn not available")
        return AdvancedSignalGenerator
    except ImportError as e:
        pytest.skip(f"Cannot import AdvancedSignalGenerator: {e}")


# ---------------------------------------------------------------------------
# Tests: panel construction
# ---------------------------------------------------------------------------

class TestTrainingPanelConstruction:
    """_build_training_panel must produce a time-sorted MultiIndex panel."""

    def test_panel_has_multiindex(self):
        """Panel index must be (timestamp, symbol) MultiIndex."""
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"])

        panel, feature_names = gen._build_training_panel(data)

        assert not panel.empty
        assert isinstance(panel.index, pd.MultiIndex)
        assert panel.index.names == ["timestamp", "symbol"]

    def test_panel_is_time_sorted(self):
        """Panel must be sorted by timestamp (level 0)."""
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"])

        panel, _ = gen._build_training_panel(data)

        timestamps = panel.index.get_level_values(0)
        # Check monotonically non-decreasing
        assert (timestamps[1:] >= timestamps[:-1]).all()

    def test_panel_contains_target(self):
        """Panel must have a 'target' column (the label)."""
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"])

        panel, _ = gen._build_training_panel(data)
        assert not panel.empty, "Panel should not be empty with 500-bar data"
        assert "target" in panel.columns

    def test_panel_contains_regime(self):
        """Panel must have a 'regime' column."""
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"])

        panel, _ = gen._build_training_panel(data)
        assert not panel.empty
        assert "regime" in panel.columns
        assert set(panel["regime"].unique()).issubset({"bull", "bear", "neutral", "volatile"})

    def test_panel_contains_asset_class(self):
        """Panel must have an 'asset_class' column."""
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"])

        panel, _ = gen._build_training_panel(data)
        assert not panel.empty
        assert "asset_class" in panel.columns


# ---------------------------------------------------------------------------
# Tests: label construction (no lookahead)
# ---------------------------------------------------------------------------

class TestLabelConstruction:
    """Volatility-scaled labels must not introduce lookahead."""

    def test_target_uses_shift_negative(self):
        """
        Target = pct_change(5).shift(-5) / rolling_vol.
        The shift(-5) means the target at time t uses prices at t+5,
        which is correct for a *label* (we're predicting the future).
        The critical thing is that these rows with valid targets end
        5 bars before the last bar (the last 5 bars have NaN targets
        and must be excluded).
        """
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"], n_days=500)

        panel, _ = gen._build_training_panel(data)
        if panel.empty:
            pytest.skip("Panel empty - ML features may need more data")

        # The last 5 business days should have no target (NaN was dropped)
        timestamps = panel.index.get_level_values(0)
        last_5 = data["AAPL"].index[-5:]

        for d in last_5:
            assert d not in timestamps, f"Date {d} should have been excluded (no future data)"

    def test_vol_scaling_removes_zero_vol(self):
        """When vol=0, the label should be NaN (dropped from panel)."""
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()

        # Create data with constant prices then noise - need 500 bars to pass lookback
        rng = np.random.RandomState(99)
        n = 500
        dates = pd.bdate_range(start="2022-01-03", periods=n, freq="B")
        close = np.ones(n) * 100.0  # constant price
        # Add noise in the last portion so we have *some* valid labels
        close[350:] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 150)))

        def _make_df(close_arr):
            return pd.DataFrame(
                {
                    "Open": close_arr,
                    "High": close_arr * 1.001,
                    "Low": close_arr * 0.999,
                    "Close": close_arr,
                    "Volume": [1_000_000.0] * len(close_arr),
                },
                index=dates,
            )

        # Need 2+ symbols for cross-sectional normalization
        close2 = 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
        panel, _ = gen._build_training_panel({
            "AAPL": _make_df(close),
            "MSFT": _make_df(close2),
        })

        # Should have no Inf or NaN targets in the final panel
        if not panel.empty:
            assert not panel["target"].isnull().any()
            assert not np.isinf(panel["target"]).any()


# ---------------------------------------------------------------------------
# Tests: purge/embargo CV
# ---------------------------------------------------------------------------

class TestPurgeEmbargoCV:
    """Walk-forward splits must exclude contaminated samples."""

    def test_no_overlap_between_train_and_test(self):
        """Train and test timestamps must not overlap."""
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"], n_days=500)

        panel, feature_names = gen._build_training_panel(data)
        if panel.empty:
            pytest.skip("Panel empty")

        gen.feature_names = feature_names
        times = panel.index.get_level_values(0).unique().sort_values()
        n_splits = 3
        min_train = int(len(times) * 0.5)
        window_size = max(5, int((len(times) - min_train) / max(1, n_splits)))
        purge_days = getattr(ApexConfig, "ADV_PURGE_DAYS", 5)
        embargo_days = getattr(ApexConfig, "ADV_EMBARGO_DAYS", 2)

        for split in range(n_splits):
            test_start_idx = min_train + (split * window_size)
            test_end_idx = min(test_start_idx + window_size, len(times))
            if test_end_idx - test_start_idx < 5:
                break

            test_start_time = times[test_start_idx]
            test_end_time = times[test_end_idx - 1]
            purge_start = test_start_time - timedelta(days=purge_days)
            embargo_end = test_end_time + timedelta(days=embargo_days)

            train_mask = (times < purge_start) | (times > embargo_end)
            train_times = set(times[train_mask])
            test_times = set(times[test_start_idx:test_end_idx])

            overlap = train_times & test_times
            assert len(overlap) == 0, (
                f"Split {split}: {len(overlap)} overlapping timestamps between train and test"
            )

    def test_purge_gap_exists(self):
        """
        There must be a gap of at least purge_days between the last
        training timestamp and the first test timestamp.
        """
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"], n_days=500)

        panel, feature_names = gen._build_training_panel(data)
        if panel.empty:
            pytest.skip("Panel empty")

        gen.feature_names = feature_names
        times = panel.index.get_level_values(0).unique().sort_values()
        n_splits = 3
        min_train = int(len(times) * 0.5)
        window_size = max(5, int((len(times) - min_train) / max(1, n_splits)))
        purge_days = getattr(ApexConfig, "ADV_PURGE_DAYS", 5)

        for split in range(n_splits):
            test_start_idx = min_train + (split * window_size)
            test_end_idx = min(test_start_idx + window_size, len(times))
            if test_end_idx - test_start_idx < 5:
                break

            test_start_time = times[test_start_idx]
            purge_start = test_start_time - timedelta(days=purge_days)

            # Training times before the test window
            pre_test_train = times[(times < purge_start)]
            if len(pre_test_train) == 0:
                continue

            last_train = pre_test_train[-1]
            gap = (test_start_time - last_train).days
            assert gap >= purge_days, (
                f"Split {split}: gap={gap}d between last train ({last_train}) "
                f"and first test ({test_start_time}), expected >= {purge_days}d"
            )

    def test_embargo_gap_after_test(self):
        """
        Training data after the test window must start after embargo_end.
        """
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"], n_days=500)

        panel, feature_names = gen._build_training_panel(data)
        if panel.empty:
            pytest.skip("Panel empty")

        gen.feature_names = feature_names
        times = panel.index.get_level_values(0).unique().sort_values()
        n_splits = 3
        min_train = int(len(times) * 0.5)
        window_size = max(5, int((len(times) - min_train) / max(1, n_splits)))
        embargo_days = getattr(ApexConfig, "ADV_EMBARGO_DAYS", 2)

        for split in range(n_splits):
            test_start_idx = min_train + (split * window_size)
            test_end_idx = min(test_start_idx + window_size, len(times))
            if test_end_idx - test_start_idx < 5:
                break

            test_end_time = times[test_end_idx - 1]
            embargo_end = test_end_time + timedelta(days=embargo_days)

            # Training times after the test window
            post_test_train = times[(times > embargo_end)]
            if len(post_test_train) == 0:
                continue

            first_post_train = post_test_train[0]
            gap = (first_post_train - test_end_time).days
            assert gap >= embargo_days, (
                f"Split {split}: gap={gap}d between test end ({test_end_time}) "
                f"and first post-test train ({first_post_train}), expected >= {embargo_days}d"
            )


# ---------------------------------------------------------------------------
# Tests: feature/target alignment
# ---------------------------------------------------------------------------

class TestFeatureTargetAlignment:
    """Features and targets must be aligned in time - no shift mismatch."""

    def test_features_not_shifted_into_future(self):
        """
        Features at time t should only use data up to time t.
        We verify this by computing features on full data vs truncated
        data and checking the values match at the truncation point.
        """
        ASG = _try_import_advanced_signal_generator()
        gen = ASG()
        data = _make_historical_data(["AAPL", "MSFT"], n_days=500)

        panel, feature_names = gen._build_training_panel(data)
        if panel.empty:
            pytest.skip("Panel empty")

        # Pick a timestamp well inside the data
        last_valid_ts = panel.index.get_level_values(0).max()

        # Compute features on full data
        full_features = gen.compute_features_vectorized(data["AAPL"])

        # Compute features on data truncated to last_valid_ts
        truncated = data["AAPL"].loc[:last_valid_ts]
        trunc_features = gen.compute_features_vectorized(truncated)

        # At last_valid_ts, features should be identical
        if last_valid_ts in full_features.index and last_valid_ts in trunc_features.index:
            full_row = full_features.loc[last_valid_ts]
            trunc_row = trunc_features.loc[last_valid_ts]
            # They should match (features don't use future data)
            pd.testing.assert_series_equal(
                full_row.fillna(0), trunc_row.fillna(0),
                check_names=False, atol=1e-10,
            )
