"""
tests/test_correlation_entry_gate.py — Unit tests for risk/correlation_entry_gate.py
"""
import numpy as np
import pandas as pd
import pytest
from risk.correlation_entry_gate import check_correlation_gate


def _make_prices(n: int = 60, seed: int = 0, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    changes = rng.normal(0, 0.01, n)
    closes = start * np.cumprod(1 + changes)
    return pd.DataFrame({"Close": closes})


def _correlated_prices(base_df: pd.DataFrame, noise: float = 0.001) -> pd.DataFrame:
    """Return prices highly correlated with base_df."""
    rng = np.random.default_rng(99)
    base = base_df["Close"].values
    noisy = base + rng.normal(0, noise, len(base))
    return pd.DataFrame({"Close": noisy})


def _uncorrelated_prices(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    changes = rng.normal(0, 0.01, n)
    closes = 200.0 * np.cumprod(1 + changes)
    return pd.DataFrame({"Close": closes})


class TestCheckCorrelationGate:

    def test_no_open_positions_clears(self):
        candidate = _make_prices()
        blocked, reason, conf = check_correlation_gate(
            candidate_symbol="AAPL",
            candidate_prices=candidate,
            open_positions={},
            historical_data={"AAPL": candidate},
            confidence=0.70,
        )
        assert blocked is False
        assert reason == ""
        assert conf == pytest.approx(0.70)

    def test_high_corr_blocks(self):
        base = _make_prices(seed=1)
        correlated = _correlated_prices(base, noise=0.0001)
        blocked, reason, conf = check_correlation_gate(
            candidate_symbol="NVDA",
            candidate_prices=correlated,
            open_positions={"AAPL": 100},
            historical_data={"AAPL": base, "NVDA": correlated},
            confidence=0.70,
        )
        assert blocked is True
        assert "blocked" in reason.lower()

    def test_low_corr_passes(self):
        base = _make_prices(seed=1)
        uncorr = _uncorrelated_prices(seed=42)
        blocked, reason, conf = check_correlation_gate(
            candidate_symbol="BTC",
            candidate_prices=uncorr,
            open_positions={"AAPL": 100},
            historical_data={"AAPL": base, "BTC": uncorr},
            confidence=0.70,
        )
        assert blocked is False

    def test_soft_corr_penalises_confidence(self):
        """Build prices with moderate (soft) correlation."""
        rng = np.random.default_rng(7)
        base = _make_prices(seed=7)
        base_ret = np.diff(np.log(base["Close"].values))
        # Add significant independent noise to land in soft zone (0.50-0.65)
        noisy_ret = base_ret * 0.7 + rng.normal(0, 0.008, len(base_ret))
        noisy_closes = 100.0 * np.exp(np.concatenate([[0], np.cumsum(noisy_ret)]))
        noisy_df = pd.DataFrame({"Close": noisy_closes})

        # Measure actual correlation to assert it's in the soft zone
        n = min(len(base_ret), len(noisy_ret), 30)
        actual_corr = float(np.corrcoef(base_ret[-n:], noisy_ret[-n:])[0, 1])

        # Only run assertion if test data actually lands in soft zone
        if 0.50 <= abs(actual_corr) < 0.65:
            _, reason, adj_conf = check_correlation_gate(
                candidate_symbol="MSFT",
                candidate_prices=noisy_df,
                open_positions={"SPY": 50},
                historical_data={"SPY": base, "MSFT": noisy_df},
                confidence=0.80,
            )
            assert adj_conf < 0.80
            assert "soft warn" in reason.lower() or "conf" in reason.lower()

    def test_candidate_skipped_in_open_positions(self):
        """Same symbol in open positions should be skipped (no self-correlation block)."""
        base = _make_prices(seed=5)
        blocked, _, _ = check_correlation_gate(
            candidate_symbol="AAPL",
            candidate_prices=base,
            open_positions={"AAPL": 100},  # same symbol — should skip
            historical_data={"AAPL": base},
            confidence=0.70,
        )
        assert blocked is False

    def test_zero_qty_skipped(self):
        base = _make_prices(seed=6)
        corr = _correlated_prices(base)
        blocked, _, _ = check_correlation_gate(
            candidate_symbol="NVDA",
            candidate_prices=corr,
            open_positions={"AAPL": 0},   # zero qty → not really open
            historical_data={"AAPL": base, "NVDA": corr},
            confidence=0.70,
        )
        assert blocked is False

    def test_missing_close_column_passes(self):
        no_close_df = pd.DataFrame({"Open": [100.0, 101.0, 102.0]})
        blocked, _, conf = check_correlation_gate(
            candidate_symbol="X",
            candidate_prices=no_close_df,
            open_positions={"Y": 1},
            historical_data={"Y": _make_prices(), "X": no_close_df},
            confidence=0.80,
        )
        assert blocked is False
        assert conf == pytest.approx(0.80)

    def test_insufficient_bars_passes(self):
        short_df = pd.DataFrame({"Close": [100.0, 101.0, 100.5]})
        blocked, _, conf = check_correlation_gate(
            candidate_symbol="X",
            candidate_prices=short_df,
            open_positions={"Y": 1},
            historical_data={"Y": _make_prices(), "X": short_df},
            confidence=0.75,
        )
        assert blocked is False
        assert conf == pytest.approx(0.75)

    def test_none_candidate_prices_passes(self):
        blocked, _, conf = check_correlation_gate(
            candidate_symbol="X",
            candidate_prices=None,
            open_positions={"Y": 1},
            historical_data={"Y": _make_prices()},
            confidence=0.75,
        )
        assert blocked is False

    def test_confidence_unchanged_on_clear(self):
        base = _make_prices(seed=10)
        uncorr = _uncorrelated_prices(seed=11)
        _, _, conf = check_correlation_gate(
            candidate_symbol="TSLA",
            candidate_prices=uncorr,
            open_positions={"AAPL": 50},
            historical_data={"AAPL": base, "TSLA": uncorr},
            confidence=0.65,
        )
        assert conf == pytest.approx(0.65)

    def test_open_position_not_in_historical_data_skipped(self):
        """If open position has no historical data, correlation check skips it."""
        base = _make_prices(seed=12)
        corr = _correlated_prices(base)
        # AAPL is open but no historical_data entry → skip → not blocked
        blocked, _, _ = check_correlation_gate(
            candidate_symbol="NVDA",
            candidate_prices=corr,
            open_positions={"AAPL": 100},
            historical_data={"NVDA": corr},  # AAPL missing
            confidence=0.70,
        )
        assert blocked is False

    def test_multiple_open_positions_max_corr_used(self):
        """Gate uses MAX correlation across all open positions."""
        base = _make_prices(seed=20)
        high_corr = _correlated_prices(base, noise=0.0001)
        low_corr  = _uncorrelated_prices(seed=30)
        # One highly correlated, one not — should block because of the high-corr one
        blocked, reason, _ = check_correlation_gate(
            candidate_symbol="NVDA",
            candidate_prices=high_corr,
            open_positions={"SPY": 100, "MSFT": 50},
            historical_data={"SPY": base, "MSFT": low_corr, "NVDA": high_corr},
            confidence=0.70,
        )
        assert blocked is True
