"""tests/test_exec_sim_wiring.py — Execution simulator pre-trade wiring tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.execution_simulator.service import simulate_execution, _fallback_simulate


# ── _fallback_simulate ────────────────────────────────────────────────────────

class TestFallbackSimulate:
    def test_returns_required_keys(self):
        r = _fallback_simulate("AAPL", 100, 150.0)
        assert "symbol" in r
        assert "quantity" in r
        assert "results" in r
        assert "best_algorithm" in r

    def test_symbol_preserved(self):
        r = _fallback_simulate("MSFT", 50, 200.0)
        assert r["symbol"] == "MSFT"

    def test_quantity_preserved(self):
        r = _fallback_simulate("TSLA", 25, 300.0)
        assert r["quantity"] == 25

    def test_notional_correct(self):
        r = _fallback_simulate("AAPL", 10, 100.0)
        assert abs(r["notional_usd"] - 1000.0) < 1e-6

    def test_best_algorithm_is_cheapest(self):
        r = _fallback_simulate("AAPL", 100, 150.0)
        best_name = r["best_algorithm"]
        best_cost = next(x["estimated_cost_bps"] for x in r["results"] if x["algorithm"] == best_name)
        all_costs = [x["estimated_cost_bps"] for x in r["results"]]
        assert best_cost == min(all_costs)

    def test_market_is_most_expensive(self):
        r = _fallback_simulate("AAPL", 100, 150.0)
        costs = {x["algorithm"]: x["estimated_cost_bps"] for x in r["results"]}
        assert costs.get("MARKET", 0) > costs.get("TWAP", 999)

    def test_three_algorithms_returned(self):
        r = _fallback_simulate("X", 10, 50.0)
        assert len(r["results"]) == 3

    def test_all_costs_positive(self):
        r = _fallback_simulate("AAPL", 100, 150.0)
        for algo in r["results"]:
            assert algo["estimated_cost_bps"] > 0

    def test_cost_usd_consistent_with_bps(self):
        r = _fallback_simulate("AAPL", 100, 200.0)
        for algo in r["results"]:
            expected = 100 * 200.0 * algo["estimated_cost_bps"] / 10000.0
            assert abs(algo["estimated_cost_usd"] - expected) < 0.01

    def test_larger_order_higher_cost_bps(self):
        r_small = _fallback_simulate("AAPL", 10, 100.0)
        r_large = _fallback_simulate("AAPL", 10_000, 100.0)
        small_market = next(x["estimated_cost_bps"] for x in r_small["results"] if x["algorithm"] == "MARKET")
        large_market = next(x["estimated_cost_bps"] for x in r_large["results"] if x["algorithm"] == "MARKET")
        assert large_market > small_market


# ── simulate_execution (fallback path) ───────────────────────────────────────

class TestSimulateExecution:
    def test_returns_dict(self):
        r = simulate_execution("AAPL", 100, reference_price=150.0)
        assert isinstance(r, dict)

    def test_has_results_list(self):
        r = simulate_execution("AAPL", 100, reference_price=150.0)
        assert isinstance(r["results"], list)
        assert len(r["results"]) >= 2

    def test_has_best_algorithm(self):
        r = simulate_execution("AAPL", 100, reference_price=150.0)
        assert isinstance(r["best_algorithm"], str)
        algos = {x["algorithm"] for x in r["results"]}
        assert r["best_algorithm"] in algos

    def test_side_is_buy(self):
        r = simulate_execution("AAPL", 100, reference_price=150.0)
        assert r["side"] == "BUY"

    def test_symbol_and_quantity_passed_through(self):
        r = simulate_execution("ETH/USD", 5, reference_price=2000.0)
        assert r["symbol"] == "ETH/USD"
        assert r["quantity"] == 5

    def test_urgency_param_accepted(self):
        # Should not raise regardless of urgency value
        r = simulate_execution("AAPL", 100, urgency="critical", reference_price=100.0)
        assert "results" in r

    def test_time_horizon_param_accepted(self):
        r = simulate_execution("AAPL", 100, time_horizon_minutes=60, reference_price=100.0)
        assert "results" in r

    def test_daily_volume_param_accepted(self):
        r = simulate_execution("AAPL", 100, daily_volume=5_000_000, reference_price=100.0)
        assert "results" in r


# ── Exec history ring-buffer behaviour ───────────────────────────────────────

class TestExecSimHistory:
    """Unit-test the history ring-buffer logic independently (no engine needed)."""

    def _make_record(self, symbol: str = "AAPL") -> dict:
        return {
            "symbol": symbol, "side": "BUY", "quantity": 100, "notional": 15000.0,
            "reference_price": 150.0, "best_algorithm": "TWAP",
            "estimated_cost_bps": 4.5, "timestamp": "2026-01-01T00:00:00+00:00",
        }

    def test_records_appended(self):
        history: list = []
        for i in range(5):
            history.append(self._make_record(f"SYM{i}"))
        assert len(history) == 5

    def test_ring_buffer_capped_at_100(self):
        history: list = []
        for i in range(150):
            history.append(self._make_record())
            if len(history) > 100:
                del history[:-100]
        assert len(history) == 100

    def test_most_recent_kept_after_trim(self):
        history: list = []
        for i in range(110):
            history.append(self._make_record(f"SYM{i:03d}"))
            if len(history) > 100:
                del history[:-100]
        # Last appended is SYM109
        assert history[-1]["symbol"] == "SYM109"

    def test_record_has_expected_fields(self):
        r = self._make_record()
        for field in ("symbol", "side", "quantity", "notional", "reference_price",
                      "best_algorithm", "estimated_cost_bps", "timestamp"):
            assert field in r
