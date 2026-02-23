"""Tests for Execution Simulator service functions."""

from services.execution_simulator.service import simulate_execution, _fallback_simulate
from services.execution_simulator.schemas import SimulateRequest, AlgoResult, SimulateResponse


class TestSimulateExecution:
    """Test the simulate_execution function."""

    def test_basic_simulation(self):
        """Simulate execution should return structured result."""
        result = simulate_execution(
            symbol="AAPL",
            quantity=1000,
            reference_price=150.0,
        )
        assert "symbol" in result
        assert result["symbol"] == "AAPL"
        assert "results" in result
        assert "best_algorithm" in result
        assert len(result["results"]) > 0

    def test_notional_calculation(self):
        """Notional USD should be quantity * reference_price."""
        result = simulate_execution(
            symbol="MSFT",
            quantity=500,
            reference_price=400.0,
        )
        assert result["notional_usd"] == 500 * 400.0

    def test_best_algorithm_is_cheapest(self):
        """Best algorithm should have lowest estimated cost."""
        result = simulate_execution(
            symbol="AAPL",
            quantity=5000,
            reference_price=150.0,
        )
        best_name = result["best_algorithm"]
        best_cost = None
        for r in result["results"]:
            if r["algorithm"] == best_name:
                best_cost = r["estimated_cost_bps"]
            else:
                assert best_cost is None or r["estimated_cost_bps"] >= best_cost or best_cost is not None

        # Verify best is indeed cheapest
        costs = [r["estimated_cost_bps"] for r in result["results"]]
        min_cost = min(costs)
        best_entry = next(r for r in result["results"] if r["algorithm"] == best_name)
        assert best_entry["estimated_cost_bps"] == min_cost

    def test_all_results_have_required_fields(self):
        """Each algo result should have expected fields."""
        result = simulate_execution(symbol="GOOGL", quantity=200, reference_price=170.0)
        for algo in result["results"]:
            assert "algorithm" in algo
            assert "estimated_cost_bps" in algo
            assert "estimated_cost_usd" in algo
            assert "time_to_complete_minutes" in algo
            assert "market_impact_bps" in algo

    def test_larger_order_higher_cost(self):
        """Larger orders should generally have higher cost in USD."""
        small = simulate_execution(symbol="AAPL", quantity=100, reference_price=150.0)
        large = simulate_execution(symbol="AAPL", quantity=10000, reference_price=150.0)

        small_total = sum(r["estimated_cost_usd"] for r in small["results"])
        large_total = sum(r["estimated_cost_usd"] for r in large["results"])
        assert large_total > small_total


class TestFallbackSimulate:
    """Test the fallback simulator when execution modules are unavailable."""

    def test_fallback_returns_results(self):
        """Fallback should return valid structure."""
        result = _fallback_simulate("AAPL", 1000, 150.0)
        assert result["symbol"] == "AAPL"
        assert result["quantity"] == 1000
        assert len(result["results"]) == 3
        assert result["best_algorithm"] in ("MARKET", "TWAP", "VWAP")

    def test_fallback_algos(self):
        """Fallback should include MARKET, TWAP, VWAP."""
        result = _fallback_simulate("MSFT", 500, 300.0)
        algo_names = {r["algorithm"] for r in result["results"]}
        assert algo_names == {"MARKET", "TWAP", "VWAP"}


class TestSimulatorSchemas:
    """Test Pydantic schemas for the simulator API."""

    def test_simulate_request_valid(self):
        """Valid SimulateRequest should parse."""
        req = SimulateRequest(symbol="AAPL", quantity=1000)
        assert req.symbol == "AAPL"
        assert req.urgency == "medium"

    def test_simulate_request_defaults(self):
        """Default urgency should be 'medium'."""
        req = SimulateRequest(symbol="MSFT", quantity=500)
        assert req.urgency == "medium"
        assert req.time_horizon_minutes is None

    def test_algo_result_schema(self):
        """AlgoResult should accept valid data."""
        ar = AlgoResult(
            algorithm="TWAP",
            estimated_cost_bps=5.2,
            estimated_cost_usd=78.0,
            time_to_complete_minutes=30.0,
            market_impact_bps=3.1,
        )
        assert ar.algorithm == "TWAP"

    def test_simulate_response_schema(self):
        """SimulateResponse should accept valid data."""
        resp = SimulateResponse(
            symbol="AAPL",
            quantity=1000,
            reference_price=150.0,
            notional_usd=150000.0,
            results=[
                AlgoResult(
                    algorithm="TWAP",
                    estimated_cost_bps=5.0,
                    estimated_cost_usd=75.0,
                    time_to_complete_minutes=30.0,
                    market_impact_bps=3.0,
                )
            ],
            best_algorithm="TWAP",
        )
        assert resp.best_algorithm == "TWAP"
