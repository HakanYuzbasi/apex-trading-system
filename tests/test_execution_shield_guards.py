from execution.execution_shield import ExecutionAlgo, ExecutionShield


def test_spread_gate_blocks_wide_spread():
    shield = ExecutionShield(max_spread_bps=10.0)
    allowed, reason, spread_bps = shield.check_spread_gate(
        symbol="AAPL",
        bid=100.0,
        ask=100.3,  # ~30 bps
    )
    assert not allowed
    assert "spread_gate_blocked" in reason
    assert spread_bps > 10.0


def test_spread_gate_allows_when_quote_unavailable():
    shield = ExecutionShield(max_spread_bps=10.0)
    allowed, reason, spread_bps = shield.check_spread_gate(
        symbol="AAPL",
        bid=0.0,
        ask=0.0,
    )
    assert allowed
    assert reason == "spread_unavailable"
    assert spread_bps == 0.0


def test_slippage_budget_blocks_after_budget_exhausted():
    shield = ExecutionShield(
        slippage_budget_bps=30.0,
        slippage_budget_window=3,
    )
    shield.record_execution(
        symbol="AAPL",
        expected_price=100.0,
        fill_price=100.2,  # 20 bps
        shares=10,
        algo=ExecutionAlgo.MARKET,
    )
    shield.record_execution(
        symbol="AAPL",
        expected_price=100.0,
        fill_price=100.2,  # 20 bps
        shares=10,
        algo=ExecutionAlgo.MARKET,
    )
    allowed, reason = shield.check_slippage_budget("AAPL")
    assert not allowed
    assert "slippage_budget_blocked" in reason


def test_can_enter_order_combines_spread_and_budget_gates():
    shield = ExecutionShield(
        max_spread_bps=15.0,
        slippage_budget_bps=10.0,
        slippage_budget_window=2,
    )
    shield.record_execution(
        symbol="AAPL",
        expected_price=100.0,
        fill_price=100.11,  # 11 bps
        shares=10,
        algo=ExecutionAlgo.MARKET,
    )
    allowed, reason = shield.can_enter_order(
        symbol="AAPL",
        bid=100.0,
        ask=100.05,  # ~5 bps spread, spread gate passes
    )
    assert not allowed
    assert "slippage_budget_blocked" in reason


def test_can_enter_order_respects_budget_override():
    shield = ExecutionShield(
        slippage_budget_bps=200.0,
        slippage_budget_window=3,
    )
    shield.record_execution(
        symbol="AAPL",
        expected_price=100.0,
        fill_price=100.5,  # 50 bps
        shares=10,
        algo=ExecutionAlgo.MARKET,
    )
    allowed, reason = shield.can_enter_order(
        symbol="AAPL",
        bid=100.0,
        ask=100.01,
        slippage_budget_bps=40.0,
    )
    assert not allowed
    assert "slippage_budget_blocked" in reason


def test_edge_gate_blocks_low_expected_edge():
    shield = ExecutionShield(max_slippage_bps=12.0)
    allowed, reason = shield.can_enter_order(
        symbol="AAPL",
        bid=100.0,
        ask=100.04,  # ~4 bps spread
        signal_strength=0.12,
        confidence=0.45,
        min_edge_over_cost_bps=8.0,
        signal_to_edge_bps=60.0,
    )
    assert not allowed
    assert "edge_gate_blocked" in reason


def test_edge_gate_allows_high_signal_quality():
    shield = ExecutionShield(max_slippage_bps=12.0)
    allowed, reason = shield.can_enter_order(
        symbol="AAPL",
        bid=100.0,
        ask=100.02,  # ~2 bps spread
        signal_strength=0.72,
        confidence=0.9,
        min_edge_over_cost_bps=8.0,
        signal_to_edge_bps=100.0,
    )
    assert allowed
    assert reason == "ok"
