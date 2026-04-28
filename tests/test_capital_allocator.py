"""tests/test_capital_allocator.py — CycleCapitalAllocator unit tests."""
import pytest
from quant_system.execution.capital_allocator import CycleCapitalAllocator, MIN_ORDER_USD


class TestSingleOrderFullApproval:
    def test_single_order_full_approval(self):
        alloc = CycleCapitalAllocator(available_cash=10_000.0, max_per_symbol=5_000.0)
        approved = alloc.request("BTC/USD", 2_000.0)
        assert approved == 2_000.0
        assert alloc.remaining() == 8_000.0
        assert alloc.committed() == {"BTC/USD": 2_000.0}

    def test_approved_exactly_equals_remaining_when_requested_less(self):
        alloc = CycleCapitalAllocator(available_cash=1_000.0, max_per_symbol=5_000.0)
        approved = alloc.request("ETH/USD", 999.0)
        assert approved == 999.0
        assert alloc.remaining() == pytest.approx(1.0)

    def test_zero_request_returns_zero(self):
        alloc = CycleCapitalAllocator(available_cash=10_000.0, max_per_symbol=5_000.0)
        assert alloc.request("SOL/USD", 0.0) == 0.0
        assert alloc.remaining() == 10_000.0

    def test_negative_request_returns_zero(self):
        alloc = CycleCapitalAllocator(available_cash=10_000.0, max_per_symbol=5_000.0)
        assert alloc.request("SOL/USD", -100.0) == 0.0


class TestSecondOrderGetsReducedBalance:
    def test_second_order_sees_reduced_balance(self):
        alloc = CycleCapitalAllocator(available_cash=3_000.0, max_per_symbol=5_000.0)
        alloc.request("BTC/USD", 2_000.0)
        approved2 = alloc.request("ETH/USD", 2_000.0)
        assert approved2 == 1_000.0
        assert alloc.remaining() == 0.0

    def test_order_skipped_when_remaining_below_min_order(self):
        alloc = CycleCapitalAllocator(available_cash=2_040.0, max_per_symbol=5_000.0)
        alloc.request("BTC/USD", 2_000.0)   # leaves 40 < MIN_ORDER_USD (50)
        approved = alloc.request("DOT/USD", 500.0)
        assert approved == 0.0
        assert alloc.remaining() == pytest.approx(40.0)   # not deducted


class TestOrderRejectedWhenRemainingBelowMin:
    def test_order_rejected_when_remaining_below_min(self):
        alloc = CycleCapitalAllocator(available_cash=30.0, max_per_symbol=5_000.0)
        approved = alloc.request("AVAX/USD", 500.0)
        assert approved == 0.0
        assert alloc.remaining() == 30.0

    def test_exactly_min_order_is_approved(self):
        alloc = CycleCapitalAllocator(
            available_cash=MIN_ORDER_USD, max_per_symbol=5_000.0
        )
        approved = alloc.request("LINK/USD", MIN_ORDER_USD)
        assert approved == MIN_ORDER_USD
        assert alloc.remaining() == 0.0


class TestMaxPerSymbolCapRespected:
    def test_max_per_symbol_caps_approved_amount(self):
        alloc = CycleCapitalAllocator(available_cash=10_000.0, max_per_symbol=1_500.0)
        approved = alloc.request("BTC/USD", 5_000.0)
        assert approved == 1_500.0
        assert alloc.remaining() == 8_500.0

    def test_max_per_symbol_takes_precedence_over_available(self):
        alloc = CycleCapitalAllocator(available_cash=500.0, max_per_symbol=200.0)
        approved = alloc.request("SOL/USD", 500.0)
        assert approved == 200.0

    def test_cumulative_per_symbol_in_committed(self):
        alloc = CycleCapitalAllocator(available_cash=10_000.0, max_per_symbol=3_000.0)
        alloc.request("BTC/USD", 2_000.0)
        alloc.request("BTC/USD", 1_000.0)
        assert alloc.committed()["BTC/USD"] == 3_000.0


class TestTenConcurrentRequestsNeverExceedTotal:
    def test_10_concurrent_requests_never_exceed_total(self):
        total = 25_000.0
        alloc = CycleCapitalAllocator(available_cash=total, max_per_symbol=5_000.0)
        symbols = [f"SYM{i}/USD" for i in range(10)]
        approved_sum = sum(alloc.request(sym, 3_000.0) for sym in symbols)
        assert approved_sum <= total
        assert alloc.remaining() >= 0.0
        assert abs(alloc.remaining() + approved_sum - total) < 1e-6

    def test_10_small_requests_all_approved_when_budget_sufficient(self):
        alloc = CycleCapitalAllocator(available_cash=10_000.0, max_per_symbol=5_000.0)
        results = [alloc.request(f"SYM{i}/USD", 500.0) for i in range(10)]
        assert all(r == 500.0 for r in results)
        assert alloc.remaining() == pytest.approx(5_000.0)

    def test_remaining_always_non_negative(self):
        alloc = CycleCapitalAllocator(available_cash=1_000.0, max_per_symbol=5_000.0)
        for i in range(20):
            alloc.request(f"SYM{i}/USD", 200.0)
        assert alloc.remaining() >= 0.0

    def test_constructor_rejects_negative_cash(self):
        with pytest.raises(ValueError, match="available_cash"):
            CycleCapitalAllocator(available_cash=-1.0, max_per_symbol=1_000.0)

    def test_constructor_rejects_zero_max_per_symbol(self):
        with pytest.raises(ValueError, match="max_per_symbol"):
            CycleCapitalAllocator(available_cash=1_000.0, max_per_symbol=0.0)
