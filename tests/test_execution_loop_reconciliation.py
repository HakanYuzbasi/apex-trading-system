from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from core.execution_loop import ApexTradingSystem


@pytest.mark.asyncio
async def test_refresh_pending_orders_collects_ibkr_and_alpaca() -> None:
    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.pending_orders = set()

    submitted_trade = SimpleNamespace(
        contract=SimpleNamespace(symbol="AAPL"),
        orderStatus=SimpleNamespace(status="Submitted"),
    )
    filled_trade = SimpleNamespace(
        contract=SimpleNamespace(symbol="MSFT"),
        orderStatus=SimpleNamespace(status="Filled"),
    )

    system.ibkr = SimpleNamespace(
        ib=SimpleNamespace(openTrades=lambda: [submitted_trade, filled_trade]),
    )
    system.alpaca = SimpleNamespace(get_open_orders=lambda: ["CRYPTO:ETH/USD"])

    await system.refresh_pending_orders()

    assert system.pending_orders == {"AAPL", "CRYPTO:ETH/USD"}


@pytest.mark.asyncio
async def test_reconcile_position_state_cleans_phantom_entries() -> None:
    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.positions = {"AAPL": 10, "CRYPTO:ETH/USD": 2}
    system.performance_attribution = SimpleNamespace(
        open_positions={"AAPL": {"qty": 10}, "MSFT": {"qty": 5}},
    )
    system.position_entry_prices = {"MSFT": 100.0, "AAPL": 50.0}
    system.position_entry_times = {"MSFT": datetime(2026, 1, 1), "AAPL": datetime(2026, 1, 2)}
    system.position_entry_signals = {"MSFT": 0.7, "AAPL": 0.5}
    system.position_peak_prices = {"MSFT": 105.0}
    system.position_stops = {"MSFT": 95.0}
    system.failed_exits = {"MSFT": 1}
    system._tp_tranches_taken = {"MSFT": {1}}
    save_calls: list[str] = []
    system._save_position_metadata = lambda: save_calls.append("saved")

    await system._reconcile_position_state()

    assert "MSFT" not in system.performance_attribution.open_positions
    assert "MSFT" not in system.position_entry_prices
    assert "MSFT" not in system.position_entry_times
    assert "MSFT" not in system.position_entry_signals
    assert "MSFT" not in system.position_peak_prices
    assert "MSFT" not in system.position_stops
    assert "MSFT" not in system.failed_exits
    assert "MSFT" not in system._tp_tranches_taken
    assert system.performance_attribution.open_positions["AAPL"]["qty"] == 10
    assert system.performance_attribution.open_positions["AAPL"]["symbol"] == "AAPL"
    assert system.performance_attribution.open_positions["AAPL"]["quantity"] == 10.0
    assert save_calls == ["saved"]


@pytest.mark.asyncio
async def test_reconcile_position_state_normalizes_crypto_keys_and_repairs_quantity() -> None:
    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.positions = {"CRYPTO:LINK/USD": 612.8948}
    save_calls: list[str] = []

    class Tracker:
        def __init__(self) -> None:
            self.open_positions = {
                "LINK/USD": {
                    "symbol": "LINK/USD",
                    "quantity": 205.0,
                    "entry_price": 8.97,
                }
            }

        def normalize_open_positions(self) -> bool:
            self.open_positions = {
                "CRYPTO:LINK/USD": {
                    "symbol": "CRYPTO:LINK/USD",
                    "quantity": 205.0,
                    "entry_price": 8.97,
                }
            }
            return True

        def _save_state(self) -> None:
            save_calls.append("saved")

    system.performance_attribution = Tracker()
    system.position_entry_prices = {}
    system.position_entry_times = {}
    system.position_entry_signals = {}
    system.position_peak_prices = {}
    system.position_stops = {}
    system.failed_exits = {}
    system._tp_tranches_taken = {}
    system._save_position_metadata = lambda: save_calls.append("meta")

    await system._reconcile_position_state()

    repaired = system.performance_attribution.open_positions["CRYPTO:LINK/USD"]
    assert repaired["symbol"] == "CRYPTO:LINK/USD"
    assert repaired["quantity"] == pytest.approx(612.8948)
    assert save_calls == ["saved"]


def test_seed_attribution_for_open_positions_purges_stale_and_seeds_missing() -> None:
    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.positions = {"CRYPTO:ETH/USD": 2, "AAPL": 5, "ZERO": 0}
    system.performance_attribution = SimpleNamespace(
        open_positions={
            "ETH/USD": {"qty": 2},
            "CRYPTO:STALE/USD": {"qty": 1},
        },
    )
    system.position_entry_prices = {"AAPL": 123.0}
    system.position_entry_times = {"AAPL": datetime(2026, 2, 1, 9, 30)}
    system.position_entry_signals = {"AAPL": 0.62}
    system.price_cache = {}
    system.historical_data = {}
    system._current_regime = "risk_on"
    system._performance_snapshot = SimpleNamespace(tier=SimpleNamespace(value="green"))
    system._risk_multiplier = 1.0
    system._vix_risk_multiplier = 0.9
    system._map_governor_regime = lambda asset_class, regime: f"{asset_class}:{regime}"

    recorded: list[dict] = []
    system._record_entry_attribution = lambda **kwargs: recorded.append(kwargs)

    system._seed_attribution_for_open_positions()

    assert "CRYPTO:STALE/USD" not in system.performance_attribution.open_positions
    assert len(recorded) == 1
    assert recorded[0]["symbol"] == "AAPL"
    assert recorded[0]["asset_class"] == "EQUITY"
    assert recorded[0]["side"] == "LONG"
    assert recorded[0]["quantity"] == 5.0
    assert recorded[0]["entry_price"] == 123.0
    assert recorded[0]["entry_signal"] == 0.62
