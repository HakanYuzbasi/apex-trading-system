"""Focused tests for execution-loop trust journaling helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.execution_loop import ApexTradingSystem


class _DummyEventStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def dispatch(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


def _build_system(tmp_path: Path) -> ApexTradingSystem:
    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.event_store = _DummyEventStore()
    system.positions = {"AAPL": 0}
    system.user_data_dir = tmp_path
    system._current_regime = "neutral"
    return system


def test_journal_order_event_records_expected_payload(tmp_path: Path):
    system = _build_system(tmp_path)

    system._journal_order_event(
        symbol="AAPL",
        asset_class="EQUITY",
        side="BUY",
        quantity=10,
        broker="ibkr",
        lifecycle="submitted",
        order_role="entry",
        signal=0.42,
        confidence=0.77,
        expected_price=123.45,
        metadata={"mode": "market"},
    )

    assert len(system.event_store.events) == 1
    event_type, payload = system.event_store.events[0]
    assert event_type == "ORDER_EXECUTION"
    assert payload["symbol"] == "AAPL"
    assert payload["order_role"] == "entry"
    assert payload["lifecycle"] == "submitted"
    assert payload["broker"] == "ibkr"
    assert payload["expected_price"] == 123.45
    assert payload["metadata"]["mode"] == "market"


def test_write_trade_rejection_also_emits_risk_decision_event(tmp_path: Path, monkeypatch):
    system = _build_system(tmp_path)
    monkeypatch.setattr("core.execution_loop.ApexConfig.TRADE_REJECTION_AUDIT_ENABLED", False)

    system._write_trade_rejection(
        "AAPL",
        "signal_threshold",
        signal=0.12,
        confidence=0.44,
        price=100.0,
        extra={"threshold": 0.2, "at": datetime(2026, 3, 20, 10, 0, 0)},
    )

    assert len(system.event_store.events) == 1
    event_type, payload = system.event_store.events[0]
    assert event_type == "RISK_DECISION"
    assert payload["decision"] == "blocked"
    assert payload["reason"] == "signal_threshold"
    assert payload["asset_class"] == "EQUITY"
    assert payload["metadata"]["threshold"] == 0.2
    assert payload["metadata"]["at"].startswith("2026-03-20T10:00:00")
