"""Unit tests for institutional pre-trade risk gateway."""

import json

from risk.pretrade_risk_gateway import (
    PreTradeDecision,
    PreTradeLimitConfig,
    PreTradeRiskGateway,
)


def _gateway(tmp_path):
    return PreTradeRiskGateway(
        config=PreTradeLimitConfig(
            enabled=True,
            max_order_notional=250_000,
            max_order_shares=10_000,
            max_price_deviation_bps=250,
            max_participation_rate=0.10,
            max_gross_exposure_ratio=2.0,
        ),
        audit_dir=tmp_path / "audit",
    )


def test_pretrade_gateway_allows_valid_entry(tmp_path):
    gate = _gateway(tmp_path)
    decision = gate.evaluate_entry(
        symbol="AAPL",
        asset_class="EQUITY",
        side="BUY",
        quantity=500,
        price=100.0,
        capital=1_000_000.0,
        current_positions={"MSFT": 400},
        price_cache={"MSFT": 200.0},
        reference_price=99.8,
        adv_shares=100_000,
    )
    assert decision.allowed is True
    assert decision.reason_code == "allowed"


def test_pretrade_gateway_blocks_notional_and_price_band(tmp_path):
    gate = _gateway(tmp_path)
    blocked_notional = gate.evaluate_entry(
        symbol="NVDA",
        asset_class="EQUITY",
        side="BUY",
        quantity=3_000,
        price=100.0,
        capital=1_000_000.0,
        current_positions={},
        price_cache={},
    )
    assert blocked_notional.allowed is False
    assert blocked_notional.reason_code == "max_order_notional"

    blocked_band = gate.evaluate_entry(
        symbol="NVDA",
        asset_class="EQUITY",
        side="BUY",
        quantity=100,
        price=105.0,
        capital=1_000_000.0,
        current_positions={},
        price_cache={},
        reference_price=100.0,
    )
    assert blocked_band.allowed is False
    assert blocked_band.reason_code == "price_band"


def test_pretrade_gateway_blocks_adv_participation_and_gross_exposure(tmp_path):
    gate = _gateway(tmp_path)
    blocked_participation = gate.evaluate_entry(
        symbol="ETH/USDC",
        asset_class="CRYPTO",
        side="BUY",
        quantity=2_000,
        price=10.0,
        capital=500_000.0,
        current_positions={},
        price_cache={},
        adv_shares=10_000,
    )
    assert blocked_participation.allowed is False
    assert blocked_participation.reason_code == "adv_participation"

    blocked_gross = gate.evaluate_entry(
        symbol="AAPL",
        asset_class="EQUITY",
        side="BUY",
        quantity=2_000,
        price=100.0,
        capital=1_000_000.0,
        current_positions={"MSFT": 10_000},
        price_cache={"MSFT": 190.0},
    )
    assert blocked_gross.allowed is False
    assert blocked_gross.reason_code == "gross_exposure"


def test_pretrade_gateway_audit_log_is_hash_chained(tmp_path):
    gate = _gateway(tmp_path)

    d1 = PreTradeDecision(True, "allowed", "ok", {"x": 1})
    d2 = PreTradeDecision(False, "max_order_notional", "blocked", {"x": 2})

    r1 = gate.record_decision(
        symbol="AAPL",
        asset_class="EQUITY",
        side="BUY",
        quantity=100,
        price=100.0,
        decision=d1,
    )
    r2 = gate.record_decision(
        symbol="AAPL",
        asset_class="EQUITY",
        side="BUY",
        quantity=5000,
        price=100.0,
        decision=d2,
    )

    assert r1["hash"]
    assert r2["prev_hash"] == r1["hash"]

    files = list((tmp_path / "audit").glob("pretrade_gateway_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    event_1 = json.loads(lines[0])
    event_2 = json.loads(lines[1])
    assert event_2["prev_hash"] == event_1["hash"]
