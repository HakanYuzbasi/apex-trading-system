"""Tests for replay inspector event reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

from services.replay_inspector.service import ReplayInspectorService


def _write_journal(audit_dir: Path, date: str, rows: list[dict]) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"event_journal_{date}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_active_policies(root: Path) -> None:
    governor_dir = root / "governor_policies"
    governor_dir.mkdir(parents=True, exist_ok=True)
    with (governor_dir / "active_policies.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "asset_class": "EQUITY",
                    "regime": "risk_on",
                    "version": "trust-v42",
                    "created_at": "2099-12-30T09:00:00",
                    "metadata": {
                        "promotion_note": "approved_after_intraday_stress",
                        "approver": "risk-committee",
                    },
                    "tier_controls": {
                        "green": {
                            "size_multiplier": 1.0,
                            "signal_threshold_boost": 0.0,
                            "confidence_boost": 0.0,
                            "halt_new_entries": False,
                        },
                        "yellow": {
                            "size_multiplier": 0.75,
                            "signal_threshold_boost": 0.03,
                            "confidence_boost": 0.05,
                            "halt_new_entries": False,
                        },
                    },
                }
            ],
            handle,
        )


def test_replay_inspector_reconstructs_blocked_and_filled_chains(tmp_path: Path):
    audit_dir = tmp_path / "audit"
    _write_active_policies(tmp_path)
    _write_journal(
        audit_dir,
        "20991231",
        [
            {
                "timestamp": "2099-12-31T10:00:00+00:00",
                "type": "SIGNAL_GENERATION",
                "hash": "sig-1",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "signal": 0.21,
                    "confidence": 0.41,
                },
            },
            {
                "timestamp": "2099-12-31T10:00:01+00:00",
                "type": "RISK_DECISION",
                "hash": "risk-1",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "decision": "blocked",
                    "stage": "entry_gate",
                    "reason": "signal_threshold",
                    "metadata": {
                        "governor_policy_key": "EQUITY:risk_on",
                        "governor_policy_version": "trust-v42",
                        "governor_policy_id": "EQUITY:risk_on:trust-v42",
                        "governor_tier": "yellow",
                    },
                },
            },
            {
                "timestamp": "2099-12-31T10:05:00+00:00",
                "type": "SIGNAL_GENERATION",
                "hash": "sig-2",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "signal": 0.62,
                    "confidence": 0.81,
                },
            },
            {
                "timestamp": "2099-12-31T10:05:01+00:00",
                "type": "RISK_DECISION",
                "hash": "risk-2",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "decision": "allowed",
                    "stage": "pretrade_gateway",
                    "reason": "ok",
                    "metadata": {
                        "governor_policy_key": "EQUITY:risk_on",
                        "governor_policy_version": "trust-v42",
                        "governor_policy_id": "EQUITY:risk_on:trust-v42",
                        "governor_tier": "yellow",
                    },
                },
            },
            {
                "timestamp": "2099-12-31T10:05:02+00:00",
                "type": "ORDER_EXECUTION",
                "hash": "ord-1",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "order_role": "entry",
                    "lifecycle": "submitted",
                    "broker": "ibkr",
                    "quantity": 10,
                    "metadata": {
                        "governor_policy_key": "EQUITY:risk_on",
                        "governor_policy_version": "trust-v42",
                        "governor_policy_id": "EQUITY:risk_on:trust-v42",
                        "governor_tier": "yellow",
                    },
                },
            },
            {
                "timestamp": "2099-12-31T10:05:03+00:00",
                "type": "ORDER_EXECUTION",
                "hash": "ord-2",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "order_role": "entry",
                    "lifecycle": "filled",
                    "broker": "ibkr",
                    "quantity": 10,
                    "status": "FILLED",
                    "fill_price": 101.25,
                    "metadata": {
                        "governor_policy_key": "EQUITY:risk_on",
                        "governor_policy_version": "trust-v42",
                        "governor_policy_id": "EQUITY:risk_on:trust-v42",
                        "governor_tier": "yellow",
                    },
                },
            },
            {
                "timestamp": "2099-12-31T10:05:04+00:00",
                "type": "POSITION_UPDATE",
                "hash": "pos-1",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "quantity": 10,
                    "price": 101.25,
                    "reason": "entry_fill",
                    "metadata": {
                        "governor_policy_key": "EQUITY:risk_on",
                        "governor_policy_version": "trust-v42",
                        "governor_policy_id": "EQUITY:risk_on:trust-v42",
                        "governor_tier": "yellow",
                    },
                },
            },
            {
                "timestamp": "2099-12-31T10:10:00+00:00",
                "type": "SIGNAL_GENERATION",
                "hash": "other-symbol",
                "payload": {
                    "symbol": "MSFT",
                    "asset_class": "EQUITY",
                    "signal": 0.5,
                    "confidence": 0.7,
                },
            },
        ],
    )

    service = ReplayInspectorService(tmp_path)
    response = service.inspect_symbol(symbol="AAPL", limit=50, days=1, include_raw=True)

    assert response.summary.total_events == 7
    assert response.summary.total_chains == 2
    assert response.summary.blocked_chains == 1
    assert response.summary.filled_chains == 1
    assert response.latest_chain is not None
    assert response.latest_chain.final_status == "filled"
    assert response.latest_chain.chain_kind == "entry_execution"
    assert response.latest_chain.signal_event is not None
    assert len(response.latest_chain.risk_events) == 1
    assert len(response.latest_chain.order_events) == 2
    assert len(response.latest_chain.position_events) == 1
    assert response.latest_chain.governor_policy is not None
    assert response.latest_chain.governor_policy.policy_id == "EQUITY:risk_on:trust-v42"
    assert response.latest_chain.governor_policy.version == "trust-v42"
    assert response.latest_chain.governor_policy.source == "active"
    assert response.latest_chain.governor_policy.observed_tier == "yellow"
    assert response.latest_chain.governor_policy.tier_controls["yellow"]["halt_new_entries"] is False
    assert response.chains[0].final_status == "blocked"
    assert response.chains[0].terminal_reason == "signal_threshold"
    assert response.chains[0].governor_policy is not None
    assert response.chains[0].governor_policy.version == "trust-v42"
    assert response.raw_events[-1].event_type == "POSITION_UPDATE"


def test_replay_inspector_links_stress_liquidation_progress(tmp_path: Path):
    audit_dir = tmp_path / "audit"
    _write_journal(
        audit_dir,
        "20991231",
        [
            {
                "timestamp": "2099-12-31T11:00:00+00:00",
                "type": "STRESS_EVALUATION",
                "hash": "stress-eval-1",
                "payload": {
                    "symbol": "PORTFOLIO",
                    "asset_class": "PORTFOLIO",
                    "action": "halt_entries",
                    "worst_scenario_id": "2020_covid_crash",
                    "worst_scenario_name": "2020 COVID Crash",
                    "worst_portfolio_return": -0.12,
                    "scenarios": [
                        {
                            "scenario_id": "2020_covid_crash",
                            "scenario_name": "2020 COVID Crash",
                            "portfolio_return": -0.12,
                            "portfolio_pnl": -12000.0,
                            "top_position_losses": [{"symbol": "AAPL", "pnl": -5000.0}],
                        }
                    ],
                },
            },
            {
                "timestamp": "2099-12-31T11:00:01+00:00",
                "type": "STRESS_ACTION",
                "hash": "stress-plan-1",
                "payload": {
                    "symbol": "PORTFOLIO",
                    "asset_class": "PORTFOLIO",
                    "action": "liquidation_plan",
                    "reason": "worst_return=-12.00% <= -8.00%",
                    "liquidation_plan_id": "stress-unwind-1-20991231T110001000000",
                    "liquidation_plan_epoch": 1,
                    "worst_scenario_id": "2020_covid_crash",
                    "worst_scenario_name": "2020 COVID Crash",
                    "candidates": [
                        {
                            "symbol": "AAPL",
                            "current_qty": 100.0,
                            "target_qty": 40.0,
                            "target_reduction_pct": 0.4,
                            "expected_stress_pnl": -5000.0,
                            "action": "partial_reduce",
                        }
                    ],
                },
            },
            {
                "timestamp": "2099-12-31T11:00:02+00:00",
                "type": "ORDER_EXECUTION",
                "hash": "trim-1",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "order_role": "exit",
                    "lifecycle": "submitted",
                    "broker": "ibkr",
                    "quantity": 20,
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-12.00% <= -8.00%",
                        "exit_type": "partial_reduce",
                        "liquidation_plan_id": "stress-unwind-1-20991231T110001000000",
                        "liquidation_plan_epoch": 1,
                    },
                },
            },
            {
                "timestamp": "2099-12-31T11:00:03+00:00",
                "type": "ORDER_EXECUTION",
                "hash": "trim-2",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "order_role": "exit",
                    "lifecycle": "filled",
                    "broker": "ibkr",
                    "quantity": 20,
                    "status": "FILLED",
                    "fill_price": 100.0,
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-12.00% <= -8.00%",
                        "exit_type": "partial_reduce",
                        "liquidation_plan_id": "stress-unwind-1-20991231T110001000000",
                        "liquidation_plan_epoch": 1,
                    },
                },
            },
            {
                "timestamp": "2099-12-31T11:00:04+00:00",
                "type": "POSITION_UPDATE",
                "hash": "trim-pos-1",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "quantity": 80,
                    "price": 100.0,
                    "reason": "partial_exit_fill",
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-12.00% <= -8.00%",
                        "liquidation_plan_id": "stress-unwind-1-20991231T110001000000",
                        "liquidation_plan_epoch": 1,
                    },
                },
            },
            {
                "timestamp": "2099-12-31T11:01:00+00:00",
                "type": "STRESS_EVALUATION",
                "hash": "stress-eval-2",
                "payload": {
                    "symbol": "PORTFOLIO",
                    "asset_class": "PORTFOLIO",
                    "action": "halt_entries",
                    "worst_scenario_id": "2020_covid_crash",
                    "worst_scenario_name": "2020 COVID Crash",
                    "worst_portfolio_return": -0.08,
                    "scenarios": [
                        {
                            "scenario_id": "2020_covid_crash",
                            "scenario_name": "2020 COVID Crash",
                            "portfolio_return": -0.08,
                            "portfolio_pnl": -8000.0,
                            "top_position_losses": [{"symbol": "AAPL", "pnl": -3000.0}],
                        }
                    ],
                },
            },
            {
                "timestamp": "2099-12-31T11:02:00+00:00",
                "type": "STRESS_ACTION",
                "hash": "stress-plan-2",
                "payload": {
                    "symbol": "PORTFOLIO",
                    "asset_class": "PORTFOLIO",
                    "action": "liquidation_plan",
                    "reason": "worst_return=-10.00% <= -8.00%",
                    "liquidation_plan_id": "stress-unwind-2-20991231T110200000000",
                    "liquidation_plan_epoch": 2,
                    "worst_scenario_id": "2020_covid_crash",
                    "worst_scenario_name": "2020 COVID Crash",
                    "candidates": [
                        {
                            "symbol": "AAPL",
                            "current_qty": 80.0,
                            "target_qty": 30.0,
                            "target_reduction_pct": 0.375,
                            "expected_stress_pnl": -3000.0,
                            "action": "partial_reduce",
                        }
                    ],
                },
            },
        ],
    )

    service = ReplayInspectorService(tmp_path)
    response = service.inspect_symbol(symbol="AAPL", limit=50, days=1, include_raw=True)

    assert response.summary.total_events == 7
    assert response.summary.stress_liquidation_chains == 1
    assert response.latest_chain is not None
    assert response.latest_chain.chain_kind == "exit_execution"
    assert response.latest_chain.liquidation_progress is not None
    assert response.latest_chain.liquidation_progress.status == "in_progress"
    assert response.latest_chain.liquidation_progress.plan_id == "stress-unwind-1-20991231T110001000000"
    assert response.latest_chain.liquidation_progress.plan_epoch == 1
    assert response.latest_chain.liquidation_progress.planned_reduction_qty == 40.0
    assert response.latest_chain.liquidation_progress.executed_reduction_qty == 20.0
    assert response.latest_chain.liquidation_progress.remaining_qty == 80.0
    assert response.latest_chain.liquidation_progress.remaining_stress_pnl == -3000.0
    assert len(response.latest_chain.stress_events) == 3
    assert response.latest_chain.liquidation_progress.plan_event is not None
    assert response.latest_chain.liquidation_progress.plan_event.hash == "stress-plan-1"
    assert response.raw_events[0].event_type == "STRESS_EVALUATION"


def test_replay_inspector_audits_liquidation_plan_across_symbols(tmp_path: Path):
    audit_dir = tmp_path / "audit"
    _write_journal(
        audit_dir,
        "20991231",
        [
            {
                "timestamp": "2099-12-31T12:00:00+00:00",
                "type": "STRESS_EVALUATION",
                "hash": "stress-eval-plan",
                "payload": {
                    "symbol": "PORTFOLIO",
                    "asset_class": "PORTFOLIO",
                    "action": "halt_entries",
                    "worst_scenario_id": "2020_covid_crash",
                    "worst_scenario_name": "2020 COVID Crash",
                    "worst_portfolio_return": -0.14,
                    "scenarios": [
                        {
                            "scenario_id": "2020_covid_crash",
                            "scenario_name": "2020 COVID Crash",
                            "portfolio_return": -0.14,
                            "portfolio_pnl": -14000.0,
                            "top_position_losses": [
                                {"symbol": "AAPL", "pnl": -5000.0},
                                {"symbol": "MSFT", "pnl": -4200.0},
                            ],
                        }
                    ],
                },
            },
            {
                "timestamp": "2099-12-31T12:00:01+00:00",
                "type": "STRESS_ACTION",
                "hash": "stress-plan-cross-book",
                "payload": {
                    "symbol": "PORTFOLIO",
                    "asset_class": "PORTFOLIO",
                    "action": "liquidation_plan",
                    "reason": "worst_return=-14.00% <= -8.00%",
                    "liquidation_plan_id": "stress-unwind-7-20991231T120001000000",
                    "liquidation_plan_epoch": 7,
                    "worst_scenario_id": "2020_covid_crash",
                    "worst_scenario_name": "2020 COVID Crash",
                    "candidates": [
                        {
                            "symbol": "AAPL",
                            "current_qty": 100.0,
                            "target_qty": 40.0,
                            "target_reduction_pct": 0.4,
                            "expected_stress_pnl": -5000.0,
                            "action": "partial_reduce",
                        },
                        {
                            "symbol": "MSFT",
                            "current_qty": 50.0,
                            "target_qty": 50.0,
                            "target_reduction_pct": 1.0,
                            "expected_stress_pnl": -4200.0,
                            "action": "full_exit",
                        },
                    ],
                },
            },
            {
                "timestamp": "2099-12-31T12:00:02+00:00",
                "type": "ORDER_EXECUTION",
                "hash": "aapl-trim-submit",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "order_role": "exit",
                    "lifecycle": "submitted",
                    "broker": "ibkr",
                    "quantity": 20,
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-14.00% <= -8.00%",
                        "exit_type": "partial_reduce",
                        "liquidation_plan_id": "stress-unwind-7-20991231T120001000000",
                        "liquidation_plan_epoch": 7,
                    },
                },
            },
            {
                "timestamp": "2099-12-31T12:00:03+00:00",
                "type": "ORDER_EXECUTION",
                "hash": "aapl-trim-fill",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "order_role": "exit",
                    "lifecycle": "filled",
                    "broker": "ibkr",
                    "quantity": 20,
                    "status": "FILLED",
                    "fill_price": 100.0,
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-14.00% <= -8.00%",
                        "exit_type": "partial_reduce",
                        "liquidation_plan_id": "stress-unwind-7-20991231T120001000000",
                        "liquidation_plan_epoch": 7,
                    },
                },
            },
            {
                "timestamp": "2099-12-31T12:00:04+00:00",
                "type": "POSITION_UPDATE",
                "hash": "aapl-trim-pos",
                "payload": {
                    "symbol": "AAPL",
                    "asset_class": "EQUITY",
                    "quantity": 80,
                    "price": 100.0,
                    "reason": "partial_exit_fill",
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-14.00% <= -8.00%",
                        "liquidation_plan_id": "stress-unwind-7-20991231T120001000000",
                        "liquidation_plan_epoch": 7,
                    },
                },
            },
            {
                "timestamp": "2099-12-31T12:00:05+00:00",
                "type": "ORDER_EXECUTION",
                "hash": "msft-exit-submit",
                "payload": {
                    "symbol": "MSFT",
                    "asset_class": "EQUITY",
                    "order_role": "exit",
                    "lifecycle": "submitted",
                    "broker": "ibkr",
                    "quantity": 50,
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-14.00% <= -8.00%",
                        "exit_type": "full_exit",
                        "liquidation_plan_id": "stress-unwind-7-20991231T120001000000",
                        "liquidation_plan_epoch": 7,
                    },
                },
            },
            {
                "timestamp": "2099-12-31T12:00:06+00:00",
                "type": "ORDER_EXECUTION",
                "hash": "msft-exit-fill",
                "payload": {
                    "symbol": "MSFT",
                    "asset_class": "EQUITY",
                    "order_role": "exit",
                    "lifecycle": "filled",
                    "broker": "ibkr",
                    "quantity": 50,
                    "status": "FILLED",
                    "fill_price": 200.0,
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-14.00% <= -8.00%",
                        "exit_type": "full_exit",
                        "liquidation_plan_id": "stress-unwind-7-20991231T120001000000",
                        "liquidation_plan_epoch": 7,
                    },
                },
            },
            {
                "timestamp": "2099-12-31T12:00:07+00:00",
                "type": "POSITION_UPDATE",
                "hash": "msft-exit-pos",
                "payload": {
                    "symbol": "MSFT",
                    "asset_class": "EQUITY",
                    "quantity": 0,
                    "price": 200.0,
                    "reason": "exit_fill",
                    "metadata": {
                        "exit_reason": "🧯 StressUnwind: worst_return=-14.00% <= -8.00%",
                        "liquidation_plan_id": "stress-unwind-7-20991231T120001000000",
                        "liquidation_plan_epoch": 7,
                    },
                },
            },
        ],
    )

    service = ReplayInspectorService(tmp_path)
    response = service.inspect_plan(
        plan_id="stress-unwind-7-20991231T120001000000",
        limit=50,
        days=1,
        include_raw=True,
    )

    assert response is not None
    assert response.mode == "plan"
    assert response.plan_audit is not None
    assert response.plan_audit.plan_id == "stress-unwind-7-20991231T120001000000"
    assert response.plan_audit.plan_epoch == 7
    assert response.plan_audit.candidate_symbols == ["AAPL", "MSFT"]
    assert response.plan_audit.completed_symbols == 1
    assert response.plan_audit.in_progress_symbols == 1
    assert response.summary.total_chains == 2
    assert response.summary.stress_liquidation_chains == 2
    assert {chain.symbol for chain in response.chains} == {"AAPL", "MSFT"}
    assert all(chain.liquidation_progress is not None for chain in response.chains)
    assert all(chain.liquidation_progress.plan_id == "stress-unwind-7-20991231T120001000000" for chain in response.chains)
    assert len([event for event in response.raw_events if event.hash == "stress-plan-cross-book"]) == 1
