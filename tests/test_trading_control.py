from core.trading_control import (
    mark_governor_policy_reload_processed,
    mark_kill_switch_reset_processed,
    read_control_state,
    request_governor_policy_reload,
    request_kill_switch_reset,
)


def test_read_control_state_defaults_when_missing(tmp_path):
    state = read_control_state(tmp_path / "missing.json")
    assert state["kill_switch_reset_requested"] is False
    assert state["request_id"] is None
    assert state["requested_by"] is None


def test_request_and_process_kill_switch_reset(tmp_path):
    path = tmp_path / "trading_control_commands.json"
    requested = request_kill_switch_reset(
        filepath=path,
        requested_by="ops-user",
        reason="Runbook verified recovery conditions are satisfied",
    )
    assert requested["kill_switch_reset_requested"] is True
    assert requested["request_id"].startswith("ksr-")
    assert requested["requested_by"] == "ops-user"
    assert requested["processed_at"] is None

    processed = mark_kill_switch_reset_processed(
        filepath=path,
        processed_by="apex-trader",
        note="Kill-switch reset applied",
    )
    assert processed["kill_switch_reset_requested"] is False
    assert processed["processed_by"] == "apex-trader"
    assert processed["processing_note"] == "Kill-switch reset applied"
    assert processed["processed_at"] is not None


def test_request_and_process_governor_policy_reload(tmp_path):
    path = tmp_path / "trading_control_commands.json"
    requested = request_governor_policy_reload(
        filepath=path,
        requested_by="risk-admin",
        reason="Approved staged policy in production",
    )
    assert requested["governor_policy_reload_requested"] is True
    assert requested["governor_policy_reload_request_id"].startswith("gpr-")
    assert requested["governor_policy_reload_requested_by"] == "risk-admin"

    processed = mark_governor_policy_reload_processed(
        filepath=path,
        processed_by="apex-trader",
        note="Governor policies reloaded from active file",
    )
    assert processed["governor_policy_reload_requested"] is False
    assert processed["governor_policy_reload_processed_by"] == "apex-trader"
    assert processed["governor_policy_reload_processing_note"] == "Governor policies reloaded from active file"
