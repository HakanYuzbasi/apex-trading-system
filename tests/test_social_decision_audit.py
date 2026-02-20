from risk.social_decision_audit import SocialDecisionAuditRepository


def test_social_decision_audit_is_hash_chained_and_filterable(tmp_path):
    repo = SocialDecisionAuditRepository(tmp_path / "audit" / "social_governor_decisions.jsonl")
    first = repo.append_event(
        {
            "asset_class": "EQUITY",
            "regime": "risk_off",
            "policy_version": "sshock-v1",
            "decision": {"block_new_entries": True},
        }
    )
    second = repo.append_event(
        {
            "asset_class": "CRYPTO",
            "regime": "high_vol",
            "policy_version": "sshock-v1",
            "decision": {"block_new_entries": False},
        }
    )

    assert first["hash"]
    assert second["hash"]
    assert second["prev_hash"] == first["hash"]

    rows = repo.load_events(limit=10, asset_class="CRYPTO", regime="high_vol")
    assert len(rows) == 1
    assert rows[0]["asset_class"] == "CRYPTO"


def test_social_decision_audit_uses_legacy_fallback_for_reads_and_hash_chain(tmp_path):
    legacy_file = tmp_path / "audit" / "social_governor_decisions.jsonl"
    runtime_file = tmp_path / "runtime" / "social_governor_decisions.jsonl"

    legacy_repo = SocialDecisionAuditRepository(legacy_file)
    legacy_row = legacy_repo.append_event(
        {
            "asset_class": "EQUITY",
            "regime": "default",
            "policy_version": "legacy-v1",
            "decision": {"block_new_entries": False},
        }
    )

    runtime_repo = SocialDecisionAuditRepository(
        runtime_file,
        fallback_filepaths=[legacy_file],
    )
    runtime_row = runtime_repo.append_event(
        {
            "asset_class": "FOREX",
            "regime": "default",
            "policy_version": "runtime-v2",
            "decision": {"block_new_entries": False},
        }
    )

    assert runtime_row["prev_hash"] == legacy_row["hash"]

    rows = runtime_repo.load_events(limit=10)
    assert len(rows) == 2
    assert rows[0]["hash"] == legacy_row["hash"]
    assert rows[1]["hash"] == runtime_row["hash"]
