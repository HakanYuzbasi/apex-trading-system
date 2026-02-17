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
