from risk.social_governor_policy import (
    SocialGovernorPolicy,
    SocialGovernorPolicyRepository,
)


def test_social_governor_policy_repo_resolve_with_fallback(tmp_path):
    repo = SocialGovernorPolicyRepository(tmp_path / "governor_policies")
    policies = [
        SocialGovernorPolicy(
            asset_class="GLOBAL",
            regime="default",
            version="sshock-v1",
            reduce_threshold=0.6,
            block_threshold=0.85,
            verified_event_weight=0.3,
            verified_event_probability_floor=0.55,
            max_probability_divergence=0.15,
            max_source_disagreement=0.2,
        ),
        SocialGovernorPolicy(
            asset_class="EQUITY",
            regime="risk_off",
            version="sshock-v1",
            reduce_threshold=0.52,
            block_threshold=0.78,
            verified_event_weight=0.35,
            verified_event_probability_floor=0.55,
            max_probability_divergence=0.12,
            max_source_disagreement=0.18,
        ),
    ]
    snapshot = repo.save_snapshot(version="sshock-v1", policies=policies)
    repo.activate_snapshot(snapshot)

    version, active = repo.load_active()
    assert version == "sshock-v1"
    resolved_exact = repo.resolve(asset_class="EQUITY", regime="risk_off", active_policies=active)
    assert resolved_exact is not None
    assert resolved_exact.block_threshold == 0.78

    resolved_fallback = repo.resolve(asset_class="FOREX", regime="carry", active_policies=active)
    assert resolved_fallback is not None
    assert resolved_fallback.asset_class == "GLOBAL"
