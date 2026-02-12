"""Tests for governor policy resolver and promotion flow."""

from pathlib import Path

from risk.governor_policy import (
    GovernorPolicy,
    GovernorPolicyRepository,
    GovernorPolicyResolver,
    PolicyPromotionService,
    PromotionStatus,
    TierControls,
)


def _policy(asset_class: str, regime: str, version: str, sharpe: float, dd: float) -> GovernorPolicy:
    return GovernorPolicy(
        asset_class=asset_class,
        regime=regime,
        version=version,
        oos_sharpe=sharpe,
        oos_drawdown=dd,
        tier_controls={
            "green": TierControls(1.0, 0.0, 0.0, False),
            "yellow": TierControls(0.8, 0.02, 0.03, False),
            "orange": TierControls(0.6, 0.05, 0.06, False),
            "red": TierControls(0.3, 0.10, 0.12, True),
        },
    )


def test_policy_resolver_exact_then_fallback():
    resolver = GovernorPolicyResolver(
        policies=[
            _policy("GLOBAL", "default", "g1", 0.0, 0.1),
            _policy("EQUITY", "default", "e1", 0.8, 0.1),
            _policy("EQUITY", "risk_off", "e2", 1.0, 0.08),
        ]
    )

    exact = resolver.resolve("EQUITY", "risk_off")
    assert exact.version == "e2"

    fallback_asset = resolver.resolve("EQUITY", "unknown_regime")
    assert fallback_asset.version == "e1"

    fallback_global = resolver.resolve("FOREX", "carry")
    assert fallback_global.version == "g1"


def test_policy_promotion_staged_in_production(tmp_path: Path):
    repo = GovernorPolicyRepository(tmp_path / "gov")
    active = _policy("EQUITY", "default", "v1", sharpe=1.1, dd=0.10)
    repo.save_active([active])

    service = PolicyPromotionService(repository=repo, environment="prod", live_trading=True)
    candidate = _policy("EQUITY", "default", "v2", sharpe=1.25, dd=0.09)
    decision = service.submit_candidate(candidate)

    assert decision.accepted is True
    assert decision.manual_approval_required is True
    assert decision.status == PromotionStatus.STAGED


def test_policy_promotion_auto_in_staging(tmp_path: Path):
    repo = GovernorPolicyRepository(tmp_path / "gov")
    active = _policy("CRYPTO", "default", "v1", sharpe=0.9, dd=0.14)
    repo.save_active([active])

    service = PolicyPromotionService(repository=repo, environment="staging", live_trading=True)
    candidate = _policy("CRYPTO", "default", "v2", sharpe=1.05, dd=0.12)
    decision = service.submit_candidate(candidate)

    assert decision.accepted is True
    assert decision.manual_approval_required is False
    assert decision.status == PromotionStatus.ACTIVE

    active_policies = repo.load_active()
    assert any(p.version == "v2" for p in active_policies)


def test_manual_approval_writes_audit_event(tmp_path: Path):
    repo = GovernorPolicyRepository(tmp_path / "gov")
    active = _policy("EQUITY", "default", "v1", sharpe=1.0, dd=0.10)
    repo.save_active([active])

    service = PolicyPromotionService(repository=repo, environment="prod", live_trading=True)
    candidate = _policy("EQUITY", "default", "v2", sharpe=1.2, dd=0.08)
    submit = service.submit_candidate(candidate)
    assert submit.status == PromotionStatus.STAGED

    decision = service.approve_staged(
        policy_id=candidate.policy_id(),
        approver="risk-admin",
        reason="Paper validation complete and approved for live",
    )
    assert decision.accepted is True

    active_policy = repo.load_active()[0]
    assert active_policy.version == "v2"
    assert active_policy.metadata["approved_by"] == "risk-admin"

    audit = repo.load_audit_events(limit=20, policy_key="EQUITY:default")
    assert any(event.get("action") == "manual_approved" for event in audit)


def test_policy_rollback_restores_previous_active_version(tmp_path: Path):
    repo = GovernorPolicyRepository(tmp_path / "gov")
    active = _policy("CRYPTO", "default", "v1", sharpe=1.1, dd=0.12)
    repo.save_active([active])

    service = PolicyPromotionService(repository=repo, environment="prod", live_trading=True)
    candidate = _policy("CRYPTO", "default", "v2", sharpe=1.25, dd=0.10)
    submit = service.submit_candidate(candidate)
    assert submit.status == PromotionStatus.STAGED
    approved = service.approve_staged(
        policy_id=candidate.policy_id(),
        approver="risk-admin",
        reason="Promoting tuned policy",
    )
    assert approved.accepted is True

    rollback = service.rollback_active(
        asset_class="CRYPTO",
        regime="default",
        approver="risk-admin",
        reason="Live degradation detected",
    )
    assert rollback.accepted is True
    assert "v1" in rollback.reason

    current = repo.load_active()[0]
    assert current.version == "v1"
    assert current.metadata["rollback_from_version"] == "v2"

    audit = repo.load_audit_events(limit=50, policy_key="CRYPTO:default")
    assert any(event.get("action") == "rollback_activated" for event in audit)
