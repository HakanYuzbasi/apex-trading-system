from __future__ import annotations

import json
from pathlib import Path

from models.model_manifest import build_manifest, write_manifest
from risk.governor_policy import GovernorPolicy, GovernorPolicyRepository, PromotionStatus, TierControls
from risk.shadow_deployment import ShadowDeploymentGate


def _write_manifest_with_model(base_dir: Path, manifest_path: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "weights.json").write_text('{"alpha": 1}', encoding="utf-8")
    write_manifest(build_manifest(base_dirs=[base_dir], patterns=("*.json",)), manifest_path)


def _seed_policy(repo: GovernorPolicyRepository, version: str = "trust-v2") -> None:
    repo.save_active(
        [
            GovernorPolicy(
                asset_class="EQUITY",
                regime="risk_on",
                version=version,
                status=PromotionStatus.ACTIVE,
                tier_controls={
                    "green": TierControls(1.0, 0.0, 0.0, False),
                    "yellow": TierControls(0.8, 0.02, 0.03, False),
                },
            )
        ]
    )


def test_shadow_deployment_stages_candidate_in_live_prod(tmp_path: Path) -> None:
    production_manifest = tmp_path / "production_manifest.json"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_manifest_with_model(tmp_path / "production_models", production_manifest)
    _write_manifest_with_model(tmp_path / "candidate_models", candidate_manifest)

    policy_repo = GovernorPolicyRepository(tmp_path / "governor_policies")
    _seed_policy(policy_repo)

    candidate_bundle = {
        "candidate_id": "shadow-eq-v2",
        "model_version": "eq-model-v2",
        "feature_set_version": "features-v3",
        "manifest_path": candidate_manifest.as_posix(),
        "governor_policy_key": "EQUITY:risk_on",
        "governor_policy_version": "trust-v2",
        "offline_metrics": {
            "sharpe": 1.35,
            "max_drawdown": 0.06,
            "win_rate": 0.58,
        },
        "decision_profile": {
            "signal_threshold": 0.55,
            "confidence_threshold": 0.60,
            "halt_on_stress": True,
        },
    }
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    (shadow_dir / "candidate_bundle.json").write_text(json.dumps(candidate_bundle), encoding="utf-8")

    gate = ShadowDeploymentGate(
        directory=shadow_dir,
        production_manifest_path=production_manifest,
        governor_repo=policy_repo,
        environment="prod",
        live_trading=True,
        min_shadow_days=0.0,
        min_observed_signals=2,
        min_decision_agreement_rate=0.50,
        min_offline_sharpe_delta=0.10,
        max_drawdown_increase=0.02,
        max_excess_block_rate=0.50,
    )

    gate.observe_signal(symbol="AAPL", signal=0.80, confidence=0.90, current_position=0.0, stress_halt_active=False)
    gate.observe_production_decision(symbol="AAPL", decision="allowed")
    gate.observe_signal(symbol="MSFT", signal=0.20, confidence=0.40, current_position=0.0, stress_halt_active=False)
    gate.observe_production_decision(symbol="MSFT", decision="blocked")

    update = gate.evaluate(
        live_sharpe=1.00,
        live_drawdown=0.05,
        live_win_rate=0.55,
        live_total_pnl=12000.0,
        live_total_trades=42,
        stress_halt_active=False,
    )

    assert update.current_status == "staged"
    assert update.status_changed is True
    assert gate.snapshot.manual_approval_required is True
    assert gate.snapshot.candidate_manifest_verified is True
    assert gate.snapshot.governor_policy_verified is True
    assert gate.snapshot.decision_agreement_rate == 1.0
    assert "manual_approval_required_for_live_prod" in gate.snapshot.reasons


def test_shadow_deployment_blocks_missing_manifest(tmp_path: Path) -> None:
    production_manifest = tmp_path / "production_manifest.json"
    _write_manifest_with_model(tmp_path / "production_models", production_manifest)

    policy_repo = GovernorPolicyRepository(tmp_path / "governor_policies")
    _seed_policy(policy_repo)

    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    (shadow_dir / "candidate_bundle.json").write_text(
        json.dumps(
            {
                "candidate_id": "shadow-eq-bad",
                "model_version": "eq-model-bad",
                "manifest_path": (tmp_path / "missing_manifest.json").as_posix(),
                "governor_policy_key": "EQUITY:risk_on",
                "governor_policy_version": "trust-v2",
                "offline_metrics": {"sharpe": 2.0, "max_drawdown": 0.01},
            }
        ),
        encoding="utf-8",
    )

    gate = ShadowDeploymentGate(
        directory=shadow_dir,
        production_manifest_path=production_manifest,
        governor_repo=policy_repo,
        environment="prod",
        live_trading=True,
        min_shadow_days=0.0,
        min_observed_signals=1,
    )

    gate.observe_signal(symbol="AAPL", signal=0.80, confidence=0.90, current_position=0.0, stress_halt_active=False)
    gate.observe_production_decision(symbol="AAPL", decision="allowed")
    update = gate.evaluate(
        live_sharpe=1.0,
        live_drawdown=0.05,
        live_win_rate=0.55,
        live_total_pnl=1000.0,
        live_total_trades=5,
        stress_halt_active=False,
    )

    assert update.current_status == "blocked"
    assert gate.snapshot.candidate_manifest_verified is False
    assert any(reason.startswith("candidate_manifest_missing") for reason in gate.snapshot.reasons)


def test_shadow_deployment_auto_promotes_in_staging(tmp_path: Path) -> None:
    production_manifest = tmp_path / "production_manifest.json"
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_manifest_with_model(tmp_path / "production_models", production_manifest)
    _write_manifest_with_model(tmp_path / "candidate_models", candidate_manifest)

    policy_repo = GovernorPolicyRepository(tmp_path / "governor_policies")
    _seed_policy(policy_repo)

    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    (shadow_dir / "candidate_bundle.json").write_text(
        json.dumps(
            {
                "candidate_id": "shadow-stage-v2",
                "model_version": "eq-model-stage-v2",
                "manifest_path": candidate_manifest.as_posix(),
                "governor_policy_key": "EQUITY:risk_on",
                "governor_policy_version": "trust-v2",
                "offline_metrics": {"sharpe": 1.4, "max_drawdown": 0.04},
                "decision_profile": {"signal_threshold": 0.50, "confidence_threshold": 0.50},
            }
        ),
        encoding="utf-8",
    )

    gate = ShadowDeploymentGate(
        directory=shadow_dir,
        production_manifest_path=production_manifest,
        governor_repo=policy_repo,
        environment="staging",
        live_trading=True,
        min_shadow_days=0.0,
        min_observed_signals=1,
        auto_promote_non_prod=True,
    )

    gate.observe_signal(symbol="AAPL", signal=0.90, confidence=0.90, current_position=0.0, stress_halt_active=False)
    gate.observe_production_decision(symbol="AAPL", decision="allowed")
    update = gate.evaluate(
        live_sharpe=1.0,
        live_drawdown=0.05,
        live_win_rate=0.5,
        live_total_pnl=5000.0,
        live_total_trades=10,
        stress_halt_active=False,
    )

    assert update.current_status == "active"
    assert update.activation_applied is True
    assert gate.active_file.exists()
