"""
Shadow deployment and promotion gate for model/policy candidates.

This module stages one active candidate bundle, observes live runtime decisions
without risking capital, and decides whether the bundle is still shadowing,
blocked, staged for approval, or auto-promoted in non-production environments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from models.model_manifest import load_manifest, verify_manifest
from risk.governor_policy import GovernorPolicy, GovernorPolicyRepository


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _manifest_fingerprint(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ShadowOfflineMetrics:
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    turnover: float = 0.0
    slippage_bps: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadowOfflineMetrics":
        return cls(
            sharpe=float(data.get("sharpe", 0.0) or 0.0),
            max_drawdown=float(data.get("max_drawdown", 0.0) or 0.0),
            win_rate=float(data.get("win_rate", 0.0) or 0.0),
            turnover=float(data.get("turnover", 0.0) or 0.0),
            slippage_bps=float(data.get("slippage_bps", 0.0) or 0.0),
        )


@dataclass
class ShadowDecisionProfile:
    signal_threshold: float = 0.55
    confidence_threshold: float = 0.60
    halt_on_stress: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadowDecisionProfile":
        return cls(
            signal_threshold=float(data.get("signal_threshold", 0.55) or 0.55),
            confidence_threshold=float(data.get("confidence_threshold", 0.60) or 0.60),
            halt_on_stress=bool(data.get("halt_on_stress", True)),
        )


@dataclass
class ShadowCandidateBundle:
    candidate_id: str
    model_version: str = ""
    feature_set_version: str = ""
    manifest_path: str = ""
    governor_policy_key: str = ""
    governor_policy_version: str = ""
    created_at: str = field(default_factory=lambda: _utc_now().isoformat())
    offline_metrics: ShadowOfflineMetrics = field(default_factory=ShadowOfflineMetrics)
    decision_profile: ShadowDecisionProfile = field(default_factory=ShadowDecisionProfile)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadowCandidateBundle":
        offline_metrics = data.get("offline_metrics")
        decision_profile = data.get("decision_profile")
        return cls(
            candidate_id=str(data.get("candidate_id", "")).strip(),
            model_version=str(data.get("model_version", "")).strip(),
            feature_set_version=str(data.get("feature_set_version", "")).strip(),
            manifest_path=str(data.get("manifest_path", "")).strip(),
            governor_policy_key=str(data.get("governor_policy_key", "")).strip(),
            governor_policy_version=str(data.get("governor_policy_version", "")).strip(),
            created_at=str(data.get("created_at", _utc_now().isoformat())),
            offline_metrics=ShadowOfflineMetrics.from_dict(offline_metrics if isinstance(offline_metrics, dict) else {}),
            decision_profile=ShadowDecisionProfile.from_dict(decision_profile if isinstance(decision_profile, dict) else {}),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_version": self.model_version,
            "feature_set_version": self.feature_set_version,
            "manifest_path": self.manifest_path,
            "governor_policy_key": self.governor_policy_key,
            "governor_policy_version": self.governor_policy_version,
            "created_at": self.created_at,
            "offline_metrics": asdict(self.offline_metrics),
            "decision_profile": asdict(self.decision_profile),
            "metadata": dict(self.metadata),
        }


@dataclass
class ShadowDeploymentSnapshot:
    status: str = "inactive"
    candidate_id: str = ""
    model_version: str = ""
    feature_set_version: str = ""
    candidate_manifest_path: str = ""
    candidate_manifest_verified: bool = False
    candidate_manifest_errors: list[str] = field(default_factory=list)
    candidate_manifest_fingerprint: str = ""
    production_manifest_fingerprint: str = ""
    governor_policy_key: str = ""
    governor_policy_version: str = ""
    governor_policy_verified: bool = False
    observed_signals: int = 0
    production_allowed_signals: int = 0
    production_blocked_signals: int = 0
    candidate_allowed_signals: int = 0
    candidate_blocked_signals: int = 0
    agreement_count: int = 0
    disagreement_count: int = 0
    excess_block_signals: int = 0
    extra_aggressive_signals: int = 0
    stress_halt_observations: int = 0
    stress_halt_candidate_breaches: int = 0
    shadow_days: float = 0.0
    first_observed_at: str = ""
    last_observed_at: str = ""
    staged_at: str = ""
    activated_at: str = ""
    updated_at: str = field(default_factory=lambda: _utc_now().isoformat())
    manual_approval_required: bool = False
    live_sharpe: float = 0.0
    live_drawdown: float = 0.0
    live_win_rate: float = 0.0
    live_total_pnl: float = 0.0
    live_total_trades: int = 0
    offline_sharpe: float = 0.0
    offline_max_drawdown: float = 0.0
    offline_win_rate: float = 0.0
    decision_agreement_rate: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShadowDeploymentSnapshot":
        return cls(
            status=str(data.get("status", "inactive")),
            candidate_id=str(data.get("candidate_id", "")),
            model_version=str(data.get("model_version", "")),
            feature_set_version=str(data.get("feature_set_version", "")),
            candidate_manifest_path=str(data.get("candidate_manifest_path", "")),
            candidate_manifest_verified=bool(data.get("candidate_manifest_verified", False)),
            candidate_manifest_errors=[str(item) for item in data.get("candidate_manifest_errors", []) if str(item)],
            candidate_manifest_fingerprint=str(data.get("candidate_manifest_fingerprint", "")),
            production_manifest_fingerprint=str(data.get("production_manifest_fingerprint", "")),
            governor_policy_key=str(data.get("governor_policy_key", "")),
            governor_policy_version=str(data.get("governor_policy_version", "")),
            governor_policy_verified=bool(data.get("governor_policy_verified", False)),
            observed_signals=int(data.get("observed_signals", 0) or 0),
            production_allowed_signals=int(data.get("production_allowed_signals", 0) or 0),
            production_blocked_signals=int(data.get("production_blocked_signals", 0) or 0),
            candidate_allowed_signals=int(data.get("candidate_allowed_signals", 0) or 0),
            candidate_blocked_signals=int(data.get("candidate_blocked_signals", 0) or 0),
            agreement_count=int(data.get("agreement_count", 0) or 0),
            disagreement_count=int(data.get("disagreement_count", 0) or 0),
            excess_block_signals=int(data.get("excess_block_signals", 0) or 0),
            extra_aggressive_signals=int(data.get("extra_aggressive_signals", 0) or 0),
            stress_halt_observations=int(data.get("stress_halt_observations", 0) or 0),
            stress_halt_candidate_breaches=int(data.get("stress_halt_candidate_breaches", 0) or 0),
            shadow_days=float(data.get("shadow_days", 0.0) or 0.0),
            first_observed_at=str(data.get("first_observed_at", "")),
            last_observed_at=str(data.get("last_observed_at", "")),
            staged_at=str(data.get("staged_at", "")),
            activated_at=str(data.get("activated_at", "")),
            updated_at=str(data.get("updated_at", _utc_now().isoformat())),
            manual_approval_required=bool(data.get("manual_approval_required", False)),
            live_sharpe=float(data.get("live_sharpe", 0.0) or 0.0),
            live_drawdown=float(data.get("live_drawdown", 0.0) or 0.0),
            live_win_rate=float(data.get("live_win_rate", 0.0) or 0.0),
            live_total_pnl=float(data.get("live_total_pnl", 0.0) or 0.0),
            live_total_trades=int(data.get("live_total_trades", 0) or 0),
            offline_sharpe=float(data.get("offline_sharpe", 0.0) or 0.0),
            offline_max_drawdown=float(data.get("offline_max_drawdown", 0.0) or 0.0),
            offline_win_rate=float(data.get("offline_win_rate", 0.0) or 0.0),
            decision_agreement_rate=float(data.get("decision_agreement_rate", 0.0) or 0.0),
            reasons=[str(item) for item in data.get("reasons", []) if str(item)],
        )


@dataclass
class ShadowEvaluationUpdate:
    status_changed: bool
    previous_status: str
    current_status: str
    activation_applied: bool
    candidate_changed: bool


class ShadowDeploymentGate:
    """File-backed shadow deployment and promotion gate."""

    def __init__(
        self,
        *,
        directory: Path,
        production_manifest_path: Path,
        governor_repo: GovernorPolicyRepository,
        environment: str,
        live_trading: bool,
        enabled: bool = True,
        evaluation_interval_cycles: int = 20,
        min_shadow_days: float = 1.0,
        min_observed_signals: int = 25,
        min_decision_agreement_rate: float = 0.60,
        min_offline_sharpe_delta: float = 0.10,
        max_drawdown_increase: float = 0.02,
        max_excess_block_rate: float = 0.35,
        auto_promote_non_prod: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.production_manifest_path = Path(production_manifest_path)
        self.governor_repo = governor_repo
        self.environment = str(environment or "prod").lower()
        self.live_trading = bool(live_trading)
        self.enabled = bool(enabled)
        self.evaluation_interval_cycles = max(1, int(evaluation_interval_cycles))
        self.min_shadow_days = float(min_shadow_days)
        self.min_observed_signals = max(1, int(min_observed_signals))
        self.min_decision_agreement_rate = float(min_decision_agreement_rate)
        self.min_offline_sharpe_delta = float(min_offline_sharpe_delta)
        self.max_drawdown_increase = float(max_drawdown_increase)
        self.max_excess_block_rate = float(max_excess_block_rate)
        self.auto_promote_non_prod = bool(auto_promote_non_prod)

        self.candidate_file = self.directory / "candidate_bundle.json"
        self.active_file = self.directory / "active_bundle.json"
        self.state_file = self.directory / "shadow_state.json"

        self._candidate: Optional[ShadowCandidateBundle] = None
        self._pending_signals: Dict[str, Dict[str, Any]] = {}
        self.snapshot = self._load_state()
        self._refresh_candidate()

    def _load_state(self) -> ShadowDeploymentSnapshot:
        if not self.state_file.exists():
            return ShadowDeploymentSnapshot()
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return ShadowDeploymentSnapshot()
        if not isinstance(payload, dict):
            return ShadowDeploymentSnapshot()
        return ShadowDeploymentSnapshot.from_dict(payload)

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self.snapshot.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_candidate_bundle(self) -> Optional[ShadowCandidateBundle]:
        if not self.enabled or not self.candidate_file.exists():
            return None
        try:
            payload = json.loads(self.candidate_file.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        candidate = ShadowCandidateBundle.from_dict(payload)
        if not candidate.candidate_id:
            return None
        return candidate

    def _reset_snapshot_for_candidate(self, candidate: ShadowCandidateBundle) -> None:
        self.snapshot = ShadowDeploymentSnapshot(
            status="shadowing" if self.enabled else "inactive",
            candidate_id=candidate.candidate_id,
            model_version=candidate.model_version,
            feature_set_version=candidate.feature_set_version,
            candidate_manifest_path=candidate.manifest_path,
            governor_policy_key=candidate.governor_policy_key,
            governor_policy_version=candidate.governor_policy_version,
            offline_sharpe=candidate.offline_metrics.sharpe,
            offline_max_drawdown=abs(float(candidate.offline_metrics.max_drawdown)),
            offline_win_rate=candidate.offline_metrics.win_rate,
        )
        self._pending_signals = {}

    def _refresh_candidate(self) -> bool:
        candidate = self._load_candidate_bundle()
        previous_id = self._candidate.candidate_id if self._candidate is not None else self.snapshot.candidate_id
        if candidate is None:
            self._candidate = None
            if previous_id:
                self.snapshot = ShadowDeploymentSnapshot()
                self._pending_signals = {}
                self._save_state()
            return previous_id != ""
        self._candidate = candidate
        if candidate.candidate_id != self.snapshot.candidate_id:
            self._reset_snapshot_for_candidate(candidate)
            self._save_state()
            return True
        self.snapshot.model_version = candidate.model_version
        self.snapshot.feature_set_version = candidate.feature_set_version
        self.snapshot.candidate_manifest_path = candidate.manifest_path
        self.snapshot.governor_policy_key = candidate.governor_policy_key
        self.snapshot.governor_policy_version = candidate.governor_policy_version
        self.snapshot.offline_sharpe = candidate.offline_metrics.sharpe
        self.snapshot.offline_max_drawdown = abs(float(candidate.offline_metrics.max_drawdown))
        self.snapshot.offline_win_rate = candidate.offline_metrics.win_rate
        return False

    def _resolve_candidate_manifest_path(self, candidate: ShadowCandidateBundle) -> Path:
        if not candidate.manifest_path:
            return Path("")
        manifest_path = Path(candidate.manifest_path)
        if manifest_path.is_absolute():
            return manifest_path
        return Path.cwd() / manifest_path

    def _validate_candidate_manifest(self, candidate: ShadowCandidateBundle) -> tuple[bool, list[str], str]:
        manifest_path = self._resolve_candidate_manifest_path(candidate)
        if not candidate.manifest_path:
            return False, ["candidate_manifest_missing"], ""
        if not manifest_path.exists():
            return False, [f"candidate_manifest_missing: {manifest_path.as_posix()}"], ""
        try:
            load_manifest(manifest_path)
        except Exception as exc:
            return False, [f"candidate_manifest_load_failed: {manifest_path.as_posix()} ({exc})"], ""
        errors = verify_manifest(manifest_path)
        return len(errors) == 0, errors, _manifest_fingerprint(manifest_path)

    def _policy_matches(self, policy: GovernorPolicy, key: str, version: str) -> bool:
        return policy.key().as_id() == key and policy.version == version

    def _validate_candidate_policy(self, candidate: ShadowCandidateBundle) -> bool:
        if not candidate.governor_policy_key or not candidate.governor_policy_version:
            return True
        for policy in [*self.governor_repo.load_active(), *self.governor_repo.load_candidates()]:
            if self._policy_matches(policy, candidate.governor_policy_key, candidate.governor_policy_version):
                return True
        for audit_event in reversed(self.governor_repo.load_audit_events(limit=2000, policy_key=candidate.governor_policy_key)):
            for field_name in ("to_policy", "from_policy"):
                payload = audit_event.get(field_name)
                if not isinstance(payload, dict):
                    continue
                try:
                    policy = GovernorPolicy.from_dict(payload)
                except Exception:
                    continue
                if self._policy_matches(policy, candidate.governor_policy_key, candidate.governor_policy_version):
                    return True
        return False

    def observe_signal(
        self,
        *,
        symbol: str,
        signal: float,
        confidence: float,
        current_position: float,
        stress_halt_active: bool,
    ) -> None:
        candidate_changed = self._refresh_candidate()
        if candidate_changed:
            self._save_state()
        candidate = self._candidate
        if candidate is None or not self.enabled:
            return
        if abs(float(current_position)) > 1e-9:
            return

        now = _utc_now().isoformat()
        if not self.snapshot.first_observed_at:
            self.snapshot.first_observed_at = now
        self.snapshot.last_observed_at = now
        self.snapshot.observed_signals += 1
        if stress_halt_active:
            self.snapshot.stress_halt_observations += 1

        allows = (
            abs(float(signal)) >= float(candidate.decision_profile.signal_threshold)
            and float(confidence) >= float(candidate.decision_profile.confidence_threshold)
            and not (candidate.decision_profile.halt_on_stress and stress_halt_active)
        )
        if allows:
            self.snapshot.candidate_allowed_signals += 1
            if stress_halt_active and not candidate.decision_profile.halt_on_stress:
                self.snapshot.stress_halt_candidate_breaches += 1
        else:
            self.snapshot.candidate_blocked_signals += 1

        self._pending_signals[str(symbol)] = {
            "candidate_allows": allows,
            "observed_at": now,
        }
        self.snapshot.updated_at = now
        self._save_state()

    def observe_production_decision(self, *, symbol: str, decision: str) -> None:
        self._refresh_candidate()
        candidate = self._candidate
        if candidate is None or not self.enabled:
            return
        pending = self._pending_signals.pop(str(symbol), None)
        if pending is None:
            return

        allows = str(decision).lower() == "allowed"
        if allows:
            self.snapshot.production_allowed_signals += 1
        else:
            self.snapshot.production_blocked_signals += 1

        candidate_allows = bool(pending.get("candidate_allows", False))
        if candidate_allows == allows:
            self.snapshot.agreement_count += 1
        else:
            self.snapshot.disagreement_count += 1
            if not candidate_allows and allows:
                self.snapshot.excess_block_signals += 1
            elif candidate_allows and not allows:
                self.snapshot.extra_aggressive_signals += 1
        self.snapshot.updated_at = _utc_now().isoformat()
        self._save_state()

    def _compute_shadow_days(self, now: datetime) -> float:
        first_observed = _parse_ts(self.snapshot.first_observed_at)
        if first_observed is None:
            return 0.0
        return max(0.0, (now - first_observed).total_seconds() / 86400.0)

    def _decision_agreement_rate(self) -> float:
        total = self.snapshot.agreement_count + self.snapshot.disagreement_count
        if total <= 0:
            return 0.0
        return float(self.snapshot.agreement_count / total)

    def _excess_block_rate(self) -> float:
        total = self.snapshot.production_allowed_signals + self.snapshot.production_blocked_signals
        if total <= 0:
            return 0.0
        return float(self.snapshot.excess_block_signals / total)

    def _activate_non_prod_candidate(self, candidate: ShadowCandidateBundle, now: datetime) -> bool:
        if self.environment == "prod" and self.live_trading:
            return False
        if not self.auto_promote_non_prod:
            return False
        if self.snapshot.activated_at:
            return False
        payload = candidate.to_dict()
        payload["activated_at"] = now.isoformat()
        self.active_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.snapshot.activated_at = payload["activated_at"]
        return True

    def evaluate(
        self,
        *,
        now: Optional[datetime] = None,
        live_sharpe: float,
        live_drawdown: float,
        live_win_rate: float,
        live_total_pnl: float,
        live_total_trades: int,
        stress_halt_active: bool,
    ) -> ShadowEvaluationUpdate:
        evaluation_time = now or _utc_now()
        candidate_changed = self._refresh_candidate()
        previous_status = self.snapshot.status
        activation_applied = False

        if not self.enabled:
            self.snapshot.status = "disabled"
            self.snapshot.reasons = ["shadow_deployment_disabled"]
        elif self._candidate is None:
            self.snapshot = ShadowDeploymentSnapshot(status="inactive", reasons=["no_candidate_bundle"])
        else:
            candidate = self._candidate
            manifest_verified, manifest_errors, manifest_fingerprint = self._validate_candidate_manifest(candidate)
            policy_verified = self._validate_candidate_policy(candidate)
            agreement_rate = self._decision_agreement_rate()
            shadow_days = self._compute_shadow_days(evaluation_time)
            resolved_manifest_path = self._resolve_candidate_manifest_path(candidate)

            self.snapshot.candidate_manifest_path = (
                resolved_manifest_path.as_posix() if candidate.manifest_path else ""
            )
            self.snapshot.candidate_manifest_verified = manifest_verified
            self.snapshot.candidate_manifest_errors = list(manifest_errors)
            self.snapshot.candidate_manifest_fingerprint = manifest_fingerprint
            self.snapshot.production_manifest_fingerprint = _manifest_fingerprint(self.production_manifest_path)
            self.snapshot.governor_policy_verified = policy_verified
            self.snapshot.live_sharpe = float(live_sharpe)
            self.snapshot.live_drawdown = abs(float(live_drawdown))
            self.snapshot.live_win_rate = float(live_win_rate)
            self.snapshot.live_total_pnl = float(live_total_pnl)
            self.snapshot.live_total_trades = int(live_total_trades)
            self.snapshot.shadow_days = shadow_days
            self.snapshot.decision_agreement_rate = agreement_rate
            self.snapshot.updated_at = evaluation_time.isoformat()

            reasons: list[str] = []
            status = "shadowing"
            manual_approval_required = False

            if stress_halt_active:
                self.snapshot.stress_halt_observations += 1

            if not manifest_verified:
                status = "blocked"
                reasons.extend(manifest_errors or ["candidate_manifest_unverified"])
            if not policy_verified:
                status = "blocked"
                reasons.append("candidate_governor_policy_unverified")
            if self.snapshot.observed_signals < self.min_observed_signals:
                reasons.append(
                    f"shadow_observations_pending: signals={self.snapshot.observed_signals} "
                    f"required={self.min_observed_signals}"
                )
            if shadow_days < self.min_shadow_days:
                reasons.append(
                    f"shadow_days_pending: days={shadow_days:.2f} required={self.min_shadow_days:.2f}"
                )
            if status != "blocked":
                if (
                    self.snapshot.observed_signals < self.min_observed_signals
                    or shadow_days < self.min_shadow_days
                ):
                    status = "shadowing"
                else:
                    required_sharpe = float(live_sharpe) + self.min_offline_sharpe_delta
                    allowed_drawdown = abs(float(live_drawdown)) + self.max_drawdown_increase
                    if float(candidate.offline_metrics.sharpe) < required_sharpe:
                        status = "blocked"
                        reasons.append(
                            f"offline_sharpe_below_gate: candidate={candidate.offline_metrics.sharpe:.2f} "
                            f"required={required_sharpe:.2f}"
                        )
                    if abs(float(candidate.offline_metrics.max_drawdown)) > allowed_drawdown:
                        status = "blocked"
                        reasons.append(
                            f"offline_drawdown_above_gate: candidate={abs(float(candidate.offline_metrics.max_drawdown)):.3f} "
                            f"limit={allowed_drawdown:.3f}"
                        )
                    if agreement_rate < self.min_decision_agreement_rate:
                        status = "blocked"
                        reasons.append(
                            f"decision_agreement_below_gate: rate={agreement_rate:.2f} "
                            f"required={self.min_decision_agreement_rate:.2f}"
                        )
                    excess_block_rate = self._excess_block_rate()
                    if excess_block_rate > self.max_excess_block_rate:
                        status = "blocked"
                        reasons.append(
                            f"shadow_excess_block_rate_high: rate={excess_block_rate:.2f} "
                            f"limit={self.max_excess_block_rate:.2f}"
                        )
                    if status != "blocked":
                        if self.environment == "prod" and self.live_trading:
                            status = "staged"
                            manual_approval_required = True
                            if not self.snapshot.staged_at:
                                self.snapshot.staged_at = evaluation_time.isoformat()
                            reasons.append("manual_approval_required_for_live_prod")
                        elif self.auto_promote_non_prod:
                            status = "active"
                            activation_applied = self._activate_non_prod_candidate(candidate, evaluation_time)
                            reasons.append("auto_promoted_non_prod_shadow_candidate")
                        else:
                            status = "ready"
                            reasons.append("promotion_gate_passed")

            self.snapshot.status = status
            self.snapshot.manual_approval_required = manual_approval_required
            self.snapshot.reasons = reasons

        status_changed = candidate_changed or self.snapshot.status != previous_status
        self._save_state()
        return ShadowEvaluationUpdate(
            status_changed=status_changed,
            previous_status=previous_status,
            current_status=self.snapshot.status,
            activation_applied=activation_applied,
            candidate_changed=candidate_changed,
        )

    def should_evaluate(self, cycle: int) -> bool:
        return self.enabled and cycle >= 1 and (cycle == 1 or cycle % self.evaluation_interval_cycles == 0)

    def runtime_state(self) -> Dict[str, Any]:
        payload = self.snapshot.to_dict()
        if self._candidate is not None:
            payload["candidate"] = self._candidate.to_dict()
        else:
            payload["candidate"] = None
        return payload
