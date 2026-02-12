"""
risk/governor_policy.py

Policy model, resolver, and promotion workflow for performance governors.
Policy keys are (asset_class, regime) with asset_class-first fallback.
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from risk.performance_governor import GovernorTier

logger = logging.getLogger(__name__)


class PromotionStatus(str, Enum):
    CANDIDATE = "candidate"
    STAGED = "staged"
    APPROVED = "approved"
    ACTIVE = "active"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PolicyKey:
    asset_class: str
    regime: str

    def normalized(self) -> "PolicyKey":
        return PolicyKey(
            asset_class=self.asset_class.upper().strip(),
            regime=self.regime.lower().strip() or "default",
        )

    def as_id(self) -> str:
        n = self.normalized()
        return f"{n.asset_class}:{n.regime}"


@dataclass
class TierControls:
    size_multiplier: float
    signal_threshold_boost: float
    confidence_boost: float
    halt_new_entries: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "size_multiplier": float(self.size_multiplier),
            "signal_threshold_boost": float(self.signal_threshold_boost),
            "confidence_boost": float(self.confidence_boost),
            "halt_new_entries": bool(self.halt_new_entries),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TierControls":
        return cls(
            size_multiplier=float(data.get("size_multiplier", 1.0)),
            signal_threshold_boost=float(data.get("signal_threshold_boost", 0.0)),
            confidence_boost=float(data.get("confidence_boost", 0.0)),
            halt_new_entries=bool(data.get("halt_new_entries", False)),
        )


@dataclass
class GovernorPolicy:
    asset_class: str
    regime: str
    version: str
    oos_sharpe: float = 0.0
    oos_drawdown: float = 0.0
    historical_mdd: float = 0.08
    sharpe_floor_63d: float = 0.2
    status: PromotionStatus = PromotionStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, object] = field(default_factory=dict)
    tier_controls: Dict[str, TierControls] = field(default_factory=dict)

    def key(self) -> PolicyKey:
        return PolicyKey(self.asset_class, self.regime).normalized()

    def policy_id(self) -> str:
        key = self.key()
        return f"{key.asset_class}:{key.regime}:{self.version}"

    def control_for_tier(self, tier: GovernorTier) -> TierControls:
        control = self.tier_controls.get(tier.value)
        if control:
            return control
        return DEFAULT_TIER_CONTROLS[tier]

    def to_dict(self) -> Dict[str, object]:
        return {
            "asset_class": self.asset_class,
            "regime": self.regime,
            "version": self.version,
            "oos_sharpe": float(self.oos_sharpe),
            "oos_drawdown": float(self.oos_drawdown),
            "historical_mdd": float(self.historical_mdd),
            "sharpe_floor_63d": float(self.sharpe_floor_63d),
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "tier_controls": {k: v.to_dict() for k, v in self.tier_controls.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GovernorPolicy":
        controls_data = data.get("tier_controls", {})
        controls = {
            str(k): TierControls.from_dict(v)
            for k, v in controls_data.items()
            if isinstance(v, dict)
        }
        return cls(
            asset_class=str(data.get("asset_class", "GLOBAL")).upper(),
            regime=str(data.get("regime", "default")).lower(),
            version=str(data.get("version", "v1")),
            oos_sharpe=float(data.get("oos_sharpe", 0.0)),
            oos_drawdown=float(data.get("oos_drawdown", 0.0)),
            historical_mdd=float(data.get("historical_mdd", 0.08)),
            sharpe_floor_63d=float(data.get("sharpe_floor_63d", 0.2)),
            status=PromotionStatus(str(data.get("status", PromotionStatus.ACTIVE.value))),
            created_at=str(data.get("created_at", datetime.utcnow().isoformat())),
            metadata=dict(data.get("metadata", {})),
            tier_controls=controls,
        )


DEFAULT_TIER_CONTROLS: Dict[GovernorTier, TierControls] = {
    GovernorTier.GREEN: TierControls(1.00, 0.00, 0.00, False),
    GovernorTier.YELLOW: TierControls(0.75, 0.03, 0.05, False),
    GovernorTier.ORANGE: TierControls(0.50, 0.07, 0.10, False),
    GovernorTier.RED: TierControls(0.25, 0.12, 0.20, True),
}


def default_policy_for(asset_class: str, regime: str = "default") -> GovernorPolicy:
    asset = asset_class.upper()
    reg = regime.lower() or "default"
    return GovernorPolicy(
        asset_class=asset,
        regime=reg,
        version="bootstrap-v1",
        tier_controls={k.value: v for k, v in DEFAULT_TIER_CONTROLS.items()},
        metadata={"source": "bootstrap_default"},
    )


class GovernorPolicyRepository:
    """File-backed policy repository."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.active_file = self.directory / "active_policies.json"
        self.candidates_file = self.directory / "candidate_policies.json"
        self.audit_file = self.directory / "policy_audit_log.jsonl"
        self.directory.mkdir(parents=True, exist_ok=True)

    def load_active(self) -> List[GovernorPolicy]:
        return self._load(self.active_file)

    def save_active(self, policies: List[GovernorPolicy]) -> None:
        self._save(self.active_file, policies)

    def load_candidates(self) -> List[GovernorPolicy]:
        return self._load(self.candidates_file)

    def save_candidates(self, policies: List[GovernorPolicy]) -> None:
        self._save(self.candidates_file, policies)

    def upsert_active(self, policy: GovernorPolicy) -> None:
        policies = self.load_active()
        key = policy.key().as_id()
        updated: List[GovernorPolicy] = []
        replaced = False
        for existing in policies:
            if existing.key().as_id() == key:
                updated.append(policy)
                replaced = True
            else:
                updated.append(existing)
        if not replaced:
            updated.append(policy)
        self.save_active(updated)

    def append_audit_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = dict(event)
        record.setdefault("event_id", f"gpa-{uuid4().hex[:12]}")
        record.setdefault("timestamp", datetime.utcnow().isoformat())
        record["prev_hash"] = self._last_audit_hash()

        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def load_audit_events(
        self,
        limit: int = 200,
        policy_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.audit_file.exists():
            return []
        events: List[Dict[str, Any]] = []
        try:
            with open(self.audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    if policy_key and str(row.get("policy_key")) != policy_key:
                        continue
                    events.append(row)
        except Exception as exc:
            logger.error("Failed to load governor policy audit events: %s", exc)
            return []

        if limit <= 0:
            return []
        return events[-limit:]

    def _load(self, filepath: Path) -> List[GovernorPolicy]:
        if not filepath.exists():
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            return [GovernorPolicy.from_dict(item) for item in data if isinstance(item, dict)]
        except Exception as exc:
            logger.error("Failed to load governor policies from %s: %s", filepath, exc)
            return []

    def _save(self, filepath: Path, policies: List[GovernorPolicy]) -> None:
        serial = [p.to_dict() for p in policies]
        tmp = filepath.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(serial, f, indent=2)
        tmp.replace(filepath)

    def _last_audit_hash(self) -> Optional[str]:
        if not self.audit_file.exists():
            return None
        last: Optional[str] = None
        try:
            with open(self.audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last = line
            if not last:
                return None
            parsed = json.loads(last)
            if not isinstance(parsed, dict):
                return None
            value = parsed.get("hash")
            return str(value) if value else None
        except Exception:
            return None


class GovernorPolicyResolver:
    """
    Resolve active policy by (asset_class, regime) with deterministic fallback:
    1) exact (asset_class, regime)
    2) (asset_class, default)
    3) (GLOBAL, default)
    4) bootstrap default for requested asset class
    """

    def __init__(self, policies: Optional[List[GovernorPolicy]] = None):
        self._policies: Dict[str, GovernorPolicy] = {}
        self.reload(policies or [])

    def reload(self, policies: List[GovernorPolicy]) -> None:
        self._policies = {p.key().as_id(): p for p in policies}

    def resolve(self, asset_class: str, regime: str) -> GovernorPolicy:
        asset = asset_class.upper().strip()
        reg = regime.lower().strip() or "default"
        candidates = [
            PolicyKey(asset, reg).as_id(),
            PolicyKey(asset, "default").as_id(),
            PolicyKey("GLOBAL", "default").as_id(),
        ]
        for candidate in candidates:
            policy = self._policies.get(candidate)
            if policy:
                return policy
        return default_policy_for(asset, "default")

    def controls_for(
        self,
        asset_class: str,
        regime: str,
        tier: GovernorTier,
    ) -> Tuple[TierControls, GovernorPolicy]:
        policy = self.resolve(asset_class, regime)
        return policy.control_for_tier(tier), policy

    def historical_mdd_baseline(self, default_value: float = 0.08) -> float:
        if not self._policies:
            return default_value
        values = [p.historical_mdd for p in self._policies.values() if p.historical_mdd > 0]
        return max(values) if values else default_value


@dataclass
class PromotionDecision:
    accepted: bool
    manual_approval_required: bool
    status: PromotionStatus
    reason: str


class PolicyPromotionService:
    """
    Promotion policy:
    - staging/paper/dev OR non-live: auto-activate if guardrails pass
    - production live: stage candidate; require manual approval
    """

    SAFE_AUTO_ENVS = {"dev", "development", "staging", "paper", "test"}

    def __init__(
        self,
        repository: GovernorPolicyRepository,
        environment: str,
        live_trading: bool,
        auto_promote_non_prod: bool = True,
    ):
        self.repository = repository
        self.environment = environment.lower().strip()
        self.live_trading = bool(live_trading)
        self.auto_promote_non_prod = bool(auto_promote_non_prod)

    def evaluate_guardrails(
        self,
        candidate: GovernorPolicy,
        active: Optional[GovernorPolicy],
    ) -> Tuple[bool, str]:
        if active is None:
            return True, "No active policy for key; accepting bootstrap candidate"

        if candidate.oos_sharpe <= active.oos_sharpe:
            return False, (
                f"OOS Sharpe did not improve ({candidate.oos_sharpe:.3f} <= "
                f"{active.oos_sharpe:.3f})"
            )

        if candidate.oos_drawdown > active.oos_drawdown:
            return False, (
                f"OOS drawdown worsened ({candidate.oos_drawdown:.2%} > "
                f"{active.oos_drawdown:.2%})"
            )

        return True, "Guardrails passed: improved OOS Sharpe and non-worse drawdown"

    def submit_candidate(self, candidate: GovernorPolicy) -> PromotionDecision:
        candidate.status = PromotionStatus.CANDIDATE
        active = self.repository.load_active()
        active_by_key = {p.key().as_id(): p for p in active}
        current = active_by_key.get(candidate.key().as_id())
        self._audit_event(
            action="candidate_submitted",
            actor="system",
            policy_key=candidate.key().as_id(),
            reason="Candidate submitted for promotion workflow",
            from_policy=current,
            to_policy=candidate,
        )
        ok, reason = self.evaluate_guardrails(candidate, current)
        if not ok:
            candidate.status = PromotionStatus.REJECTED
            self._upsert_candidate(candidate)
            self._audit_event(
                action="promotion_rejected",
                actor="system",
                policy_key=candidate.key().as_id(),
                reason=reason,
                from_policy=current,
                to_policy=candidate,
            )
            return PromotionDecision(False, False, candidate.status, reason)

        auto_promote = (
            self.auto_promote_non_prod
            and ((self.environment in self.SAFE_AUTO_ENVS) or (not self.live_trading))
        )
        if auto_promote:
            candidate.status = PromotionStatus.ACTIVE
            self.repository.upsert_active(candidate)
            self._remove_candidate(candidate.policy_id())
            self._audit_event(
                action="auto_activated",
                actor="system",
                policy_key=candidate.key().as_id(),
                reason="Auto-promoted in non-prod mode",
                from_policy=current,
                to_policy=candidate,
            )
            return PromotionDecision(True, False, candidate.status, "Auto-promoted in non-prod mode")

        candidate.status = PromotionStatus.STAGED
        self._upsert_candidate(candidate)
        self._audit_event(
            action="staged_for_approval",
            actor="system",
            policy_key=candidate.key().as_id(),
            reason="Staged for manual production approval",
            from_policy=current,
            to_policy=candidate,
        )
        return PromotionDecision(True, True, candidate.status, "Staged for manual production approval")

    def approve_staged(self, policy_id: str, approver: str, reason: str = "") -> PromotionDecision:
        candidates = self.repository.load_candidates()
        target: Optional[GovernorPolicy] = None
        for candidate in candidates:
            if candidate.policy_id() == policy_id and candidate.status == PromotionStatus.STAGED:
                target = candidate
                break

        if target is None:
            self._audit_event(
                action="manual_approval_failed",
                actor=approver,
                policy_key="unknown",
                reason=f"Staged policy not found: {policy_id}",
            )
            return PromotionDecision(False, True, PromotionStatus.REJECTED, "Staged policy not found")

        active_by_key = self._active_by_key()
        current = active_by_key.get(target.key().as_id())
        target.status = PromotionStatus.ACTIVE
        target.metadata["approved_by"] = approver
        target.metadata["approved_at"] = datetime.utcnow().isoformat()
        if reason.strip():
            target.metadata["approval_reason"] = reason.strip()
        self.repository.upsert_active(target)
        self._remove_candidate(policy_id)
        self._audit_event(
            action="manual_approved",
            actor=approver,
            policy_key=target.key().as_id(),
            reason=reason or "Manual production approval",
            from_policy=current,
            to_policy=target,
        )
        return PromotionDecision(True, True, PromotionStatus.ACTIVE, "Policy approved and activated")

    def rollback_active(
        self,
        asset_class: str,
        regime: str,
        approver: str,
        reason: str,
        target_version: Optional[str] = None,
    ) -> PromotionDecision:
        key = PolicyKey(asset_class, regime).as_id()
        active_by_key = self._active_by_key()
        current = active_by_key.get(key)
        if current is None:
            msg = f"No active policy found for {key}"
            self._audit_event(
                action="rollback_failed",
                actor=approver,
                policy_key=key,
                reason=msg,
            )
            return PromotionDecision(False, True, PromotionStatus.REJECTED, msg)

        if target_version and target_version.strip() == current.version:
            msg = f"Current policy already on requested version {current.version}"
            self._audit_event(
                action="rollback_failed",
                actor=approver,
                policy_key=key,
                reason=msg,
                from_policy=current,
            )
            return PromotionDecision(False, True, PromotionStatus.REJECTED, msg)

        target = self._resolve_rollback_target(
            policy_key=key,
            current_version=current.version,
            target_version=target_version,
        )
        if target is None:
            if target_version:
                msg = f"Rollback target version not found: {target_version}"
            else:
                msg = "No historical active policy available for rollback"
            self._audit_event(
                action="rollback_failed",
                actor=approver,
                policy_key=key,
                reason=msg,
                from_policy=current,
            )
            return PromotionDecision(False, True, PromotionStatus.REJECTED, msg)

        target.status = PromotionStatus.ACTIVE
        target.metadata["rolled_back_by"] = approver
        target.metadata["rolled_back_at"] = datetime.utcnow().isoformat()
        target.metadata["rollback_reason"] = reason
        target.metadata["rollback_from_version"] = current.version
        self.repository.upsert_active(target)
        self._audit_event(
            action="rollback_activated",
            actor=approver,
            policy_key=key,
            reason=reason,
            from_policy=current,
            to_policy=target,
            metadata={"target_version_requested": target_version},
        )
        return PromotionDecision(
            True,
            True,
            PromotionStatus.ACTIVE,
            f"Rolled back to version {target.version}",
        )

    def _upsert_candidate(self, policy: GovernorPolicy) -> None:
        candidates = self.repository.load_candidates()
        updated: List[GovernorPolicy] = []
        replaced = False
        for existing in candidates:
            if existing.policy_id() == policy.policy_id():
                updated.append(policy)
                replaced = True
            else:
                updated.append(existing)
        if not replaced:
            updated.append(policy)
        self.repository.save_candidates(updated)

    def _remove_candidate(self, policy_id: str) -> None:
        candidates = self.repository.load_candidates()
        remaining = [p for p in candidates if p.policy_id() != policy_id]
        self.repository.save_candidates(remaining)

    def _active_by_key(self) -> Dict[str, GovernorPolicy]:
        return {p.key().as_id(): p for p in self.repository.load_active()}

    def _resolve_rollback_target(
        self,
        policy_key: str,
        current_version: str,
        target_version: Optional[str],
    ) -> Optional[GovernorPolicy]:
        events = self.repository.load_audit_events(limit=5000, policy_key=policy_key)
        wanted = target_version.strip() if target_version else None

        for event in reversed(events):
            for field_name in ("from_policy", "to_policy"):
                snapshot = event.get(field_name)
                if not isinstance(snapshot, dict):
                    continue
                try:
                    policy = GovernorPolicy.from_dict(snapshot)
                except Exception:
                    continue
                if policy.key().as_id() != policy_key:
                    continue
                if policy.version == current_version:
                    continue
                if wanted and policy.version != wanted:
                    continue
                return policy
        return None

    def _audit_event(
        self,
        action: str,
        actor: str,
        policy_key: str,
        reason: str,
        from_policy: Optional[GovernorPolicy] = None,
        to_policy: Optional[GovernorPolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "action": action,
            "actor": actor,
            "policy_key": policy_key,
            "reason": reason,
            "environment": self.environment,
            "live_trading": self.live_trading,
        }
        if from_policy is not None:
            payload["from_policy"] = from_policy.to_dict()
        if to_policy is not None:
            payload["to_policy"] = to_policy.to_dict()
        if metadata:
            payload["metadata"] = dict(metadata)
        self.repository.append_audit_event(payload)
