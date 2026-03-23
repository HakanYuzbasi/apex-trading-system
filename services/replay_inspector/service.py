"""Replay inspector for symbol- and plan-scoped event-journal reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from risk.governor_policy import GovernorPolicy, GovernorPolicyRepository
from services.replay_inspector.schemas import (
    ReplayChain,
    ReplayGovernorPolicySnapshot,
    ReplayInspectionResponse,
    ReplayInspectionSummary,
    ReplayLiquidationProgress,
    ReplayPlanAudit,
    ReplayTimelineEvent,
)


def _parse_ts(value: str) -> datetime:
    normalized = str(value or "").replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class _ChainState:
    chain_id: str
    symbol: str
    asset_class: str
    chain_kind: str
    started_at: str
    signal_event: Optional[ReplayTimelineEvent] = None
    risk_events: List[ReplayTimelineEvent] = field(default_factory=list)
    order_events: List[ReplayTimelineEvent] = field(default_factory=list)
    position_events: List[ReplayTimelineEvent] = field(default_factory=list)
    governor_policy: Optional[ReplayGovernorPolicySnapshot] = None
    completed_at: Optional[str] = None
    final_status: str = "open"
    terminal_reason: str = ""

    def to_schema(self) -> ReplayChain:
        return ReplayChain(
            chain_id=self.chain_id,
            symbol=self.symbol,
            asset_class=self.asset_class,
            chain_kind=self.chain_kind,
            started_at=self.started_at,
            completed_at=self.completed_at,
            final_status=self.final_status,
            terminal_reason=self.terminal_reason,
            signal_event=self.signal_event,
            risk_events=self.risk_events,
            order_events=self.order_events,
            position_events=self.position_events,
            governor_policy=self.governor_policy,
        )


class EventReplayRepository:
    """Load symbol-scoped event journal rows from append-only daily files."""

    def __init__(self, audit_dir: Path) -> None:
        self.audit_dir = Path(audit_dir)

    def _journal_files(self, *, days: int) -> List[Path]:
        files = sorted(self.audit_dir.glob("event_journal_*.jsonl"))
        if days <= 0:
            return files
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=max(days - 1, 0))
        selected: List[Path] = []
        for path in files:
            stem = path.stem
            try:
                file_date = datetime.strptime(stem.rsplit("_", 1)[-1], "%Y%m%d").date()
            except ValueError:
                selected.append(path)
                continue
            if file_date >= cutoff:
                selected.append(path)
        return selected

    def load_symbol_events(self, *, symbol: str, limit: int = 500, days: int = 7) -> List[ReplayTimelineEvent]:
        bounded_limit = min(max(int(limit), 1), 5000)
        symbol_filter = str(symbol).strip()
        rows: List[ReplayTimelineEvent] = []
        for path in self._journal_files(days=days):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    event_payload = payload.get("payload") or {}
                    if not isinstance(event_payload, dict):
                        continue
                    if str(event_payload.get("symbol", "")).strip() != symbol_filter:
                        continue
                    rows.append(
                        ReplayTimelineEvent(
                            timestamp=str(payload.get("timestamp", "")),
                            event_type=str(payload.get("type", "")),
                            symbol=symbol_filter,
                            asset_class=str(event_payload.get("asset_class", "")),
                            hash=str(payload.get("hash", "")),
                            payload=event_payload,
                        )
                    )
        rows.sort(key=lambda row: (_parse_ts(row.timestamp), row.hash))
        return rows[-bounded_limit:]

    def load_portfolio_stress_events(self, *, limit: int = 500, days: int = 7) -> List[ReplayTimelineEvent]:
        bounded_limit = min(max(int(limit), 1), 5000)
        rows: List[ReplayTimelineEvent] = []
        for path in self._journal_files(days=days):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    event_payload = payload.get("payload") or {}
                    if not isinstance(event_payload, dict):
                        continue
                    if str(event_payload.get("symbol", "")).strip() != "PORTFOLIO":
                        continue
                    event_type = str(payload.get("type", ""))
                    if event_type not in {"STRESS_EVALUATION", "STRESS_ACTION"}:
                        continue
                    rows.append(
                        ReplayTimelineEvent(
                            timestamp=str(payload.get("timestamp", "")),
                            event_type=event_type,
                            symbol="PORTFOLIO",
                            asset_class=str(event_payload.get("asset_class", "")),
                            hash=str(payload.get("hash", "")),
                            payload=event_payload,
                        )
                    )
        rows.sort(key=lambda row: (_parse_ts(row.timestamp), row.hash))
        return rows[-bounded_limit:]


class ReplayInspectorService:
    """Reconstruct actionable symbol decision chains from journaled events."""

    def __init__(self, tenant_root: Path) -> None:
        self.tenant_root = Path(tenant_root)
        self.repo = EventReplayRepository(self.tenant_root / "audit")
        self.governor_repo = GovernorPolicyRepository(self.tenant_root / "governor_policies")

    @staticmethod
    def _governor_metadata_from_event(event: ReplayTimelineEvent) -> Optional[Dict[str, str]]:
        metadata = event.payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        policy_key = str(metadata.get("governor_policy_key", "")).strip()
        version = str(metadata.get("governor_policy_version", "")).strip()
        if not policy_key or not version:
            return None
        return {
            "policy_key": policy_key,
            "policy_id": str(metadata.get("governor_policy_id", "")).strip(),
            "version": version,
            "observed_tier": str(metadata.get("governor_tier", "")).strip(),
        }

    def _resolve_governor_policy_snapshot(self, chain: _ChainState) -> Optional[ReplayGovernorPolicySnapshot]:
        policy_meta: Optional[Dict[str, str]] = None
        for event in [*chain.risk_events, *chain.order_events, *chain.position_events]:
            policy_meta = self._governor_metadata_from_event(event)
            if policy_meta:
                break
        if not policy_meta:
            return None

        policy_key = policy_meta["policy_key"]
        version = policy_meta["version"]
        policy_id = policy_meta["policy_id"] or f"{policy_key}:{version}"

        active_policy = next(
            (
                policy for policy in self.governor_repo.load_active()
                if policy.key().as_id() == policy_key and policy.version == version
            ),
            None,
        )
        if active_policy is not None:
            return self._policy_to_snapshot(active_policy, policy_key, policy_id, policy_meta["observed_tier"], "active")

        for audit_event in reversed(self.governor_repo.load_audit_events(limit=5000, policy_key=policy_key)):
            for field_name in ("to_policy", "from_policy"):
                payload = audit_event.get(field_name)
                if not isinstance(payload, dict):
                    continue
                try:
                    policy = GovernorPolicy.from_dict(payload)
                except Exception:
                    continue
                if policy.key().as_id() != policy_key or policy.version != version:
                    continue
                return self._policy_to_snapshot(policy, policy_key, policy_id, policy_meta["observed_tier"], f"audit:{field_name}")

        asset_class, _, regime = policy_key.partition(":")
        return ReplayGovernorPolicySnapshot(
            policy_key=policy_key,
            policy_id=policy_id,
            version=version,
            asset_class=asset_class,
            regime=regime or "default",
            observed_tier=policy_meta["observed_tier"],
            source="event_metadata_only",
        )

    @staticmethod
    def _policy_to_snapshot(
        policy: GovernorPolicy,
        policy_key: str,
        policy_id: str,
        observed_tier: str,
        source: str,
    ) -> ReplayGovernorPolicySnapshot:
        return ReplayGovernorPolicySnapshot(
            policy_key=policy_key,
            policy_id=policy_id,
            version=policy.version,
            asset_class=policy.asset_class,
            regime=policy.regime,
            created_at=policy.created_at,
            observed_tier=observed_tier,
            tier_controls={k: v.to_dict() for k, v in policy.tier_controls.items()},
            metadata=dict(policy.metadata),
            source=source,
        )

    @staticmethod
    def _stress_event_mentions_symbol(event: ReplayTimelineEvent, symbol: str) -> bool:
        if event.event_type == "STRESS_ACTION":
            candidates = event.payload.get("candidates")
            if isinstance(candidates, list):
                for row in candidates:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("symbol", "")).strip() == symbol:
                        return True
        if event.event_type == "STRESS_EVALUATION":
            scenarios = event.payload.get("scenarios")
            if isinstance(scenarios, list):
                for scenario in scenarios:
                    if not isinstance(scenario, dict):
                        continue
                    for row in scenario.get("top_position_losses") or []:
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("symbol", "")).strip() == symbol:
                            return True
        return False

    @staticmethod
    def _stress_plan_candidate(
        event: ReplayTimelineEvent,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        candidates = event.payload.get("candidates")
        if not isinstance(candidates, list):
            return None
        for row in candidates:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol", "")).strip() == symbol:
                return dict(row)
        return None

    @staticmethod
    def _stress_plan_identity_from_event(event: ReplayTimelineEvent) -> Dict[str, Any]:
        return {
            "plan_id": str(event.payload.get("liquidation_plan_id", "")).strip(),
            "plan_epoch": int(event.payload.get("liquidation_plan_epoch", 0) or 0),
        }

    @staticmethod
    def _stress_plan_identity_from_chain(chain: ReplayChain) -> Dict[str, Any]:
        plan_id = ""
        plan_epoch = 0
        for event in [*chain.order_events, *chain.position_events]:
            metadata = event.payload.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            plan_id = str(metadata.get("liquidation_plan_id", "")).strip()
            plan_epoch = int(metadata.get("liquidation_plan_epoch", 0) or 0)
            if plan_id or plan_epoch:
                break
        return {"plan_id": plan_id, "plan_epoch": plan_epoch}

    @staticmethod
    def _event_identity(event: ReplayTimelineEvent) -> str:
        if event.hash:
            return f"hash:{event.hash}"
        return "|".join(
            [
                event.timestamp,
                event.event_type,
                event.symbol,
                event.asset_class,
                json.dumps(event.payload, sort_keys=True, default=str),
            ]
        )

    def _dedupe_events(self, events: Iterable[ReplayTimelineEvent]) -> List[ReplayTimelineEvent]:
        rows: Dict[str, ReplayTimelineEvent] = {}
        for event in events:
            rows[self._event_identity(event)] = event
        return sorted(rows.values(), key=lambda row: (_parse_ts(row.timestamp), row.hash))

    @staticmethod
    def _chain_events(chain: ReplayChain) -> List[ReplayTimelineEvent]:
        rows: List[ReplayTimelineEvent] = []
        if chain.signal_event is not None:
            rows.append(chain.signal_event)
        rows.extend(chain.risk_events)
        rows.extend(chain.order_events)
        rows.extend(chain.position_events)
        rows.extend(chain.stress_events)
        return rows

    @staticmethod
    def _stress_symbol_snapshot(
        event: ReplayTimelineEvent,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        scenarios = event.payload.get("scenarios")
        if not isinstance(scenarios, list):
            return None
        preferred_id = str(event.payload.get("worst_scenario_id", "")).strip()
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            scenario_rows = scenario.get("top_position_losses") or []
            if preferred_id and str(scenario.get("scenario_id", "")).strip() != preferred_id:
                continue
            for row in scenario_rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("symbol", "")).strip() == symbol:
                    return {
                        "scenario_id": str(scenario.get("scenario_id", "")).strip(),
                        "scenario_name": str(scenario.get("scenario_name", "")).strip(),
                        "portfolio_return": float(scenario.get("portfolio_return", 0.0) or 0.0),
                        "portfolio_pnl": float(scenario.get("portfolio_pnl", 0.0) or 0.0),
                        "position_pnl": float(row.get("pnl", 0.0) or 0.0),
                    }
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            for row in scenario.get("top_position_losses") or []:
                if not isinstance(row, dict):
                    continue
                if str(row.get("symbol", "")).strip() == symbol:
                    return {
                        "scenario_id": str(scenario.get("scenario_id", "")).strip(),
                        "scenario_name": str(scenario.get("scenario_name", "")).strip(),
                        "portfolio_return": float(scenario.get("portfolio_return", 0.0) or 0.0),
                        "portfolio_pnl": float(scenario.get("portfolio_pnl", 0.0) or 0.0),
                        "position_pnl": float(row.get("pnl", 0.0) or 0.0),
                    }
        return None

    @staticmethod
    def _new_chain(
        *,
        anchor: ReplayTimelineEvent,
        kind: str,
        signal_event: Optional[ReplayTimelineEvent],
        risk_events: Iterable[ReplayTimelineEvent],
    ) -> _ChainState:
        chain_id = anchor.hash or f"{anchor.event_type.lower()}-{anchor.timestamp}"
        return _ChainState(
            chain_id=chain_id,
            symbol=anchor.symbol,
            asset_class=anchor.asset_class or (signal_event.asset_class if signal_event else ""),
            chain_kind=kind,
            started_at=signal_event.timestamp if signal_event else anchor.timestamp,
            signal_event=signal_event,
            risk_events=list(risk_events),
        )

    def _finalize_chain(self, chain: _ChainState, *, fallback_event: Optional[ReplayTimelineEvent] = None) -> ReplayChain:
        terminal = (
            chain.position_events[-1]
            if chain.position_events
            else chain.order_events[-1]
            if chain.order_events
            else chain.risk_events[-1]
            if chain.risk_events
            else chain.signal_event
            or fallback_event
        )
        chain.completed_at = terminal.timestamp if terminal else chain.completed_at
        if chain.position_events:
            reason = str(chain.position_events[-1].payload.get("reason", "position_update"))
            chain.terminal_reason = reason
            if "exit" in reason:
                chain.final_status = "closed"
            else:
                chain.final_status = "filled"
        elif chain.order_events:
            last_order = chain.order_events[-1]
            lifecycle = str(last_order.payload.get("lifecycle", "")).lower()
            status = str(last_order.payload.get("status", "")).upper()
            chain.terminal_reason = status or lifecycle or "order_event"
            if status == "NO_TRADE":
                chain.final_status = "no_trade"
            elif status == "FILLED" or lifecycle == "filled":
                chain.final_status = "filled"
            else:
                chain.final_status = "open"
        elif chain.risk_events:
            last_risk = chain.risk_events[-1]
            chain.terminal_reason = str(last_risk.payload.get("reason", "risk_decision"))
            decision = str(last_risk.payload.get("decision", "")).lower()
            chain.final_status = "blocked" if decision == "blocked" else "allowed"
        else:
            chain.final_status = "observed"
            chain.terminal_reason = "signal_only"
        chain.governor_policy = self._resolve_governor_policy_snapshot(chain)
        return chain.to_schema()

    def _augment_chain_with_stress_context(
        self,
        chain: ReplayChain,
        *,
        symbol: str,
        stress_events: List[ReplayTimelineEvent],
    ) -> ReplayChain:
        if not stress_events:
            return chain

        started_at = _parse_ts(chain.started_at)
        completed_at = _parse_ts(chain.completed_at) if chain.completed_at else started_at
        last_exit_ts: Optional[datetime] = None
        executed_qty = 0.0
        remaining_qty = None
        stress_marker_seen = chain.chain_kind == "stress_liquidation_plan"
        for event in chain.order_events:
            payload = event.payload or {}
            metadata = payload.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            exit_reason = str(metadata.get("exit_reason", "")).lower()
            exit_type = str(metadata.get("exit_type", "")).lower()
            if str(payload.get("order_role", "")).lower() != "exit":
                continue
            if "stressunwind" in exit_reason or exit_type == "partial_reduce":
                stress_marker_seen = True
            lifecycle = str(payload.get("lifecycle", "")).lower()
            status = str(payload.get("status", "")).upper()
            if "stressunwind" not in exit_reason and exit_type != "partial_reduce":
                continue
            if lifecycle not in {"filled", "result"} and status not in {"FILLED", "SIMULATED"}:
                continue
            executed_qty += abs(float(payload.get("quantity", 0.0) or 0.0))
            last_exit_ts = _parse_ts(event.timestamp)
        for event in reversed(chain.position_events):
            metadata = event.payload.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            exit_reason = str(metadata.get("exit_reason", "")).lower()
            reason = str(event.payload.get("reason", "")).lower()
            if "stressunwind" in exit_reason or "partial_exit_fill" in reason:
                stress_marker_seen = True
            if "stressunwind" not in exit_reason and "partial_exit_fill" not in reason and "exit_fill" not in reason:
                continue
            remaining_qty = abs(float(event.payload.get("quantity", 0.0) or 0.0))
            break
        if not stress_marker_seen:
            return chain

        candidate_plans = [
            event for event in stress_events
            if event.event_type == "STRESS_ACTION" and self._stress_plan_candidate(event, symbol) is not None
        ]
        plan_event: Optional[ReplayTimelineEvent] = None
        chain_plan_identity = self._stress_plan_identity_from_chain(chain)
        if chain_plan_identity["plan_id"] or chain_plan_identity["plan_epoch"]:
            explicit = []
            for event in candidate_plans:
                identity = self._stress_plan_identity_from_event(event)
                if chain_plan_identity["plan_id"] and identity["plan_id"] == chain_plan_identity["plan_id"]:
                    explicit.append(event)
                elif (
                    not chain_plan_identity["plan_id"]
                    and chain_plan_identity["plan_epoch"] > 0
                    and identity["plan_epoch"] == chain_plan_identity["plan_epoch"]
                ):
                    explicit.append(event)
            if explicit:
                plan_event = explicit[-1]
        if plan_event is None and last_exit_ts is not None:
            prior = [
                event for event in candidate_plans
                if _parse_ts(event.timestamp) <= last_exit_ts
            ]
            plan_event = prior[-1] if prior else None
        if plan_event is None:
            prior = [event for event in candidate_plans if _parse_ts(event.timestamp) <= completed_at]
            plan_event = prior[-1] if prior else None
        if plan_event is None:
            near = [event for event in candidate_plans if _parse_ts(event.timestamp) >= started_at - timedelta(hours=2)]
            plan_event = near[-1] if near else None
        if plan_event is None:
            return chain

        plan_candidate = self._stress_plan_candidate(plan_event, symbol) or {}
        planned_qty = abs(float(plan_candidate.get("target_qty", 0.0) or 0.0))
        initial_qty = abs(float(plan_candidate.get("current_qty", 0.0) or 0.0))
        target_reduction_pct = float(plan_candidate.get("target_reduction_pct", 0.0) or 0.0)
        progress_pct = min(1.0, executed_qty / planned_qty) if planned_qty > 0 else 0.0
        if remaining_qty is None:
            remaining_qty = max(0.0, initial_qty - executed_qty) if initial_qty > 0 else 0.0

        plan_ts = _parse_ts(plan_event.timestamp)
        breach_candidates = [
            event
            for event in stress_events
            if event.event_type == "STRESS_EVALUATION" and _parse_ts(event.timestamp) <= plan_ts
        ]
        breach_event = breach_candidates[-1] if breach_candidates else None

        subsequent_evaluations = [
            event
            for event in stress_events
            if event.event_type == "STRESS_EVALUATION" and _parse_ts(event.timestamp) >= plan_ts
        ]
        latest_remaining_eval = subsequent_evaluations[-1] if subsequent_evaluations else breach_event
        remaining_snapshot = (
            self._stress_symbol_snapshot(latest_remaining_eval, symbol)
            if latest_remaining_eval is not None
            else None
        )

        linked_stress_events: List[ReplayTimelineEvent] = []
        if breach_event is not None:
            linked_stress_events.append(breach_event)
        linked_stress_events.append(plan_event)
        if latest_remaining_eval is not None and latest_remaining_eval.hash != plan_event.hash:
            linked_stress_events.append(latest_remaining_eval)
        chain.stress_events = linked_stress_events

        if planned_qty <= 0:
            status = "planned"
        elif progress_pct >= 0.999 or remaining_qty <= max(0.0, initial_qty - planned_qty) + 1e-9:
            status = "completed"
        elif executed_qty > 0:
            status = "in_progress"
        else:
            status = "planned"

        worst_scenario_id = str(plan_event.payload.get("worst_scenario_id", "")).strip()
        worst_scenario_name = str(plan_event.payload.get("worst_scenario_name", "")).strip()
        if remaining_snapshot:
            worst_scenario_id = remaining_snapshot.get("scenario_id") or worst_scenario_id
            worst_scenario_name = remaining_snapshot.get("scenario_name") or worst_scenario_name

        chain.liquidation_progress = ReplayLiquidationProgress(
            symbol=symbol,
            status=status,
            plan_id=str(plan_event.payload.get("liquidation_plan_id", "")).strip(),
            plan_epoch=int(plan_event.payload.get("liquidation_plan_epoch", 0) or 0),
            planned_reduction_qty=planned_qty,
            executed_reduction_qty=executed_qty,
            remaining_qty=float(remaining_qty or 0.0),
            progress_pct=progress_pct,
            target_reduction_pct=target_reduction_pct,
            initial_position_qty=initial_qty,
            expected_stress_pnl=float(plan_candidate.get("expected_stress_pnl", 0.0) or 0.0),
            remaining_stress_pnl=(
                float(remaining_snapshot["position_pnl"])
                if remaining_snapshot and remaining_snapshot.get("position_pnl") is not None
                else None
            ),
            remaining_stress_return=(
                float(remaining_snapshot["portfolio_return"])
                if remaining_snapshot and remaining_snapshot.get("portfolio_return") is not None
                else None
            ),
            worst_scenario_id=worst_scenario_id,
            worst_scenario_name=worst_scenario_name,
            breach_event=breach_event,
            plan_event=plan_event,
        )
        return chain

    def _build_stress_only_chain(
        self,
        *,
        symbol: str,
        asset_class: str,
        stress_events: List[ReplayTimelineEvent],
    ) -> Optional[ReplayChain]:
        candidate_plans = [
            event for event in stress_events
            if event.event_type == "STRESS_ACTION" and self._stress_plan_candidate(event, symbol) is not None
        ]
        if not candidate_plans:
            return None
        plan_event = candidate_plans[-1]
        plan_candidate = self._stress_plan_candidate(plan_event, symbol) or {}
        breach_candidates = [
            event
            for event in stress_events
            if event.event_type == "STRESS_EVALUATION" and _parse_ts(event.timestamp) <= _parse_ts(plan_event.timestamp)
        ]
        breach_event = breach_candidates[-1] if breach_candidates else None
        chain = ReplayChain(
            chain_id=plan_event.hash or f"stress-{symbol}-{plan_event.timestamp}",
            symbol=symbol,
            asset_class=asset_class,
            chain_kind="stress_liquidation_plan",
            started_at=breach_event.timestamp if breach_event is not None else plan_event.timestamp,
            completed_at=plan_event.timestamp,
            final_status="planned",
            terminal_reason=str(plan_event.payload.get("reason", "stress_liquidation_plan")),
            stress_events=([breach_event] if breach_event else []) + [plan_event],
        )
        return self._augment_chain_with_stress_context(chain, symbol=symbol, stress_events=stress_events)

    @staticmethod
    def _stress_plan_breach_event(
        *,
        plan_event: ReplayTimelineEvent,
        portfolio_stress: List[ReplayTimelineEvent],
    ) -> Optional[ReplayTimelineEvent]:
        plan_ts = _parse_ts(plan_event.timestamp)
        prior = [
            event
            for event in portfolio_stress
            if event.event_type == "STRESS_EVALUATION" and _parse_ts(event.timestamp) <= plan_ts
        ]
        return prior[-1] if prior else None

    def _reconstruct_chains(self, events: List[ReplayTimelineEvent]) -> List[ReplayChain]:
        chains: List[ReplayChain] = []
        pending_signal: Optional[ReplayTimelineEvent] = None
        pending_risks: List[ReplayTimelineEvent] = []
        active_chain: Optional[_ChainState] = None

        def flush_active(fallback_event: Optional[ReplayTimelineEvent] = None) -> None:
            nonlocal active_chain
            if active_chain is None:
                return
            chains.append(self._finalize_chain(active_chain, fallback_event=fallback_event))
            active_chain = None

        for event in events:
            if event.event_type == "SIGNAL_GENERATION":
                flush_active(fallback_event=event)
                pending_signal = event
                pending_risks = []
                continue

            if event.event_type == "RISK_DECISION":
                if active_chain is not None:
                    active_chain.risk_events.append(event)
                    if str(event.payload.get("decision", "")).lower() == "blocked":
                        flush_active(fallback_event=event)
                    continue

                pending_risks.append(event)
                if str(event.payload.get("decision", "")).lower() == "blocked":
                    anchor = event
                    blocked_chain = self._new_chain(
                        anchor=anchor,
                        kind="blocked_entry",
                        signal_event=pending_signal,
                        risk_events=pending_risks,
                    )
                    chains.append(self._finalize_chain(blocked_chain, fallback_event=event))
                    pending_signal = None
                    pending_risks = []
                continue

            if event.event_type == "ORDER_EXECUTION":
                order_role = str(event.payload.get("order_role", "order")).lower()
                lifecycle = str(event.payload.get("lifecycle", "")).lower()
                if lifecycle == "submitted":
                    flush_active(fallback_event=event)
                    active_chain = self._new_chain(
                        anchor=event,
                        kind=f"{order_role}_execution",
                        signal_event=pending_signal,
                        risk_events=pending_risks,
                    )
                    active_chain.order_events.append(event)
                    pending_signal = None
                    pending_risks = []
                else:
                    if active_chain is None:
                        active_chain = self._new_chain(
                            anchor=event,
                            kind=f"{order_role}_execution",
                            signal_event=pending_signal,
                            risk_events=pending_risks,
                        )
                        pending_signal = None
                        pending_risks = []
                    active_chain.order_events.append(event)
                    if str(event.payload.get("status", "")).upper() == "NO_TRADE":
                        flush_active(fallback_event=event)
                continue

            if event.event_type == "POSITION_UPDATE":
                if active_chain is None:
                    active_chain = self._new_chain(
                        anchor=event,
                        kind="position_update",
                        signal_event=pending_signal,
                        risk_events=pending_risks,
                    )
                    pending_signal = None
                    pending_risks = []
                active_chain.position_events.append(event)
                reason = str(event.payload.get("reason", "")).lower()
                if "fill" in reason or "exit" in reason:
                    flush_active(fallback_event=event)
                continue

        flush_active()
        return chains

    def inspect_symbol(
        self,
        *,
        symbol: str,
        limit: int = 500,
        days: int = 7,
        include_raw: bool = False,
    ) -> ReplayInspectionResponse:
        symbol_events = self.repo.load_symbol_events(symbol=symbol, limit=limit, days=days)
        portfolio_stress = self.repo.load_portfolio_stress_events(limit=max(limit * 2, 200), days=days)
        stress_events = [
            event for event in portfolio_stress if self._stress_event_mentions_symbol(event, symbol)
        ]
        chains = self._reconstruct_chains(symbol_events)
        asset_class = None
        if symbol_events:
            asset_class = symbol_events[-1].asset_class or None
        for index, chain in enumerate(chains):
            chains[index] = self._augment_chain_with_stress_context(
                chain,
                symbol=symbol,
                stress_events=stress_events,
            )
        if stress_events and not any(chain.liquidation_progress for chain in chains):
            synthetic = self._build_stress_only_chain(
                symbol=symbol,
                asset_class=asset_class or "EQUITY",
                stress_events=stress_events,
            )
            if synthetic is not None:
                chains.append(synthetic)
                chains.sort(key=lambda row: _parse_ts(row.started_at))
        latest_chain = chains[-1] if chains else None
        raw_events = sorted(
            [*symbol_events, *stress_events],
            key=lambda row: (_parse_ts(row.timestamp), row.hash),
        )
        summary = ReplayInspectionSummary(
            symbol=symbol,
            asset_class=asset_class,
            total_events=len(raw_events),
            total_chains=len(chains),
            blocked_chains=sum(1 for chain in chains if chain.final_status == "blocked"),
            filled_chains=sum(1 for chain in chains if chain.final_status == "filled"),
            open_chains=sum(1 for chain in chains if chain.final_status == "open"),
            stress_liquidation_chains=sum(1 for chain in chains if chain.liquidation_progress is not None),
            latest_event_at=raw_events[-1].timestamp if raw_events else None,
        )
        return ReplayInspectionResponse(
            mode="symbol",
            symbol=symbol,
            days=days,
            limit=limit,
            summary=summary,
            latest_chain=latest_chain,
            chains=chains,
            raw_events=raw_events if include_raw else [],
        )

    def inspect_plan(
        self,
        *,
        plan_id: str,
        limit: int = 500,
        days: int = 7,
        include_raw: bool = False,
    ) -> Optional[ReplayInspectionResponse]:
        normalized_plan_id = str(plan_id or "").strip()
        if not normalized_plan_id:
            return None

        portfolio_stress = self.repo.load_portfolio_stress_events(limit=max(limit * 4, 500), days=days)
        matching_actions = [
            event
            for event in portfolio_stress
            if event.event_type == "STRESS_ACTION"
            and str(event.payload.get("liquidation_plan_id", "")).strip() == normalized_plan_id
        ]
        if not matching_actions:
            return None

        plan_event = matching_actions[-1]
        breach_event = self._stress_plan_breach_event(plan_event=plan_event, portfolio_stress=portfolio_stress)
        plan_epoch = int(plan_event.payload.get("liquidation_plan_epoch", 0) or 0)
        candidates = plan_event.payload.get("candidates")
        candidate_rows = candidates if isinstance(candidates, list) else []
        candidate_symbols = list(
            dict.fromkeys(
                str(row.get("symbol", "")).strip()
                for row in candidate_rows
                if isinstance(row, dict) and str(row.get("symbol", "")).strip()
            )
        )

        plan_chains: List[ReplayChain] = []
        raw_pool: List[ReplayTimelineEvent] = []
        asset_class: Optional[str] = None
        for symbol in candidate_symbols:
            symbol_inspection = self.inspect_symbol(
                symbol=symbol,
                limit=limit,
                days=days,
                include_raw=False,
            )
            matching_chains = [
                chain
                for chain in symbol_inspection.chains
                if chain.liquidation_progress is not None
                and chain.liquidation_progress.plan_id == normalized_plan_id
            ]
            if not matching_chains and plan_epoch > 0:
                matching_chains = [
                    chain
                    for chain in symbol_inspection.chains
                    if chain.liquidation_progress is not None
                    and chain.liquidation_progress.plan_epoch == plan_epoch
                ]
            plan_chains.extend(matching_chains)
            if asset_class is None and symbol_inspection.summary.asset_class:
                asset_class = symbol_inspection.summary.asset_class
            for chain in matching_chains:
                raw_pool.extend(self._chain_events(chain))

        raw_pool.extend([event for event in (breach_event, plan_event) if event is not None])
        raw_events = self._dedupe_events(raw_pool)
        plan_chains.sort(key=lambda row: (_parse_ts(row.started_at), row.chain_id))
        latest_chain = plan_chains[-1] if plan_chains else None

        completed_symbols = 0
        in_progress_symbols = 0
        planned_symbols = 0
        for chain in plan_chains:
            status = str(chain.liquidation_progress.status if chain.liquidation_progress else "").lower()
            if status == "completed":
                completed_symbols += 1
            elif status == "in_progress":
                in_progress_symbols += 1
            elif status:
                planned_symbols += 1

        summary = ReplayInspectionSummary(
            symbol=f"PLAN:{normalized_plan_id}",
            asset_class=asset_class or "PORTFOLIO",
            total_events=len(raw_events),
            total_chains=len(plan_chains),
            blocked_chains=sum(1 for chain in plan_chains if chain.final_status == "blocked"),
            filled_chains=sum(1 for chain in plan_chains if chain.final_status == "filled"),
            open_chains=sum(1 for chain in plan_chains if chain.final_status == "open"),
            stress_liquidation_chains=sum(1 for chain in plan_chains if chain.liquidation_progress is not None),
            latest_event_at=raw_events[-1].timestamp if raw_events else plan_event.timestamp,
        )

        return ReplayInspectionResponse(
            mode="plan",
            symbol=f"PLAN:{normalized_plan_id}",
            days=days,
            limit=limit,
            summary=summary,
            latest_chain=latest_chain,
            chains=plan_chains,
            raw_events=raw_events if include_raw else [],
            plan_audit=ReplayPlanAudit(
                plan_id=normalized_plan_id,
                plan_epoch=plan_epoch,
                started_at=breach_event.timestamp if breach_event is not None else plan_event.timestamp,
                worst_scenario_id=str(plan_event.payload.get("worst_scenario_id", "")).strip(),
                worst_scenario_name=str(plan_event.payload.get("worst_scenario_name", "")).strip(),
                candidate_symbols=candidate_symbols,
                completed_symbols=completed_symbols,
                in_progress_symbols=in_progress_symbols,
                planned_symbols=planned_symbols,
                breach_event=breach_event,
                plan_event=plan_event,
            ),
        )
