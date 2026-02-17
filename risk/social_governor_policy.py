"""
risk/social_governor_policy.py

Versioned social-governor policy snapshots and runtime resolver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SocialGovernorPolicy:
    asset_class: str
    regime: str
    version: str
    reduce_threshold: float
    block_threshold: float
    verified_event_weight: float
    verified_event_probability_floor: float
    max_probability_divergence: float
    max_source_disagreement: float
    min_independent_sources: int = 2
    minimum_market_probability: float = 0.05
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, object] = field(default_factory=dict)

    def key(self) -> Tuple[str, str]:
        return (str(self.asset_class).upper(), str(self.regime).lower() or "default")

    def to_dict(self) -> Dict[str, object]:
        return {
            "asset_class": str(self.asset_class).upper(),
            "regime": str(self.regime).lower() or "default",
            "version": str(self.version),
            "reduce_threshold": float(self.reduce_threshold),
            "block_threshold": float(self.block_threshold),
            "verified_event_weight": float(self.verified_event_weight),
            "verified_event_probability_floor": float(self.verified_event_probability_floor),
            "max_probability_divergence": float(self.max_probability_divergence),
            "max_source_disagreement": float(self.max_source_disagreement),
            "min_independent_sources": int(self.min_independent_sources),
            "minimum_market_probability": float(self.minimum_market_probability),
            "created_at": str(self.created_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SocialGovernorPolicy":
        return cls(
            asset_class=str(payload.get("asset_class", "GLOBAL")).upper(),
            regime=str(payload.get("regime", "default")).lower() or "default",
            version=str(payload.get("version", "unknown")),
            reduce_threshold=float(payload.get("reduce_threshold", 0.60)),
            block_threshold=float(payload.get("block_threshold", 0.85)),
            verified_event_weight=float(payload.get("verified_event_weight", 0.30)),
            verified_event_probability_floor=float(payload.get("verified_event_probability_floor", 0.55)),
            max_probability_divergence=float(payload.get("max_probability_divergence", 0.15)),
            max_source_disagreement=float(payload.get("max_source_disagreement", 0.20)),
            min_independent_sources=int(payload.get("min_independent_sources", 2)),
            minimum_market_probability=float(payload.get("minimum_market_probability", 0.05)),
            created_at=str(payload.get("created_at", datetime.utcnow().isoformat())),
            metadata=dict(payload.get("metadata", {})),
        )


class SocialGovernorPolicyRepository:
    """
    File-backed repository:
    - social_shock_policy_<version>.json (immutable snapshots)
    - social_shock_active.json (active version pointer + policy set)
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.active_file = self.directory / "social_shock_active.json"

    def save_snapshot(
        self,
        *,
        version: str,
        policies: List[SocialGovernorPolicy],
        metadata: Optional[Dict[str, object]] = None,
    ) -> Path:
        payload = {
            "policy_type": "social_shock",
            "version": str(version),
            "created_at": datetime.utcnow().isoformat(),
            "metadata": dict(metadata or {}),
            "policies": [policy.to_dict() for policy in policies],
        }
        path = self.directory / f"social_shock_policy_{version}.json"
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(path)
        return path

    def activate_snapshot(self, snapshot_path: Path) -> None:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tmp = self.active_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self.active_file)

    def load_active(self) -> Tuple[str, Dict[Tuple[str, str], SocialGovernorPolicy]]:
        if not self.active_file.exists():
            return "runtime-default", {}
        try:
            with open(self.active_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            version = str(payload.get("version", "runtime-default"))
            rows = payload.get("policies", [])
            policies = {}
            if isinstance(rows, list):
                for item in rows:
                    if isinstance(item, dict):
                        policy = SocialGovernorPolicy.from_dict(item)
                        policies[policy.key()] = policy
            return version, policies
        except Exception:
            return "runtime-default", {}

    def resolve(
        self,
        *,
        asset_class: str,
        regime: str,
        active_policies: Dict[Tuple[str, str], SocialGovernorPolicy],
    ) -> Optional[SocialGovernorPolicy]:
        asset = str(asset_class).upper().strip() or "GLOBAL"
        reg = str(regime).lower().strip() or "default"
        for key in ((asset, reg), (asset, "default"), ("GLOBAL", "default")):
            if key in active_policies:
                return active_policies[key]
        return None
