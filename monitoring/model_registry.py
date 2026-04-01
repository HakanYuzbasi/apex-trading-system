"""
monitoring/model_registry.py — ML Model Registry & Versioning

Tracks trained model versions with performance metadata.  Supports:
  - Registering new model versions (champion/challenger concept)
  - Automatic rollback when a new version degrades below threshold
  - Snapshot API for dashboard visibility

The registry is purely metadata — actual model weights are managed by
the signal generators.  The registry tracks *which* version is live and
*why* decisions were made.

Usage:
    from monitoring.model_registry import ModelRegistry
    reg = ModelRegistry(state_dir=Path("data/model_registry"))

    # After training a new version:
    vid = reg.register_version("god_level", metrics={"ic": 0.12, "sharpe": 1.4})
    if reg.should_promote(vid):
        reg.promote(vid)       # becomes champion
    else:
        reg.rollback()         # revert to previous champion
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Thresholds for auto-promote/rollback ──────────────────────────────────────
IC_PROMOTE_DELTA   =  0.005   # new IC must beat champion by at least this
IC_ROLLBACK_THRESH = -0.010   # new IC below (champion - this) triggers rollback
SHARPE_MIN         =  0.20    # absolute minimum Sharpe to keep model live
MAX_VERSIONS_KEPT  =  10      # prune oldest beyond this

_STATUS_CHAMPION   = "champion"
_STATUS_CHALLENGER = "challenger"
_STATUS_RETIRED    = "retired"
_STATUS_ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    version_id: str
    model_name: str
    registered_at: float
    metrics: Dict[str, float]           # ic, sharpe, win_rate, hit_rate, …
    status: str = _STATUS_CHALLENGER    # champion | challenger | retired | rolled_back
    promoted_at: Optional[float] = None
    retired_at: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Round floats for readability
        d["metrics"] = {k: round(v, 6) for k, v in d["metrics"].items()}
        return d


@dataclass
class PromotionEvent:
    ts: float
    version_id: str
    model_name: str
    action: str                 # "promote" | "rollback" | "register"
    reason: str
    champion_ic_before: float
    champion_ic_after: float

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelRegistry:
    """
    Lightweight model version registry.

    Key design choices:
    - One champion per model_name at any time
    - Challenger is auto-evaluated on register; promote if better by threshold
    - Rollback triggers when champion degrades by IC_ROLLBACK_THRESH
    - Full version history kept (up to MAX_VERSIONS_KEPT per model)
    """

    def __init__(self, state_dir: Optional[Path] = None):
        self._versions: Dict[str, ModelVersion] = {}      # version_id → ModelVersion
        self._events: List[PromotionEvent] = []            # audit trail
        self._champions: Dict[str, str] = {}              # model_name → version_id
        self._seq: int = 0                                 # per-instance counter for unique IDs

        if state_dir:
            self._state_dir = Path(state_dir)
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load()
        else:
            self._state_dir = None

    # ── Write API ────────────────────────────────────────────────────────────

    def register_version(
        self,
        model_name: str,
        metrics: Dict[str, float],
        notes: str = "",
    ) -> str:
        """
        Register a new model version and return its version_id.

        The version starts as 'challenger'.  Call should_promote() to check
        whether it should become champion, then promote() or rollback().
        """
        ts = time.time()
        self._seq += 1
        version_id = f"{model_name}_{int(ts * 1000)}_{self._seq}"
        v = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            registered_at=ts,
            metrics=dict(metrics),
            status=_STATUS_CHALLENGER,
            notes=notes,
        )
        self._versions[version_id] = v
        self._record_event(
            version_id=version_id,
            model_name=model_name,
            action="register",
            reason="new version registered",
            before_ic=self._champion_ic(model_name),
            after_ic=metrics.get("ic", 0.0),
        )
        self._prune_old_versions(model_name)
        self._save()
        logger.info("ModelRegistry: registered %s ic=%.4f", version_id, metrics.get("ic", 0.0))
        return version_id

    def should_promote(self, version_id: str) -> bool:
        """
        Return True if this version should replace the current champion.

        Rules:
        1. No current champion → always promote
        2. New IC must exceed champion IC by IC_PROMOTE_DELTA
        3. New Sharpe must be >= SHARPE_MIN
        """
        v = self._versions.get(version_id)
        if v is None:
            return False

        current_champ_id = self._champions.get(v.model_name)
        if current_champ_id is None:
            return True  # first version — always promote

        champ = self._versions.get(current_champ_id)
        if champ is None:
            return True

        new_ic    = v.metrics.get("ic", 0.0)
        champ_ic  = champ.metrics.get("ic", 0.0)
        new_sharpe = v.metrics.get("sharpe", 0.0)

        if new_sharpe < SHARPE_MIN:
            logger.info("ModelRegistry: %s Sharpe %.3f < %.3f — not promoting", version_id, new_sharpe, SHARPE_MIN)
            return False
        if new_ic < champ_ic + IC_PROMOTE_DELTA:
            logger.info(
                "ModelRegistry: %s IC %.4f not > champion IC %.4f + delta %.4f — not promoting",
                version_id, new_ic, champ_ic, IC_PROMOTE_DELTA,
            )
            return False
        return True

    def promote(self, version_id: str) -> None:
        """Promote version_id to champion, retiring the current one."""
        v = self._versions.get(version_id)
        if v is None:
            logger.warning("ModelRegistry.promote: unknown version %s", version_id)
            return

        old_champ_id = self._champions.get(v.model_name)
        old_ic = self._champion_ic(v.model_name)

        # Retire current champion
        if old_champ_id and old_champ_id in self._versions:
            self._versions[old_champ_id].status = _STATUS_RETIRED
            self._versions[old_champ_id].retired_at = time.time()

        # Promote new version
        v.status = _STATUS_CHAMPION
        v.promoted_at = time.time()
        self._champions[v.model_name] = version_id

        self._record_event(
            version_id=version_id,
            model_name=v.model_name,
            action="promote",
            reason=f"IC {v.metrics.get('ic', 0.0):.4f} beats champion {old_ic:.4f}",
            before_ic=old_ic,
            after_ic=v.metrics.get("ic", 0.0),
        )
        self._save()
        logger.info("ModelRegistry: promoted %s → champion (replaced %s)", version_id, old_champ_id)

    def rollback(self, model_name: str, reason: str = "performance degradation") -> Optional[str]:
        """
        Roll back to the previous champion (the most recent RETIRED version).

        Returns the rolled-back version_id, or None if no prior version exists.
        """
        current_champ_id = self._champions.get(model_name)
        current_ic = self._champion_ic(model_name)

        # Find most recently retired version for this model
        retired = [
            v for v in self._versions.values()
            if v.model_name == model_name and v.status == _STATUS_RETIRED
        ]
        if not retired:
            logger.warning("ModelRegistry.rollback: no retired versions for %s", model_name)
            return None

        prev = max(retired, key=lambda v: v.registered_at)

        # Mark current champion as rolled_back
        if current_champ_id and current_champ_id in self._versions:
            self._versions[current_champ_id].status = _STATUS_ROLLED_BACK
            self._versions[current_champ_id].retired_at = time.time()

        # Restore previous
        prev.status = _STATUS_CHAMPION
        prev.promoted_at = time.time()
        self._champions[model_name] = prev.version_id

        self._record_event(
            version_id=prev.version_id,
            model_name=model_name,
            action="rollback",
            reason=reason,
            before_ic=current_ic,
            after_ic=prev.metrics.get("ic", 0.0),
        )
        self._save()
        logger.warning("ModelRegistry: rolled back %s → %s (%s)", model_name, prev.version_id, reason)
        return prev.version_id

    def should_rollback(self, model_name: str, live_ic: float) -> bool:
        """
        Return True if the live IC has degraded enough to trigger rollback.
        Only meaningful when we have a previous version to fall back to.
        """
        champ_ic = self._champion_ic(model_name)
        if live_ic < champ_ic - abs(IC_ROLLBACK_THRESH):
            retired = [v for v in self._versions.values()
                       if v.model_name == model_name and v.status == _STATUS_RETIRED]
            return len(retired) > 0
        return False

    # ── Query API ────────────────────────────────────────────────────────────

    def get_champion(self, model_name: str) -> Optional[ModelVersion]:
        vid = self._champions.get(model_name)
        return self._versions.get(vid) if vid else None

    def get_versions(self, model_name: str) -> List[ModelVersion]:
        """Return all versions for a model, newest first."""
        vs = [v for v in self._versions.values() if v.model_name == model_name]
        return sorted(vs, key=lambda v: v.registered_at, reverse=True)

    def get_snapshot(self) -> Dict:
        """Return serialisable dashboard snapshot."""
        model_names = sorted({v.model_name for v in self._versions.values()})
        models_summary = {}
        for name in model_names:
            champ = self.get_champion(name)
            versions = self.get_versions(name)
            models_summary[name] = {
                "champion_id": champ.version_id if champ else None,
                "champion_ic": round(champ.metrics.get("ic", 0.0), 6) if champ else None,
                "champion_sharpe": round(champ.metrics.get("sharpe", 0.0), 4) if champ else None,
                "champion_promoted_at": champ.promoted_at if champ else None,
                "total_versions": len(versions),
                "versions": [v.to_dict() for v in versions[:5]],  # last 5
            }
        recent_events = [e.to_dict() for e in reversed(self._events[-20:])]
        return {
            "available": True,
            "total_models": len(model_names),
            "total_versions": len(self._versions),
            "models": models_summary,
            "recent_events": recent_events,
            "ic_promote_delta": IC_PROMOTE_DELTA,
            "ic_rollback_thresh": IC_ROLLBACK_THRESH,
            "sharpe_min": SHARPE_MIN,
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _champion_ic(self, model_name: str) -> float:
        champ = self.get_champion(model_name)
        return champ.metrics.get("ic", 0.0) if champ else 0.0

    def _record_event(
        self,
        version_id: str,
        model_name: str,
        action: str,
        reason: str,
        before_ic: float,
        after_ic: float,
    ) -> None:
        evt = PromotionEvent(
            ts=time.time(),
            version_id=version_id,
            model_name=model_name,
            action=action,
            reason=reason,
            champion_ic_before=round(before_ic, 6),
            champion_ic_after=round(after_ic, 6),
        )
        self._events.append(evt)
        if len(self._events) > 500:
            self._events = self._events[-500:]

    def _prune_old_versions(self, model_name: str) -> None:
        versions = self.get_versions(model_name)
        if len(versions) > MAX_VERSIONS_KEPT:
            to_remove = versions[MAX_VERSIONS_KEPT:]
            for v in to_remove:
                if v.status not in (_STATUS_CHAMPION,):
                    self._versions.pop(v.version_id, None)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._state_dir is None:
            return
        try:
            state = {
                "versions": {k: v.to_dict() for k, v in self._versions.items()},
                "champions": self._champions,
                "events": [e.to_dict() for e in self._events[-200:]],
            }
            p = self._state_dir / "model_registry.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception as exc:
            logger.debug("ModelRegistry: save failed: %s", exc)

    def _load(self) -> None:
        if self._state_dir is None:
            return
        try:
            p = self._state_dir / "model_registry.json"
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            for vid, vd in raw.get("versions", {}).items():
                self._versions[vid] = ModelVersion(
                    version_id=vd["version_id"],
                    model_name=vd["model_name"],
                    registered_at=float(vd["registered_at"]),
                    metrics=vd.get("metrics", {}),
                    status=vd.get("status", _STATUS_CHALLENGER),
                    promoted_at=vd.get("promoted_at"),
                    retired_at=vd.get("retired_at"),
                    notes=vd.get("notes", ""),
                )
            self._champions = raw.get("champions", {})
            for ed in raw.get("events", []):
                self._events.append(PromotionEvent(
                    ts=float(ed["ts"]),
                    version_id=ed["version_id"],
                    model_name=ed["model_name"],
                    action=ed["action"],
                    reason=ed["reason"],
                    champion_ic_before=float(ed["champion_ic_before"]),
                    champion_ic_after=float(ed["champion_ic_after"]),
                ))
            logger.info("ModelRegistry: loaded %d versions, %d models", len(self._versions), len(self._champions))
        except Exception as exc:
            logger.debug("ModelRegistry: load failed: %s", exc)
