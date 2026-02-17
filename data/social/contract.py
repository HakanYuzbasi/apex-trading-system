"""
data/social/contract.py

Build and persist normalized social-risk input contract consumed by runtime.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from data.social.adapters import SourceSnapshot, default_social_adapters
from data.social.validator import SocialInputValidationReport, validate_social_risk_inputs


def build_social_risk_inputs(
    *,
    data_dir: Path,
    now: Optional[datetime] = None,
    freshness_sla_seconds: int = 1800,
) -> Tuple[Dict[str, object], SocialInputValidationReport]:
    now_dt = now or datetime.utcnow()
    snapshots: Dict[str, SourceSnapshot] = {}
    for adapter in default_social_adapters(data_dir):
        snap = adapter.fetch(now=now_dt, freshness_sla_seconds=freshness_sla_seconds)
        snapshots[snap.source] = snap

    sources = {
        source: snap.to_source_entry(freshness_sla_seconds=freshness_sla_seconds)
        for source, snap in snapshots.items()
    }
    platforms = {
        source: snap.to_platform_signal()
        for source, snap in snapshots.items()
        if snap.quality_status in {"ok", "degraded", "stale"}
    }
    payload: Dict[str, object] = {
        "schema_version": "1.0",
        "generated_at": now_dt.isoformat(),
        "freshness_sla_seconds": int(freshness_sla_seconds),
        "sources": sources,
        "platforms": platforms,
        "asset_classes": {},
        "prediction_events": [],
    }

    report = validate_social_risk_inputs(
        payload,
        now=now_dt,
        freshness_sla_seconds=freshness_sla_seconds,
    )
    payload["validation"] = report.summary()
    return payload, report


def write_social_risk_inputs(
    *,
    data_dir: Path,
    output_path: Optional[Path] = None,
    now: Optional[datetime] = None,
    freshness_sla_seconds: int = 1800,
) -> Tuple[Path, Dict[str, object], SocialInputValidationReport]:
    payload, report = build_social_risk_inputs(
        data_dir=data_dir,
        now=now,
        freshness_sla_seconds=freshness_sla_seconds,
    )
    path = output_path or (Path(data_dir) / "social_risk_inputs.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    tmp.replace(path)
    return path, payload, report
