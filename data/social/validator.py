"""
data/social/validator.py

Validation for normalized social risk ingestion payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional


REQUIRED_SOURCES = ("X", "TIKTOK", "INSTAGRAM", "YOUTUBE")


def _parse_dt(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _normalize_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class SocialInputValidationIssue:
    level: str
    code: str
    message: str


@dataclass
class SocialInputValidationReport:
    valid: bool
    has_usable_feeds: bool
    source_status: Dict[str, str] = field(default_factory=dict)
    usable_sources: List[str] = field(default_factory=list)
    stale_sources: List[str] = field(default_factory=list)
    warnings: List[SocialInputValidationIssue] = field(default_factory=list)
    errors: List[SocialInputValidationIssue] = field(default_factory=list)

    def summary(self) -> Dict[str, object]:
        return {
            "valid": bool(self.valid),
            "has_usable_feeds": bool(self.has_usable_feeds),
            "source_status": dict(self.source_status),
            "usable_sources": list(self.usable_sources),
            "stale_sources": list(self.stale_sources),
            "warnings": [i.__dict__ for i in self.warnings],
            "errors": [i.__dict__ for i in self.errors],
        }


def validate_social_risk_inputs(
    payload: Mapping[str, object],
    *,
    now: Optional[datetime] = None,
    required_sources: Iterable[str] = REQUIRED_SOURCES,
    freshness_sla_seconds: int = 1800,
) -> SocialInputValidationReport:
    report = SocialInputValidationReport(valid=True, has_usable_feeds=False)
    now_dt = _normalize_dt(now or datetime.utcnow())

    if not isinstance(payload, Mapping):
        report.valid = False
        report.errors.append(
            SocialInputValidationIssue(
                level="error",
                code="payload_not_mapping",
                message="social risk payload must be a JSON object",
            )
        )
        return report

    if not _parse_dt(payload.get("generated_at")):
        report.valid = False
        report.errors.append(
            SocialInputValidationIssue(
                level="error",
                code="missing_generated_at",
                message="generated_at must be an ISO-8601 timestamp",
            )
        )

    sources = payload.get("sources", {})
    if not isinstance(sources, Mapping):
        report.valid = False
        report.errors.append(
            SocialInputValidationIssue(
                level="error",
                code="missing_sources",
                message="sources section must exist and be an object",
            )
        )
        return report

    platforms = payload.get("platforms", {})
    if not isinstance(platforms, Mapping):
        report.valid = False
        report.errors.append(
            SocialInputValidationIssue(
                level="error",
                code="missing_platforms",
                message="platforms section must exist and be an object",
            )
        )

    allowed_status = {"ok", "degraded", "stale", "missing"}
    fresh_limit = max(60, int(freshness_sla_seconds))
    for source in required_sources:
        name = str(source).upper()
        row = sources.get(name)
        if not isinstance(row, Mapping):
            report.valid = False
            report.errors.append(
                SocialInputValidationIssue(
                    level="error",
                    code="source_missing",
                    message=f"missing source entry for {name}",
                )
            )
            report.source_status[name] = "missing"
            continue

        quality = row.get("quality", {})
        if not isinstance(quality, Mapping):
            quality = {}
        status = str(quality.get("status", "missing")).lower()
        if status not in allowed_status:
            report.valid = False
            report.errors.append(
                SocialInputValidationIssue(
                    level="error",
                    code="invalid_source_status",
                    message=f"source {name} has invalid status '{status}'",
                )
            )
            status = "missing"
        report.source_status[name] = status

        fresh_ts = _parse_dt(row.get("freshness_ts"))
        if fresh_ts is None:
            report.warnings.append(
                SocialInputValidationIssue(
                    level="warning",
                    code="freshness_missing",
                    message=f"source {name} missing freshness_ts",
                )
            )
        else:
            fresh_ts = _normalize_dt(fresh_ts)
            age = max(0.0, (now_dt - fresh_ts).total_seconds())
            if age > fresh_limit and status in {"ok", "degraded"}:
                report.warnings.append(
                    SocialInputValidationIssue(
                        level="warning",
                        code="source_stale",
                        message=f"source {name} is stale by {int(age)}s",
                    )
                )
                report.stale_sources.append(name)
                status = "stale"
                report.source_status[name] = status

        if status in {"ok", "degraded"}:
            report.usable_sources.append(name)

        if isinstance(platforms, Mapping):
            platform_signal = platforms.get(name)
            if status != "missing" and not isinstance(platform_signal, Mapping):
                report.warnings.append(
                    SocialInputValidationIssue(
                        level="warning",
                        code="platform_signal_missing",
                        message=f"platforms.{name} is missing while source status is {status}",
                    )
                )
            if isinstance(platform_signal, Mapping):
                for metric_key, low, high in (
                    ("attention_z", -8.0, 12.0),
                    ("sentiment_score", -1.0, 1.0),
                    ("confidence", 0.0, 1.0),
                ):
                    metric = platform_signal.get(metric_key)
                    try:
                        metric_v = float(metric)
                    except Exception:
                        report.warnings.append(
                            SocialInputValidationIssue(
                                level="warning",
                                code="platform_metric_invalid",
                                message=f"platforms.{name}.{metric_key} must be numeric",
                            )
                        )
                        continue
                    if metric_v < low or metric_v > high:
                        report.warnings.append(
                            SocialInputValidationIssue(
                                level="warning",
                                code="platform_metric_out_of_range",
                                message=f"platforms.{name}.{metric_key}={metric_v} outside [{low}, {high}]",
                            )
                        )

    report.has_usable_feeds = any(status in {"ok", "degraded"} for status in report.source_status.values())
    if not report.has_usable_feeds:
        report.warnings.append(
            SocialInputValidationIssue(
                level="warning",
                code="all_sources_unusable",
                message="all social feeds are missing/stale; runtime should fail-open for entry blocking",
            )
        )
    return report
