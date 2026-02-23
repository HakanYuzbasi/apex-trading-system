"""
data/social/adapters.py

Social feed adapters for cross-platform ingestion.
Each adapter returns a normalized snapshot with quality metadata.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
class SourceSnapshot:
    source: str
    fetched_at: str
    freshness_ts: str
    attention_z: float
    sentiment_score: float
    confidence: float
    sample_size: int
    quality_status: str
    quality_score: float
    quality_flags: List[str]
    adapter_mode: str
    raw_payload: Dict[str, object]

    def to_source_entry(self, freshness_sla_seconds: int) -> Dict[str, object]:
        fetched = _normalize_dt(_parse_dt(self.fetched_at) or datetime.utcnow())
        fresh_until = fetched + timedelta(seconds=max(60, int(freshness_sla_seconds)))
        return {
            "source": self.source,
            "adapter_mode": self.adapter_mode,
            "fetched_at": self.fetched_at,
            "freshness_ts": self.freshness_ts,
            "fresh_until": fresh_until.isoformat(),
            "sample_size": int(self.sample_size),
            "quality": {
                "status": self.quality_status,
                "score": float(self.quality_score),
                "flags": list(self.quality_flags),
            },
        }

    def to_platform_signal(self) -> Dict[str, object]:
        return {
            "attention_z": float(self.attention_z),
            "sentiment_score": float(self.sentiment_score),
            "confidence": float(self.confidence),
            "freshness_ts": self.freshness_ts,
            "quality_flags": list(self.quality_flags),
        }


class SocialSourceAdapter:
    source_name = "UNKNOWN"
    env_endpoint_key = ""
    env_token_key = ""
    default_local_name = ""

    def __init__(self, data_dir: Path, timeout_seconds: float = 3.0):
        self.data_dir = Path(data_dir)
        self.timeout_seconds = float(timeout_seconds)

    def fetch(self, now: Optional[datetime] = None, freshness_sla_seconds: int = 1800) -> SourceSnapshot:
        now_dt = _normalize_dt(now or datetime.utcnow())
        payload, mode, flags = self._load_payload()
        if payload is None:
            return SourceSnapshot(
                source=self.source_name,
                fetched_at=now_dt.isoformat(),
                freshness_ts=now_dt.isoformat(),
                attention_z=0.0,
                sentiment_score=0.0,
                confidence=0.0,
                sample_size=0,
                quality_status="missing",
                quality_score=0.0,
                quality_flags=sorted(set(flags + ["source_unavailable"])),
                adapter_mode=mode,
                raw_payload={},
            )

        normalized, normalize_flags = self._normalize_payload(payload)
        flags.extend(normalize_flags)
        fetched_at = _normalize_dt(_parse_dt(payload.get("fetched_at") or payload.get("timestamp")) or now_dt)
        freshness_ts = _normalize_dt(_parse_dt(
            payload.get("freshness_ts")
            or payload.get("observed_at")
            or payload.get("observed_timestamp")
            or payload.get("captured_at")
            or payload.get("fetched_at")
            or payload.get("timestamp")
        ) or fetched_at)
        age_seconds = max(0.0, (now_dt - freshness_ts).total_seconds())

        if age_seconds > max(60.0, float(freshness_sla_seconds)):
            flags.append("stale_data")
            quality_status = "stale"
        elif flags:
            quality_status = "degraded"
        else:
            quality_status = "ok"

        confidence = float(normalized.get("confidence", 0.0) or 0.0)
        if quality_status == "stale":
            confidence *= 0.35
        elif quality_status == "degraded":
            confidence *= 0.70

        quality_score = _clamp(
            confidence * (0.6 if quality_status == "ok" else 0.35) + (0.4 if quality_status == "ok" else 0.2),
            0.0,
            1.0,
        )
        return SourceSnapshot(
            source=self.source_name,
            fetched_at=fetched_at.isoformat(),
            freshness_ts=freshness_ts.isoformat(),
            attention_z=float(normalized.get("attention_z", 0.0) or 0.0),
            sentiment_score=float(normalized.get("sentiment_score", 0.0) or 0.0),
            confidence=float(_clamp(confidence, 0.0, 1.0)),
            sample_size=int(normalized.get("sample_size", 0) or 0),
            quality_status=quality_status,
            quality_score=quality_score,
            quality_flags=sorted(set(flags)),
            adapter_mode=mode,
            raw_payload=payload,
        )

    def _load_payload(self) -> Tuple[Optional[Dict[str, object]], str, List[str]]:
        endpoint = os.getenv(self.env_endpoint_key, "").strip() if self.env_endpoint_key else ""
        if endpoint:
            data, flags = self._fetch_http_payload(endpoint)
            if data is not None:
                return data, "http", flags
            return None, "http", flags

        local_override = os.getenv(f"APEX_SOCIAL_{self.source_name}_FILE", "").strip()
        local_path = Path(local_override) if local_override else (self.data_dir / "social" / self.default_local_name)
        if local_path.exists():
            try:
                loaded = json.loads(local_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded, "file", []
                return None, "file", ["invalid_payload_type"]
            except Exception:
                return None, "file", ["invalid_json"]

        return None, "none", ["missing_source_file"]

    def _fetch_http_payload(self, endpoint: str) -> Tuple[Optional[Dict[str, object]], List[str]]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        token = os.getenv(self.env_token_key, "").strip() if self.env_token_key else ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = Request(endpoint, headers=headers, method="GET")
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            loaded = json.loads(body)
            if isinstance(loaded, dict):
                return loaded, []
            return None, ["invalid_payload_type"]
        except URLError:
            return None, ["endpoint_unreachable"]
        except Exception:
            return None, ["fetch_error"]

    def _normalize_payload(self, payload: Dict[str, object]) -> Tuple[Dict[str, object], List[str]]:
        flags: List[str] = []
        attention, attention_ok = self._coerce_attention_z(payload)
        sentiment, sentiment_ok = self._coerce_sentiment(payload)
        confidence, confidence_ok = self._coerce_confidence(payload)
        sample_size = self._coerce_sample_size(payload)

        if not attention_ok:
            flags.append("attention_missing")
        if not sentiment_ok:
            flags.append("sentiment_missing")
        if not confidence_ok:
            flags.append("confidence_missing")
        if sample_size <= 0:
            flags.append("sample_size_missing")

        if sample_size > 0:
            coverage_factor = _clamp(sample_size / 300.0, 0.25, 1.0)
            confidence = _clamp(confidence * coverage_factor, 0.0, 1.0)
        return (
            {
                "attention_z": attention,
                "sentiment_score": sentiment,
                "confidence": confidence,
                "sample_size": sample_size,
            },
            flags,
        )

    def _coerce_attention_z(self, payload: Dict[str, object]) -> Tuple[float, bool]:
        for key in ("attention_z", "mentions_zscore", "volume_zscore", "attention_score"):
            if key in payload:
                try:
                    return _clamp(float(payload.get(key) or 0.0), -5.0, 8.0), True
                except Exception:
                    break
        mentions = payload.get("mentions")
        baseline = payload.get("baseline_mentions")
        try:
            m = float(mentions or 0.0)
            b = max(1.0, float(baseline or 0.0))
            return _clamp((m - b) / max(1.0, b * 0.25), -5.0, 8.0), True
        except Exception:
            return 0.0, False

    def _coerce_sentiment(self, payload: Dict[str, object]) -> Tuple[float, bool]:
        for key in ("sentiment_score", "sentiment", "sentiment_compound"):
            if key in payload:
                try:
                    return _clamp(float(payload.get(key) or 0.0), -1.0, 1.0), True
                except Exception:
                    break
        return 0.0, False

    def _coerce_confidence(self, payload: Dict[str, object]) -> Tuple[float, bool]:
        for key in ("confidence", "model_confidence"):
            if key in payload:
                try:
                    return _clamp(float(payload.get(key) or 0.0), 0.0, 1.0), True
                except Exception:
                    break
        return 0.5, False

    def _coerce_sample_size(self, payload: Dict[str, object]) -> int:
        for key in ("sample_size", "posts_sampled", "post_count", "mentions"):
            if key in payload:
                try:
                    return max(0, int(float(payload.get(key) or 0.0)))
                except Exception:
                    return 0
        return 0


class XAdapter(SocialSourceAdapter):
    source_name = "X"
    env_endpoint_key = "APEX_SOCIAL_X_ENDPOINT"
    env_token_key = "APEX_SOCIAL_X_TOKEN"
    default_local_name = "x.json"


class TikTokAdapter(SocialSourceAdapter):
    source_name = "TIKTOK"
    env_endpoint_key = "APEX_SOCIAL_TIKTOK_ENDPOINT"
    env_token_key = "APEX_SOCIAL_TIKTOK_TOKEN"
    default_local_name = "tiktok.json"


class InstagramAdapter(SocialSourceAdapter):
    source_name = "INSTAGRAM"
    env_endpoint_key = "APEX_SOCIAL_INSTAGRAM_ENDPOINT"
    env_token_key = "APEX_SOCIAL_INSTAGRAM_TOKEN"
    default_local_name = "instagram.json"


class YouTubeAdapter(SocialSourceAdapter):
    source_name = "YOUTUBE"
    env_endpoint_key = "APEX_SOCIAL_YOUTUBE_ENDPOINT"
    env_token_key = "APEX_SOCIAL_YOUTUBE_TOKEN"
    default_local_name = "youtube.json"


def default_social_adapters(data_dir: Path) -> List[SocialSourceAdapter]:
    adapters: List[SocialSourceAdapter] = [
        XAdapter(data_dir=data_dir),
        TikTokAdapter(data_dir=data_dir),
        InstagramAdapter(data_dir=data_dir),
        YouTubeAdapter(data_dir=data_dir),
    ]

    # Dynamic adapters — guarded by SOCIAL_DYNAMIC_ADAPTERS_ENABLED kill switch
    try:
        from config import ApexConfig
        enabled = getattr(ApexConfig, "SOCIAL_DYNAMIC_ADAPTERS_ENABLED", True)
    except Exception:
        enabled = True

    if enabled:
        try:
            from data.social.news_adapter import NewsAggregatorAdapter
            adapters.append(NewsAggregatorAdapter(data_dir=data_dir))
        except Exception:
            pass
        try:
            from data.social.market_adapter import MarketSentimentAdapter
            adapters.append(MarketSentimentAdapter(data_dir=data_dir))
        except Exception:
            pass

    return adapters
