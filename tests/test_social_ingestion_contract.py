from datetime import datetime
import json

from data.social.contract import write_social_risk_inputs


def test_social_ingestion_contract_builds_with_quality_flags(tmp_path):
    social_dir = tmp_path / "social"
    social_dir.mkdir(parents=True, exist_ok=True)
    (social_dir / "x.json").write_text(
        json.dumps(
            {
                "fetched_at": datetime.utcnow().isoformat(),
                "freshness_ts": datetime.utcnow().isoformat(),
                "attention_z": 1.8,
                "sentiment_score": -0.3,
                "confidence": 0.8,
                "sample_size": 420,
            }
        ),
        encoding="utf-8",
    )

    path, payload, report = write_social_risk_inputs(
        data_dir=tmp_path,
        freshness_sla_seconds=1800,
    )

    assert path.exists()
    assert report.valid
    assert report.has_usable_feeds
    assert payload["schema_version"] == "1.0"
    assert "sources" in payload
    assert "platforms" in payload
    assert payload["sources"]["X"]["quality"]["status"] in {"ok", "degraded"}
    assert payload["sources"]["TIKTOK"]["quality"]["status"] == "missing"
