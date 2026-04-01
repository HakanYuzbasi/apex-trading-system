"""tests/test_daily_digest.py — Automated Daily Digest tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from monitoring.daily_digest import (
    DailyDigest,
    DigestReport,
    Recommendation,
    _build_recommendations,
    _walk_forward_sharpe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_data(tmp_path):
    return tmp_path


@pytest.fixture()
def digest(tmp_data):
    return DailyDigest(data_dir=tmp_data, lookback_days=1)


def _write_eod_digest(data_dir: Path, payload: dict) -> None:
    p = data_dir / "users" / "admin" / "digests"
    p.mkdir(parents=True, exist_ok=True)
    (p / "eod_digest_20250115.json").write_text(json.dumps(payload))


def _write_audit_exits(data_dir: Path, exits: list[dict]) -> None:
    from datetime import datetime, timezone
    p = data_dir / "users" / "admin" / "audit"
    p.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    lines = "\n".join(json.dumps(e) for e in exits)
    (p / f"trade_audit_{today}.jsonl").write_text(lines)


def _recent_ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── DigestReport ──────────────────────────────────────────────────────────────

class TestDigestReport:
    def test_defaults(self):
        r = DigestReport()
        assert r.total_trades == 0
        assert r.recommendations == []

    def test_to_dict_keys(self):
        r = DigestReport(total_pnl_pct=0.01, total_trades=5, win_rate=0.6)
        d = r.to_dict()
        assert "total_pnl_pct" in d
        assert "recommendations" in d
        assert isinstance(d["recommendations"], list)

    def test_text_summary_pnl(self):
        r = DigestReport(
            generated_at="2025-01-15T10:00:00+00:00",
            total_pnl_pct=0.015,
            total_trades=10,
            win_rate=0.6,
        )
        text = r.text_summary()
        assert "P&L:" in text
        assert "+1.50" in text

    def test_text_summary_recommendations(self):
        r = DigestReport()
        r.recommendations = [
            Recommendation("high", "risk", "test message", "test action")
        ]
        text = r.text_summary()
        assert "Recommendations" in text
        assert "HIGH/risk" in text
        assert "test message" in text

    def test_text_summary_sharpe(self):
        r = DigestReport(sharpe_7d=1.2, sharpe_30d=0.9, sharpe_drift=0.3)
        text = r.text_summary()
        assert "Sharpe" in text
        assert "1.20" in text


# ── Recommendation ────────────────────────────────────────────────────────────

class TestRecommendation:
    def test_fields(self):
        rec = Recommendation("high", "risk", "message", "action", value=0.5)
        assert rec.priority == "high"
        assert rec.value == 0.5


# ── _build_recommendations ────────────────────────────────────────────────────

class TestBuildRecommendations:
    def test_high_block_rate(self):
        r = DigestReport(block_rate=0.80, total_decisions=100)
        recs = _build_recommendations(r)
        assert any(r.priority == "high" and r.category == "signal" for r in recs)

    def test_medium_block_rate(self):
        r = DigestReport(block_rate=0.55, total_decisions=100)
        recs = _build_recommendations(r)
        assert any(r.priority == "medium" for r in recs)

    def test_low_win_rate_critical(self):
        r = DigestReport(win_rate=0.25, total_trades=10)
        recs = _build_recommendations(r)
        assert any(r.priority == "high" and r.category == "model" for r in recs)

    def test_low_win_rate_medium(self):
        r = DigestReport(win_rate=0.40, total_trades=10)
        recs = _build_recommendations(r)
        assert any(r.priority == "medium" and r.category == "model" for r in recs)

    def test_insufficient_trades_no_win_rate_rec(self):
        r = DigestReport(win_rate=0.20, total_trades=2)  # < 5 trades → skip
        recs = _build_recommendations(r)
        assert not any(r.category == "model" for r in recs)

    def test_negative_pnl(self):
        r = DigestReport(total_pnl_pct=-0.03)
        recs = _build_recommendations(r)
        assert any(r.category == "risk" for r in recs)

    def test_sharpe_drift_negative(self):
        r = DigestReport(sharpe_7d=0.3, sharpe_30d=1.0, sharpe_drift=-0.7)
        recs = _build_recommendations(r)
        assert any("Sharpe" in r.message for r in recs)

    def test_mc_sentinel_amber(self):
        r = DigestReport(mc_sentinel_tier="amber", mc_breach_prob=0.40)
        recs = _build_recommendations(r)
        assert any("Sentinel" in r.message for r in recs)

    def test_mc_sentinel_defensive_high_priority(self):
        r = DigestReport(mc_sentinel_tier="defensive", mc_breach_prob=0.75)
        recs = _build_recommendations(r)
        rec = next((r for r in recs if "Sentinel" in r.message), None)
        assert rec is not None
        assert rec.priority == "high"

    def test_no_recommendations_when_healthy(self):
        r = DigestReport(
            total_pnl_pct=0.01,
            block_rate=0.30,
            win_rate=0.60,
            total_trades=10,
            sharpe_drift=0.1,
            mc_sentinel_tier="green",
        )
        recs = _build_recommendations(r)
        assert len(recs) == 0


# ── _walk_forward_sharpe ──────────────────────────────────────────────────────

class TestWalkForwardSharpe:
    def test_empty_returns_none(self, tmp_data):
        result = _walk_forward_sharpe(tmp_data, 7)
        assert result is None

    def test_insufficient_returns_none(self, tmp_data):
        ts = _recent_ts()
        exits = [{"event_type": "EXIT", "pnl_pct": 0.01, "timestamp": ts}] * 3
        _write_audit_exits(tmp_data, exits)
        result = _walk_forward_sharpe(tmp_data, 7)
        assert result is None

    def test_positive_sharpe(self, tmp_data):
        ts = _recent_ts()
        exits = [{"event_type": "EXIT", "pnl_pct": 0.01, "timestamp": ts}] * 20
        _write_audit_exits(tmp_data, exits)
        result = _walk_forward_sharpe(tmp_data, 30)
        assert result is not None
        assert result > 0

    def test_negative_sharpe(self, tmp_data):
        ts = _recent_ts()
        exits = [{"event_type": "EXIT", "pnl_pct": -0.01, "timestamp": ts}] * 20
        _write_audit_exits(tmp_data, exits)
        result = _walk_forward_sharpe(tmp_data, 30)
        assert result is not None
        assert result < 0


# ── DailyDigest.generate ──────────────────────────────────────────────────────

class TestDailyDigestGenerate:
    def test_empty_data_returns_report(self, digest):
        report = digest.generate()
        assert isinstance(report, DigestReport)
        assert report.generated_at != ""

    def test_reads_eod_digest(self, digest, tmp_data):
        _write_eod_digest(tmp_data, {
            "daily_return": 0.015,
            "win_rate": 0.60,
            "total_trades": 12,
            "realized_pnl": 1500.0,
        })
        report = digest.generate()
        assert abs(report.total_pnl_pct - 0.015) < 1e-9
        assert abs(report.win_rate - 0.60) < 1e-9

    def test_reads_auto_tuner(self, digest, tmp_data):
        (tmp_data / "auto_tuner_state.json").write_text(
            json.dumps({"last_run": "2025-01-15", "adjustments_made": 3})
        )
        (tmp_data / "auto_tuned_thresholds.json").write_text(
            json.dumps({"bull": 0.14, "bear": 0.18})
        )
        report = digest.generate()
        assert report.tuner_active is True
        assert len(report.tuner_adjustments) == 2

    def test_sharpe_computed(self, digest, tmp_data):
        ts = _recent_ts()
        exits = [
            {
                "event_type": "EXIT",
                "pnl_pct": 0.005 if i % 3 != 0 else -0.002,
                "timestamp": ts,
            }
            for i in range(20)
        ]
        _write_audit_exits(tmp_data, exits)
        report = digest.generate()
        assert report.sharpe_7d is not None

    def test_saves_output_file(self, digest, tmp_data):
        digest.generate()
        assert (tmp_data / "daily_digest_latest.json").exists()

    def test_recommendations_generated(self, digest, tmp_data):
        _write_eod_digest(tmp_data, {
            "daily_return": -0.025,
            "win_rate": 0.30,
            "total_trades": 8,
        })
        report = digest.generate()
        assert len(report.recommendations) > 0

    def test_mc_sentinel_from_history(self, digest, tmp_data):
        (tmp_data / "monte_carlo_sentinel_state.json").write_text(
            json.dumps({"history": [-0.025, -0.018, -0.022], "last_eval_ts": 0.0})
        )
        report = digest.generate()
        assert report.mc_sentinel_tier is not None


# ── DailyDigest.maybe_post_slack ─────────────────────────────────────────────

class TestMaybePostSlack:
    def test_no_webhook_returns_false(self, digest):
        report = DigestReport(generated_at="2025-01-15")
        assert digest.maybe_post_slack(report) is False

    def test_webhook_posts(self, tmp_data):
        d = DailyDigest(data_dir=tmp_data, slack_webhook="https://hooks.slack.com/fake")
        report = DigestReport(generated_at="2025-01-15T10:00:00+00:00")
        with patch("urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = d.maybe_post_slack(report)
        assert result is True

    def test_webhook_failure_returns_false(self, tmp_data):
        d = DailyDigest(data_dir=tmp_data, slack_webhook="https://hooks.slack.com/fake")
        report = DigestReport(generated_at="2025-01-15T10:00:00+00:00")
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = d.maybe_post_slack(report)
        assert result is False
