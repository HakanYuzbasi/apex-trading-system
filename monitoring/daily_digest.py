"""
monitoring/daily_digest.py
──────────────────────────
Automated Daily Digest Generator.

Combines P&L attribution, gate diagnostics, model drift signals, and
auto-tuner recommendations into a structured daily summary with plain-English
actionable recommendations.

Optionally posts to a Slack webhook (if APEX_SLACK_WEBHOOK is set).

Usage
─────
    digest = DailyDigest(data_dir="data")
    report = digest.generate()        # → DigestReport
    print(report.text_summary())
    digest.maybe_post_slack(report)   # no-op if webhook not configured
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    priority: str        # "high" | "medium" | "low"
    category: str        # "risk" | "signal" | "execution" | "model"
    message: str
    action: str          # short imperative action
    value: Optional[float] = None


@dataclass
class DigestReport:
    generated_at: str    = ""
    lookback_days: int   = 1

    # P&L summary
    total_pnl_pct: Optional[float]     = None
    ibkr_pnl_pct: Optional[float]      = None
    alpaca_pnl_pct: Optional[float]    = None
    realized_pnl: Optional[float]      = None
    total_trades: int                  = 0
    win_rate: Optional[float]          = None

    # Gate diagnostics
    total_decisions: int               = 0
    total_blocked: int                 = 0
    block_rate: Optional[float]        = None
    top_blocking_gate: Optional[str]   = None
    top_blocking_gate_rate: Optional[float] = None

    # Model drift
    sharpe_7d: Optional[float]         = None
    sharpe_30d: Optional[float]        = None
    sharpe_drift: Optional[float]      = None   # 7d - 30d

    # Auto-tuner
    tuner_adjustments: List[Dict]      = field(default_factory=list)
    tuner_active: bool                 = False

    # MC Sentinel
    mc_sentinel_tier: Optional[str]    = None
    mc_breach_prob: Optional[float]    = None

    # Recommendations
    recommendations: List[Recommendation] = field(default_factory=list)

    def text_summary(self) -> str:
        lines = [
            f"═══ APEX Daily Digest — {self.generated_at[:10]} ═══",
            "",
        ]
        # P&L
        if self.total_pnl_pct is not None:
            lines.append(
                f"P&L: {self.total_pnl_pct*100:+.2f}%  "
                f"(IBKR {(self.ibkr_pnl_pct or 0)*100:+.2f}%  "
                f"Alpaca {(self.alpaca_pnl_pct or 0)*100:+.2f}%)"
            )
        if self.win_rate is not None:
            lines.append(
                f"Trades: {self.total_trades}  Win Rate: {self.win_rate*100:.1f}%"
            )

        # Gates
        if self.block_rate is not None:
            lines.append(
                f"Gate Block Rate: {self.block_rate*100:.1f}%"
                + (f"  Top: {self.top_blocking_gate} ({self.top_blocking_gate_rate*100:.0f}%)"
                   if self.top_blocking_gate else "")
            )

        # Model drift
        if self.sharpe_drift is not None:
            sign = "↑" if self.sharpe_drift > 0 else "↓"
            lines.append(
                f"Sharpe: 7d={self.sharpe_7d:.2f}  30d={self.sharpe_30d:.2f}  drift={sign}{abs(self.sharpe_drift):.2f}"
            )

        # MC Sentinel
        if self.mc_sentinel_tier:
            lines.append(
                f"MC Sentinel: [{self.mc_sentinel_tier.upper()}]  "
                f"P(breach)={((self.mc_breach_prob or 0)*100):.1f}%"
            )

        # Auto-tuner
        if self.tuner_active and self.tuner_adjustments:
            lines.append(f"AutoTuner: {len(self.tuner_adjustments)} threshold adjustment(s)")

        # Recommendations
        if self.recommendations:
            lines.append("")
            lines.append("── Recommendations ──")
            for r in self.recommendations:
                lines.append(f"[{r.priority.upper()}/{r.category}] {r.message}")
                lines.append(f"  → {r.action}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recommendations"] = [asdict(r) for r in self.recommendations]
        return d


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_get(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_last_eod_digest(data_dir: Path) -> Optional[Dict]:
    """Find the most recent EOD digest JSON in data/users/*/digests/."""
    candidates = sorted(data_dir.glob("users/*/digests/eod_digest_*.json"), reverse=True)
    for p in candidates[:1]:
        return _safe_get(p)
    return None


def _read_auto_tuner_state(data_dir: Path) -> Optional[Dict]:
    return _safe_get(data_dir / "auto_tuner_state.json")


def _read_auto_tuned_thresholds(data_dir: Path) -> Optional[Dict]:
    return _safe_get(data_dir / "auto_tuned_thresholds.json")


def _read_mc_sentinel(data_dir: Path) -> Optional[Dict]:
    return _safe_get(data_dir / "monte_carlo_sentinel_state.json")


def _read_trade_diagnostics(data_dir: Path) -> Optional[Dict]:
    """Load latest TradeDiagnosticsTracker report summary."""
    p = data_dir / "trade_diagnostics.jsonl"
    if not p.exists():
        return None
    # Read last few lines for a quick count
    try:
        lines = p.read_text().strip().split("\n")
        total = len(lines)
        blocked = sum(1 for ln in lines if '"blocked"' in ln)
        entered = total - blocked
        return {"total": total, "blocked": blocked, "entered": entered}
    except Exception:
        return None


def _walk_forward_sharpe(data_dir: Path, window_days: int) -> Optional[float]:
    """Approximate Sharpe from daily P&L audit entries."""
    import math
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).timestamp()
    pnl_list: List[float] = []
    for p in sorted((data_dir / "users").glob("*/audit/trade_audit_*.jsonl"), reverse=True):
        try:
            for line in p.read_text().strip().split("\n"):
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("event_type") != "EXIT":
                    continue
                ts = datetime.fromisoformat(rec.get("timestamp", "1970-01-01")).timestamp()
                if ts < cutoff:
                    continue
                pnl_list.append(float(rec.get("pnl_pct", 0.0)))
        except Exception:
            continue
        if len(pnl_list) >= 30:
            break
    if len(pnl_list) < 5:
        return None
    n = len(pnl_list)
    mu = sum(pnl_list) / n
    var = sum((x - mu) ** 2 for x in pnl_list) / n
    std = math.sqrt(var) + 1e-9
    # Annualise assuming ~250 daily observations
    return round((mu / std) * (250 ** 0.5), 3)


# ── Recommendation engine ─────────────────────────────────────────────────────

def _build_recommendations(report: DigestReport) -> List[Recommendation]:
    recs: List[Recommendation] = []

    # 1. Block rate too high
    if report.block_rate is not None and report.block_rate > 0.70:
        recs.append(Recommendation(
            priority="high", category="signal",
            message=f"Gate block rate is very high ({report.block_rate*100:.0f}%). "
                    "Most signals are being filtered out — likely miscalibrated thresholds.",
            action="Run scripts/regime_conditional_backtest.py and lower regime thresholds by 0.02.",
        ))
    elif report.block_rate is not None and report.block_rate > 0.50:
        recs.append(Recommendation(
            priority="medium", category="signal",
            message=f"Gate block rate {report.block_rate*100:.0f}% — consider loosening entry gates.",
            action="Review ThresholdCalibrator state and check symbol probation lists.",
        ))

    # 2. Win rate low
    if report.win_rate is not None and report.total_trades >= 5:
        if report.win_rate < 0.35:
            recs.append(Recommendation(
                priority="high", category="model",
                message=f"Win rate critically low at {report.win_rate*100:.0f}% ({report.total_trades} trades).",
                action="Halt new entries, retrain models, review regime classification.",
                value=report.win_rate,
            ))
        elif report.win_rate < 0.45:
            recs.append(Recommendation(
                priority="medium", category="model",
                message=f"Win rate below target: {report.win_rate*100:.0f}%.",
                action="Check signal auto-tuner state and recent regime distribution.",
                value=report.win_rate,
            ))

    # 3. Negative P&L
    if report.total_pnl_pct is not None and report.total_pnl_pct < -0.02:
        recs.append(Recommendation(
            priority="high", category="risk",
            message=f"Daily P&L {report.total_pnl_pct*100:+.1f}% — approaching daily loss limit.",
            action="Reduce position sizes by 50%, disable new crypto entries.",
            value=report.total_pnl_pct,
        ))

    # 4. Sharpe drift negative
    if report.sharpe_drift is not None and report.sharpe_drift < -0.5:
        recs.append(Recommendation(
            priority="medium", category="model",
            message=f"Sharpe degraded by {abs(report.sharpe_drift):.2f} (7d vs 30d).",
            action="Review recent regime shifts; re-run walk-forward validation.",
            value=report.sharpe_drift,
        ))

    # 5. MC Sentinel amber/red
    if report.mc_sentinel_tier in ("amber", "red", "defensive"):
        recs.append(Recommendation(
            priority="high" if report.mc_sentinel_tier in ("red", "defensive") else "medium",
            category="risk",
            message=f"Monte Carlo Sentinel elevated: [{report.mc_sentinel_tier.upper()}] "
                    f"P(breach)={((report.mc_breach_prob or 0)*100):.0f}%.",
            action="Reduce position sizes per sentinel multiplier; review daily loss exposure.",
            value=report.mc_breach_prob,
        ))

    # 6. Auto-tuner active adjustments
    if report.tuner_active and len(report.tuner_adjustments) >= 3:
        recs.append(Recommendation(
            priority="low", category="signal",
            message=f"AutoTuner made {len(report.tuner_adjustments)} threshold adjustments today.",
            action="Review auto_tuned_thresholds.json and verify changes are within expected bounds.",
        ))

    return recs


# ── Main class ─────────────────────────────────────────────────────────────────

class DailyDigest:
    """
    Automated Daily Digest Generator.

    Aggregates signals from multiple monitoring subsystems and produces
    a structured DigestReport with plain-English recommendations.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        lookback_days: int    = 1,
        slack_webhook: Optional[str] = None,
    ) -> None:
        self._data_dir    = Path(data_dir)
        self._lookback    = lookback_days
        self._webhook     = slack_webhook or os.getenv("APEX_SLACK_WEBHOOK", "")
        self._output_path = self._data_dir / "daily_digest_latest.json"

    # ── Generate ───────────────────────────────────────────────────────────────

    def generate(self) -> DigestReport:
        report = DigestReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            lookback_days=self._lookback,
        )

        # ── P&L from EOD digest ─────────────────────────────────────────────
        eod = _read_last_eod_digest(self._data_dir)
        if eod:
            report.total_pnl_pct   = eod.get("daily_return")
            report.realized_pnl    = eod.get("realized_pnl")
            report.win_rate        = eod.get("win_rate")
            report.total_trades    = int(eod.get("total_trades", 0))
            bkr = eod.get("by_broker", {})
            report.ibkr_pnl_pct    = bkr.get("ibkr_return")
            report.alpaca_pnl_pct  = bkr.get("alpaca_return")

        # ── Gate diagnostics from TradeDiagnosticsTracker ─────────────────
        diag = _read_trade_diagnostics(self._data_dir)
        if diag:
            report.total_decisions = diag.get("total", 0)
            report.total_blocked   = diag.get("blocked", 0)
            if report.total_decisions > 0:
                report.block_rate = report.total_blocked / report.total_decisions

        # Also pull from the diagnostics report endpoint if available
        try:
            from monitoring.trade_diagnostics import TradeDiagnosticsTracker
            tracker = TradeDiagnosticsTracker(data_dir=self._data_dir)
            dr = tracker.get_report(lookback_days=self._lookback)
            report.total_decisions = int(dr.get("total_records", report.total_decisions))
            ba = dr.get("blocked_analysis", {})
            report.block_rate      = ba.get("block_rate", report.block_rate)
            report.total_blocked   = int(ba.get("total_blocked", report.total_blocked))
            # Top blocking gate
            ga = dr.get("gate_attribution", {})
            if ga:
                top = max(ga.items(), key=lambda kv: kv[1].get("block_rate", 0))
                report.top_blocking_gate      = top[0]
                report.top_blocking_gate_rate = top[1].get("block_rate")
        except Exception:
            pass

        # ── Sharpe drift ───────────────────────────────────────────────────
        report.sharpe_7d  = _walk_forward_sharpe(self._data_dir, 7)
        report.sharpe_30d = _walk_forward_sharpe(self._data_dir, 30)
        if report.sharpe_7d is not None and report.sharpe_30d is not None:
            report.sharpe_drift = round(report.sharpe_7d - report.sharpe_30d, 3)

        # ── Auto-tuner ─────────────────────────────────────────────────────
        tuner = _read_auto_tuner_state(self._data_dir)
        if tuner:
            thresh = _read_auto_tuned_thresholds(self._data_dir) or {}
            report.tuner_active      = True
            report.tuner_adjustments = [
                {"regime": k, "threshold": v}
                for k, v in thresh.items()
                if isinstance(v, (int, float))
            ]

        # ── MC Sentinel ────────────────────────────────────────────────────
        mc = _read_mc_sentinel(self._data_dir)
        if mc:
            hist = mc.get("history", [])
            if hist:
                import math
                mu = sum(hist) / len(hist)
                var = sum((x - mu) ** 2 for x in hist) / len(hist)
                std = math.sqrt(var) + 1e-9
                # Proxy for last evaluation tier from the most recent recorded pnl
                last_pnl = hist[-1] if hist else 0.0
                report.mc_sentinel_tier = "green"
                if last_pnl < -0.015:
                    report.mc_sentinel_tier = "amber"
                if last_pnl < -0.02:
                    report.mc_sentinel_tier = "red"
                report.mc_breach_prob = max(0.0, min(1.0, abs(last_pnl) / 0.02))

        # ── Build recommendations ──────────────────────────────────────────
        report.recommendations = _build_recommendations(report)

        # ── Persist ────────────────────────────────────────────────────────
        self._save(report)
        return report

    def _save(self, report: DigestReport) -> None:
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._output_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(report.to_dict(), indent=2, default=str))
            tmp.replace(self._output_path)
        except Exception as exc:
            logger.warning("DailyDigest save failed: %s", exc)

    # ── Slack webhook ──────────────────────────────────────────────────────────

    def maybe_post_slack(self, report: DigestReport) -> bool:
        """Post digest to Slack webhook if configured. Returns True on success."""
        if not self._webhook:
            return False
        try:
            text = report.text_summary()
            payload = json.dumps({"text": f"```{text}```"}).encode()
            req = urllib.request.Request(
                self._webhook,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                success = resp.status == 200
            if success:
                logger.info("DailyDigest posted to Slack")
            return success
        except Exception as exc:
            logger.warning("DailyDigest Slack post failed: %s", exc)
            return False
