"""
monitoring/daily_briefing.py — Daily Automated Strategy Briefing

Generates a concise end-of-day summary combining:
  - Today's trade performance (wins, losses, P&L)
  - Active signal factors and IC leaderboard
  - Regime snapshot and regime transitions
  - System health (backtest gate mode, strategy health mode)
  - Top recommendations for tomorrow

Output channels:
  - JSON file (data/daily_briefings/YYYY-MM-DD.json)
  - Human-readable text summary (logged + optionally to Telegram)
  - API: GET /ops/daily-briefing

Config keys:
    DAILY_BRIEFING_ENABLED           = True
    DAILY_BRIEFING_TELEGRAM_ENABLED  = False
    DAILY_BRIEFING_TELEGRAM_TOKEN    = ""
    DAILY_BRIEFING_TELEGRAM_CHAT_ID  = ""
    DAILY_BRIEFING_OUTPUT_DIR        = "data/daily_briefings"
    DAILY_BRIEFING_KEEP_DAYS         = 30
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "DAILY_BRIEFING_ENABLED":          True,
    "DAILY_BRIEFING_TELEGRAM_ENABLED": False,
    "DAILY_BRIEFING_TELEGRAM_TOKEN":   "",
    "DAILY_BRIEFING_TELEGRAM_CHAT_ID": "",
    "DAILY_BRIEFING_OUTPUT_DIR":       "data/daily_briefings",
    "DAILY_BRIEFING_KEEP_DAYS":        30,
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TradeStats:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    best_trade: Optional[Dict] = None
    worst_trade: Optional[Dict] = None


@dataclass
class SignalStats:
    name: str
    ic: float
    obs: int
    win_rate: float
    status: str    # active | weak | unreliable


@dataclass
class DailyBriefing:
    date: str
    generated_at: str
    # Performance
    trade_stats: TradeStats
    # Factors
    top_signals: List[SignalStats]
    weak_signals: List[str]
    # System state
    regime: str
    backtest_gate_mode: str          # live | paper | unknown
    strategy_health_paper_only: bool
    adaptive_weights: Dict[str, float]  # signal → current weight
    # Recommendations
    recommendations: List[str]
    # Raw extras
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """Human-readable briefing text."""
        ts = self.trade_stats
        lines = [
            f"╔══════════════════════════════════════════════════╗",
            f"║  APEX TRADING — DAILY BRIEFING  {self.date}        ║",
            f"╚══════════════════════════════════════════════════╝",
            f"",
            f"📊 PERFORMANCE",
            f"   Trades: {ts.total_trades}  |  Wins: {ts.wins}  |  Losses: {ts.losses}",
            f"   Win Rate: {ts.win_rate*100:.1f}%  |  Avg P&L: {ts.avg_pnl_pct*100:+.3f}%"
            f"  |  Total P&L: {ts.total_pnl_pct*100:+.3f}%",
        ]
        if ts.best_trade:
            lines.append(
                f"   Best: {ts.best_trade.get('symbol','?')} "
                f"{float(ts.best_trade.get('pnl_pct',0))*100:+.2f}%"
            )
        if ts.worst_trade:
            lines.append(
                f"   Worst: {ts.worst_trade.get('symbol','?')} "
                f"{float(ts.worst_trade.get('pnl_pct',0))*100:+.2f}%"
            )

        lines += [
            f"",
            f"🧠 SIGNAL FACTORS  (IC leaderboard)",
        ]
        if self.top_signals:
            for s in self.top_signals[:5]:
                lines.append(
                    f"   {s.name:<22} IC={s.ic:+.3f}  WR={s.win_rate*100:.0f}%"
                    f"  obs={s.obs}  [{s.status}]"
                )
        else:
            lines.append("   (insufficient data)")

        if self.weak_signals:
            lines.append(f"   Weak/unreliable: {', '.join(self.weak_signals)}")

        lines += [
            f"",
            f"🌡 SYSTEM STATE",
            f"   Regime:          {self.regime}",
            f"   Backtest Gate:   {self.backtest_gate_mode.upper()}",
            f"   Strategy Health: {'PAPER-ONLY' if self.strategy_health_paper_only else 'live'}",
        ]

        if self.adaptive_weights:
            aw_str = "  ".join(
                f"{k}={v:.3f}" for k, v in sorted(self.adaptive_weights.items())
                if k != "primary_signal"
            )
            lines.append(f"   Adaptive Weights: {aw_str}")

        lines += ["", "💡 RECOMMENDATIONS"]
        for rec in self.recommendations:
            lines.append(f"   • {rec}")

        if self.notes:
            lines += ["", "📝 NOTES"]
            for note in self.notes:
                lines.append(f"   {note}")

        lines.append("")
        return "\n".join(lines)


# ── Generator ─────────────────────────────────────────────────────────────────

class DailyBriefingGenerator:
    """
    Assembles a DailyBriefing from live engine state and audit files.
    Designed to be called at EOD or on-demand.
    """

    def __init__(self, data_dir: Optional[str] = None):
        try:
            from config import ApexConfig
            self._data_dir = Path(data_dir or str(ApexConfig.DATA_DIR))
        except Exception:
            self._data_dir = Path(data_dir or "data")

        cfg_out = str(_cfg("DAILY_BRIEFING_OUTPUT_DIR"))
        out_path = Path(cfg_out)
        # If config path is relative (e.g. "data/daily_briefings"), anchor it
        # to the resolved data_dir so tests with temp dirs stay isolated.
        if not out_path.is_absolute():
            self._output_dir = self._data_dir / "daily_briefings"
        else:
            self._output_dir = out_path
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        report_date: Optional[date] = None,
        regime: str = "unknown",
        engine: Optional[Any] = None,
    ) -> DailyBriefing:
        """
        Generate the daily briefing.

        Args:
            report_date: Date to report on (default: today)
            regime: Current regime string from engine
            engine: Running TradingEngine instance (for live state reads)
        """
        today = report_date or date.today()
        generated_at = datetime.now(timezone.utc).isoformat()

        trade_stats = self._compute_trade_stats(today)
        signal_stats, weak_signals = self._compute_signal_stats(engine)
        backtest_mode = self._get_backtest_gate_mode(engine)
        health_paper = self._get_strategy_health(engine)
        adaptive_weights = self._get_adaptive_weights(engine)
        recommendations = self._build_recommendations(
            trade_stats, signal_stats, backtest_mode, health_paper
        )

        briefing = DailyBriefing(
            date=str(today),
            generated_at=generated_at,
            trade_stats=trade_stats,
            top_signals=signal_stats,
            weak_signals=weak_signals,
            regime=regime,
            backtest_gate_mode=backtest_mode,
            strategy_health_paper_only=health_paper,
            adaptive_weights=adaptive_weights,
            recommendations=recommendations,
        )

        self._save(briefing)
        logger.info("DailyBriefing generated for %s — %d trades, mode=%s",
                    today, trade_stats.total_trades, backtest_mode)

        if _cfg("DAILY_BRIEFING_TELEGRAM_ENABLED"):
            self._send_telegram(briefing.to_text())

        return briefing

    def get_latest(self) -> Optional[dict]:
        """Return the most recent saved briefing as a dict."""
        files = sorted(self._output_dir.glob("????-??-??.json"), reverse=True)
        if not files:
            return None
        try:
            with open(files[0]) as f:
                return json.load(f)
        except Exception as e:
            logger.debug("DailyBriefing: could not load latest — %s", e)
            return None

    def get_history(self, days: int = 7) -> List[dict]:
        """Return up to `days` most recent briefings."""
        files = sorted(self._output_dir.glob("????-??-??.json"), reverse=True)
        result = []
        for f in files[:days]:
            try:
                with open(f) as fh:
                    result.append(json.load(fh))
            except Exception:
                pass
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_trade_stats(self, today: date) -> TradeStats:
        """Load today's EXIT rows from audit JSONL and compute stats."""
        exits: List[dict] = []
        for f in sorted(self._data_dir.glob("trade_audit_*.jsonl")):
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if row.get("event") != "EXIT":
                            continue
                        ts_str = row.get("ts") or row.get("timestamp") or ""
                        if ts_str:
                            try:
                                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                if ts.date() != today:
                                    continue
                            except ValueError:
                                pass
                        exits.append(row)
            except Exception:
                pass

        if not exits:
            return TradeStats(0, 0, 0, 0.0, 0.0, 0.0)

        pnls = [float(r.get("pnl_pct", 0)) for r in exits]
        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins
        best = max(exits, key=lambda r: float(r.get("pnl_pct", 0)))
        worst = min(exits, key=lambda r: float(r.get("pnl_pct", 0)))

        return TradeStats(
            total_trades=len(exits),
            wins=wins,
            losses=losses,
            win_rate=round(wins / len(pnls), 4),
            avg_pnl_pct=round(float(np.mean(pnls)), 5),
            total_pnl_pct=round(float(np.sum(pnls)), 5),
            best_trade={"symbol": best.get("symbol", "?"), "pnl_pct": float(best.get("pnl_pct", 0))},
            worst_trade={"symbol": worst.get("symbol", "?"), "pnl_pct": float(worst.get("pnl_pct", 0))},
        )

    def _compute_signal_stats(self, engine: Optional[Any]) -> tuple:
        """Read FactorICTracker from engine or load from state file."""
        try:
            tracker = None
            if engine is not None:
                tracker = getattr(engine, "_factor_ic_tracker", None)
            if tracker is None:
                from monitoring.factor_ic_tracker import FactorICTracker
                state_path = str(self._data_dir / "factor_ic_state.json")
                tracker = FactorICTracker(persist_path=state_path)

            report = tracker.get_report()
            top = [
                SignalStats(
                    name=r.signal_name, ic=r.ic, obs=r.obs,
                    win_rate=r.win_rate, status=r.status,
                )
                for r in report.signals
                if r.status in ("active", "weak")
            ]
            weak = report.weak_factors
            return top, weak
        except Exception as e:
            logger.debug("DailyBriefing: signal stats failed — %s", e)
            return [], []

    def _get_backtest_gate_mode(self, engine: Optional[Any]) -> str:
        try:
            gate = None
            if engine is not None:
                gate = getattr(engine, "_backtest_gate", None)
            if gate is None:
                from monitoring.backtest_gate import BacktestGate
                gate = BacktestGate(audit_dir=str(self._data_dir))
            return gate.mode
        except Exception:
            return "unknown"

    def _get_strategy_health(self, engine: Optional[Any]) -> bool:
        try:
            shm = None
            if engine is not None:
                shm = getattr(engine, "_strategy_health", None)
            if shm is None:
                from monitoring.strategy_health_monitor import StrategyHealthMonitor
                shm = StrategyHealthMonitor()
            return shm.paper_only
        except Exception:
            return False

    def _get_adaptive_weights(self, engine: Optional[Any]) -> Dict[str, float]:
        try:
            awm = None
            if engine is not None:
                awm = getattr(engine, "_adaptive_weights", None)
            if awm is None:
                from monitoring.adaptive_weight_manager import AdaptiveWeightManager
                awm = AdaptiveWeightManager()
            report = awm.get_report()
            return {
                k: round(v["current"], 4)
                for k, v in report.get("weights", {}).items()
                if k != "primary_signal"
            }
        except Exception:
            return {}

    def _build_recommendations(
        self,
        ts: TradeStats,
        signals: List[SignalStats],
        gate_mode: str,
        paper_only: bool,
    ) -> List[str]:
        recs: List[str] = []

        if gate_mode == "paper":
            recs.append("BacktestGate is in PAPER mode — strategy degraded. Review performance before going live.")
        if paper_only:
            recs.append("StrategyHealthMonitor flagged low Sharpe — live trading suppressed.")

        if ts.total_trades > 0:
            if ts.win_rate < 0.40:
                recs.append(f"Win rate {ts.win_rate*100:.1f}% is below 40% — review entry filters.")
            elif ts.win_rate >= 0.60:
                recs.append(f"Strong win rate {ts.win_rate*100:.1f}% — consider loosening entry thresholds.")

            if ts.avg_pnl_pct < -0.005:
                recs.append("Avg P&L is negative — check stop placement and sizing.")

        weak_ic = [s for s in signals if s.status == "weak" and s.obs >= 10]
        if weak_ic:
            names = ", ".join(s.name for s in weak_ic)
            recs.append(f"Weak IC signals ({names}) contributing little alpha — review blend weights.")

        top_ic = [s for s in signals if s.status == "active" and s.ic > 0.20]
        if top_ic:
            names = ", ".join(s.name for s in top_ic[:2])
            recs.append(f"High-IC signals ({names}) — adaptive weights should be increasing these.")

        if not recs:
            recs.append("System operating normally — no immediate actions required.")

        return recs

    def _save(self, briefing: DailyBriefing) -> None:
        path = self._output_dir / f"{briefing.date}.json"
        try:
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(briefing.to_dict(), f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.debug("DailyBriefing: save failed — %s", e)

        # Prune old files
        try:
            keep = int(_cfg("DAILY_BRIEFING_KEEP_DAYS"))
            cutoff = date.today() - timedelta(days=keep)
            for old_f in self._output_dir.glob("????-??-??.json"):
                try:
                    d = date.fromisoformat(old_f.stem)
                    if d < cutoff:
                        old_f.unlink()
                except ValueError:
                    pass
        except Exception:
            pass

    def _send_telegram(self, text: str) -> None:
        """Send briefing to Telegram chat (best-effort, non-blocking)."""
        try:
            import urllib.request
            token = str(_cfg("DAILY_BRIEFING_TELEGRAM_TOKEN"))
            chat_id = str(_cfg("DAILY_BRIEFING_TELEGRAM_CHAT_ID"))
            if not token or not chat_id:
                return
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = json.dumps({"chat_id": chat_id, "text": text[:4096]}).encode()
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            logger.info("DailyBriefing: Telegram notification sent")
        except Exception as e:
            logger.debug("DailyBriefing: Telegram send failed — %s", e)


# ── Module-level singleton ────────────────────────────────────────────────────

_generator: Optional[DailyBriefingGenerator] = None


def get_briefing_generator(data_dir: Optional[str] = None) -> DailyBriefingGenerator:
    global _generator
    if _generator is None:
        _generator = DailyBriefingGenerator(data_dir=data_dir)
    return _generator
