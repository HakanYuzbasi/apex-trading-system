"""
monitoring/eod_digest.py — End-of-Day Performance Intelligence Digest

Runs at market close (4 PM ET) and writes a structured JSON report + human-readable
summary to data/eod_reports/YYYYMMDD_digest.json.

Report contains:
- Net P&L breakdown by broker and regime
- Win rate by signal source with actionable threshold recommendations
- Top 3 winners + bottom 3 losers with signal attribution
- Performance governor tier and rolling Sharpe
- Auto-generated recommendations ranked by urgency

Triggered from execution_loop.py when market transitions to AFTER_HOURS.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeSummary:
    symbol: str
    side: str
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    regime: str
    broker: str
    hold_hours: float
    signal: float
    confidence: float
    entry_signal: float


@dataclass
class BrokerDigest:
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades: int = 0
    wins: int = 0

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.trades if self.trades else None


@dataclass
class RegimeDigest:
    trades: int = 0
    wins: int = 0
    pnl_usd: float = 0.0

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.trades if self.trades else None

    @property
    def avg_pnl_pct(self) -> float:
        return (self.pnl_usd / max(self.trades, 1))


@dataclass
class SignalSourceDigest:
    trades: int = 0
    wins: int = 0
    total_pnl_usd: float = 0.0

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.trades if self.trades else None


@dataclass
class EODReport:
    report_date: str
    generated_at: str
    # Totals
    total_trades: int
    total_entries: int
    total_exits: int
    total_realized_pnl: float
    total_unrealized_pnl: float
    overall_win_rate: Optional[float]
    avg_hold_hours: float
    avg_slippage_bps: float
    # Breakdowns
    by_broker: Dict[str, Any]
    by_regime: Dict[str, Any]
    # Top / bottom
    top_trades: List[Dict]
    bottom_trades: List[Dict]
    # Exits by reason category
    exit_reason_summary: Dict[str, int]
    # Governor
    governor_tier: str
    governor_sharpe: Optional[float]
    # Recommendations
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class EODDigestGenerator:
    """
    Reads today's audit trail and generates an end-of-day performance report.

    Usage (from execution_loop):
        gen = EODDigestGenerator(data_dir=ApexConfig.DATA_DIR)
        report = gen.generate()
        gen.save(report)
        gen.log_summary(report)
    """

    def __init__(
        self,
        data_dir: "str | Path" = "data",
        user_id: str = "admin",
        win_rate_warn_threshold: float = 0.45,
        win_rate_urgent_threshold: float = 0.35,
    ):
        self.data_dir = Path(data_dir)
        self.user_id = user_id
        self.win_rate_warn = win_rate_warn_threshold
        self.win_rate_urgent = win_rate_urgent_threshold
        self._reports_dir = self.data_dir / "eod_reports"
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        report_date: Optional[date] = None,
        positions: Optional[Dict[str, Any]] = None,
        price_cache: Optional[Dict[str, float]] = None,
        governor_snapshot: Optional[Any] = None,
    ) -> EODReport:
        """
        Generate the EOD report.

        Args:
            report_date: Date to report on (default: today)
            positions: Open positions dict {symbol: {qty, avg_cost, ...}}
            price_cache: Current prices for unrealized P&L
            governor_snapshot: GovernorSnapshot from PerformanceGovernor
        """
        today = report_date or date.today()
        exits, entries, slippages = self._load_audit(today)

        # ---- P&L from exits --------------------------------------------------
        broker_map: Dict[str, BrokerDigest] = defaultdict(BrokerDigest)
        regime_map: Dict[str, RegimeDigest] = defaultdict(RegimeDigest)
        exit_reasons: Dict[str, int] = defaultdict(int)
        closed_trades: List[TradeSummary] = []
        hold_hours_list: List[float] = []

        for row in exits:
            pnl_usd = float(row.get("pnl_usd", 0.0))
            pnl_pct = float(row.get("pnl_pct", 0.0))
            broker = str(row.get("broker", "unknown"))
            regime = str(row.get("regime", "unknown"))
            exit_reason = str(row.get("exit_reason", "unknown"))

            # Parse hold time
            entry_ts_str = row.get("entry_ts") or row.get("ts")
            exit_ts_str = row.get("ts")
            hold_h = 0.0
            try:
                if entry_ts_str and exit_ts_str:
                    entry_dt = datetime.fromisoformat(entry_ts_str.replace("Z", "+00:00"))
                    exit_dt = datetime.fromisoformat(exit_ts_str.replace("Z", "+00:00"))
                    hold_h = (exit_dt - entry_dt).total_seconds() / 3600
            except Exception:
                hold_h = float(row.get("holding_days", 0)) * 24

            is_win = pnl_usd > 0

            # Broker
            bd = broker_map[broker]
            bd.realized_pnl += pnl_usd
            bd.trades += 1
            if is_win:
                bd.wins += 1

            # Regime
            rd = regime_map[regime]
            rd.trades += 1
            rd.pnl_usd += pnl_usd
            if is_win:
                rd.wins += 1

            # Exit reason bucket
            bucket = self._bucket_exit_reason(exit_reason)
            exit_reasons[bucket] += 1

            hold_hours_list.append(hold_h)
            closed_trades.append(TradeSummary(
                symbol=row.get("symbol", "?"),
                side=row.get("side", "?"),
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                regime=regime,
                broker=broker,
                hold_hours=round(hold_h, 2),
                signal=float(row.get("signal", 0)),
                confidence=float(row.get("confidence", 0)),
                entry_signal=float(row.get("entry_signal", 0)),
            ))

        # ---- Unrealized from open positions ----------------------------------
        unrealized_by_broker: Dict[str, float] = defaultdict(float)
        if positions and price_cache:
            for sym, pos in positions.items():
                qty = float(pos.get("qty", pos.get("quantity", 0)))
                avg_cost = float(pos.get("avg_cost", pos.get("entry_price", 0)))
                cur_price = price_cache.get(sym, 0.0)
                if qty and avg_cost and cur_price:
                    unreal = (cur_price - avg_cost) * qty
                    b = "alpaca" if self._is_crypto(sym) else "ibkr"
                    unrealized_by_broker[b] += unreal

        for b, unreal in unrealized_by_broker.items():
            broker_map[b].unrealized_pnl += unreal

        # ---- Summary metrics -------------------------------------------------
        total_exits = len(exits)
        total_entries = len(entries)
        all_wins = sum(1 for t in closed_trades if t.pnl_usd > 0)
        total_real = sum(t.pnl_usd for t in closed_trades)
        total_unreal = sum(unrealized_by_broker.values())
        overall_wr = all_wins / total_exits if total_exits else None
        avg_hold = sum(hold_hours_list) / len(hold_hours_list) if hold_hours_list else 0.0
        avg_slip = sum(slippages) / len(slippages) if slippages else 0.0

        # ---- Top / bottom trades ---------------------------------------------
        sorted_trades = sorted(closed_trades, key=lambda t: t.pnl_usd, reverse=True)
        top_trades = [self._trade_to_dict(t) for t in sorted_trades[:3]]
        bottom_trades = [self._trade_to_dict(t) for t in sorted_trades[-3:] if t.pnl_usd < 0]

        # ---- Governor --------------------------------------------------------
        gov_tier = "unknown"
        gov_sharpe = None
        if governor_snapshot is not None:
            try:
                gov_tier = str(getattr(governor_snapshot, "tier", "unknown"))
                if hasattr(governor_snapshot, "sharpe"):
                    gov_sharpe = float(governor_snapshot.sharpe) if not math.isnan(governor_snapshot.sharpe) else None
            except Exception:
                pass

        # ---- Recommendations -------------------------------------------------
        recommendations = self._generate_recommendations(
            regime_map=regime_map,
            broker_map=broker_map,
            overall_wr=overall_wr,
            exit_reasons=exit_reasons,
            gov_tier=gov_tier,
            avg_slip=avg_slip,
        )

        return EODReport(
            report_date=today.isoformat(),
            generated_at=datetime.utcnow().isoformat() + "Z",
            total_trades=total_exits,
            total_entries=total_entries,
            total_exits=total_exits,
            total_realized_pnl=round(total_real, 2),
            total_unrealized_pnl=round(total_unreal, 2),
            overall_win_rate=round(overall_wr, 3) if overall_wr is not None else None,
            avg_hold_hours=round(avg_hold, 2),
            avg_slippage_bps=round(avg_slip, 2),
            by_broker={
                b: {
                    "realized_pnl": round(bd.realized_pnl, 2),
                    "unrealized_pnl": round(bd.unrealized_pnl, 2),
                    "net_pnl": round(bd.realized_pnl + bd.unrealized_pnl, 2),
                    "trades": bd.trades,
                    "win_rate": round(bd.win_rate, 3) if bd.win_rate is not None else None,
                }
                for b, bd in broker_map.items()
            },
            by_regime={
                r: {
                    "trades": rd.trades,
                    "win_rate": round(rd.win_rate, 3) if rd.win_rate is not None else None,
                    "pnl_usd": round(rd.pnl_usd, 2),
                }
                for r, rd in sorted(regime_map.items(), key=lambda x: -x[1].trades)
            },
            top_trades=top_trades,
            bottom_trades=bottom_trades,
            exit_reason_summary=dict(exit_reasons),
            governor_tier=gov_tier,
            governor_sharpe=round(gov_sharpe, 3) if gov_sharpe is not None else None,
            recommendations=recommendations,
        )

    def save(self, report: EODReport) -> Path:
        """Write report JSON to data/eod_reports/YYYYMMDD_digest.json."""
        path = self._reports_dir / f"{report.report_date}_digest.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2, default=str)
            logger.info("EOD digest saved → %s", path)
        except Exception as exc:
            logger.warning("EOD digest save failed: %s", exc)
        return path

    def log_summary(self, report: EODReport) -> None:
        """Write a human-readable summary to the main log."""
        wr_str = f"{report.overall_win_rate:.1%}" if report.overall_win_rate is not None else "n/a"
        total_net = report.total_realized_pnl + report.total_unrealized_pnl
        lines = [
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  📊 EOD PERFORMANCE DIGEST  {report.report_date}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Net P&L : ${total_net:+.2f}  (realized ${report.total_realized_pnl:+.2f}  unrealized ${report.total_unrealized_pnl:+.2f})",
            f"  Trades  : {report.total_exits} exits  |  Win rate: {wr_str}  |  Avg hold: {report.avg_hold_hours:.1f}h",
            f"  Slippage: {report.avg_slippage_bps:.1f} bps avg  |  Governor: {report.governor_tier.upper()}" +
            (f"  Sharpe {report.governor_sharpe:.2f}" if report.governor_sharpe is not None else ""),
        ]
        # Broker breakdown
        for b, bd in report.by_broker.items():
            wr = f"{bd['win_rate']:.1%}" if bd.get("win_rate") is not None else "n/a"
            lines.append(f"  {b.upper():8s}: ${bd['net_pnl']:+.2f}  ({bd['trades']} trades  WR {wr})")
        # Regime breakdown (top 3)
        lines.append("  ─── by regime ───")
        for r, rd in list(report.by_regime.items())[:4]:
            wr = f"{rd['win_rate']:.1%}" if rd.get("win_rate") is not None else "n/a"
            lines.append(f"  {r:12s}: {rd['trades']} trades  WR {wr}  P&L ${rd['pnl_usd']:+.2f}")
        # Top winner
        if report.top_trades:
            t = report.top_trades[0]
            lines.append(f"  🏆 Best : {t['symbol']} ${t['pnl_usd']:+.2f} ({t['pnl_pct']:+.2%})  [{t['exit_reason']}]")
        # Worst loser
        if report.bottom_trades:
            t = report.bottom_trades[-1]
            lines.append(f"  💀 Worst: {t['symbol']} ${t['pnl_usd']:+.2f} ({t['pnl_pct']:+.2%})  [{t['exit_reason']}]")
        # Recommendations
        if report.recommendations:
            lines.append("  ─── recommendations ───")
            for rec in report.recommendations[:4]:
                lines.append(f"  ⚡ {rec}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for line in lines:
            logger.info(line)

    def load_latest(self, days_back: int = 7) -> List[EODReport]:
        """Load the most recent N daily reports from disk."""
        reports = []
        today = date.today()
        for i in range(days_back):
            d = today - timedelta(days=i)
            path = self._reports_dir / f"{d.isoformat()}_digest.json"
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    reports.append(data)
                except Exception:
                    pass
        return reports

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_audit(self, day: date) -> Tuple[List[Dict], List[Dict], List[float]]:
        """Load exits, entries, and slippages from today's audit JSONL."""
        exits: List[Dict] = []
        entries: List[Dict] = []
        slippages: List[float] = []

        for uid in [self.user_id, f"{self.user_id}-1"]:
            path = self.data_dir / "users" / uid / "audit" / f"trade_audit_{day.strftime('%Y%m%d')}.jsonl"
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        event = str(row.get("event", "")).upper()
                        slip = row.get("slippage_bps")
                        if slip is not None:
                            try:
                                slippages.append(float(slip))
                            except (ValueError, TypeError):
                                pass
                        if event == "EXIT":
                            exits.append(row)
                        elif event == "ENTRY":
                            entries.append(row)
            except Exception as exc:
                logger.debug("EOD digest: error reading audit %s: %s", path, exc)

        return exits, entries, slippages

    def _generate_recommendations(
        self,
        regime_map: Dict[str, RegimeDigest],
        broker_map: Dict[str, BrokerDigest],
        overall_wr: Optional[float],
        exit_reasons: Dict[str, int],
        gov_tier: str,
        avg_slip: float,
    ) -> List[str]:
        recs: List[Tuple[int, str]] = []  # (priority, text)

        # Governor tier
        if gov_tier in ("red", "emergency"):
            recs.append((0, f"Governor is {gov_tier.upper()} — halve position sizes until recovery"))

        # Overall win rate
        if overall_wr is not None:
            if overall_wr < self.win_rate_urgent:
                recs.append((1, f"Overall win rate {overall_wr:.1%} is critically low (<{self.win_rate_urgent:.0%}) — review entry thresholds immediately"))
            elif overall_wr < self.win_rate_warn:
                recs.append((2, f"Overall win rate {overall_wr:.1%} is below target ({self.win_rate_warn:.0%}) — monitor closely"))

        # Per-regime win rates — suggest threshold raises
        regime_threshold_map = {
            "bull": ("SIGNAL_THRESHOLDS_BY_REGIME[bull]", 0.14, 0.02),
            "neutral": ("SIGNAL_THRESHOLDS_BY_REGIME[neutral]", 0.18, 0.02),
            "bear": ("SIGNAL_THRESHOLDS_BY_REGIME[bear]", 0.21, 0.02),
            "strong_bear": ("SIGNAL_THRESHOLDS_BY_REGIME[strong_bear]", 0.24, 0.02),
            "volatile": ("SIGNAL_THRESHOLDS_BY_REGIME[volatile]", 0.20, 0.02),
        }
        for regime, rd in regime_map.items():
            if rd.trades < 3:
                continue
            wr = rd.win_rate
            if wr is None:
                continue
            if regime in regime_threshold_map and wr < self.win_rate_warn:
                key, cur_thresh, step = regime_threshold_map[regime]
                new_thresh = round(cur_thresh + step, 3)
                priority = 1 if wr < self.win_rate_urgent else 2
                recs.append((priority, f"{regime} regime WR {wr:.1%} ({rd.trades} trades) — raise {key} from {cur_thresh} → {new_thresh}"))

        # Stop loss heavy losses
        stop_exits = exit_reasons.get("stop_loss", 0)
        excellence_exits = exit_reasons.get("excellence", 0)
        total_exits = sum(exit_reasons.values())
        if total_exits > 0 and stop_exits / max(total_exits, 1) > 0.4:
            recs.append((2, f"Stop-loss exits are {stop_exits/total_exits:.0%} of all exits — consider wider stops or smaller sizes"))

        if total_exits > 0 and excellence_exits / max(total_exits, 1) < 0.15 and total_exits >= 5:
            recs.append((3, "Excellence exits <15% of total — signal decay detection may be under-triggering"))

        # High slippage
        if avg_slip > 20:
            recs.append((2, f"Avg slippage {avg_slip:.1f} bps is elevated — consider TWAP for orders >$5K"))
        elif avg_slip > 10:
            recs.append((3, f"Avg slippage {avg_slip:.1f} bps — watch for liquidity issues on smaller cap names"))

        # Sort by priority and return text
        recs.sort(key=lambda x: x[0])
        return [r[1] for r in recs]

    @staticmethod
    def _bucket_exit_reason(reason: str) -> str:
        reason_lower = reason.lower()
        if "stop" in reason_lower:
            return "stop_loss"
        if "excellence" in reason_lower or "signal" in reason_lower:
            return "excellence"
        if "hedge" in reason_lower or "corr" in reason_lower:
            return "hedge_force"
        if "profit" in reason_lower or "ratchet" in reason_lower:
            return "profit_target"
        if "hold" in reason_lower or "max" in reason_lower:
            return "max_hold"
        if "manual" in reason_lower:
            return "manual"
        return "other"

    @staticmethod
    def _trade_to_dict(t: TradeSummary) -> Dict[str, Any]:
        return {
            "symbol": t.symbol,
            "side": t.side,
            "pnl_usd": round(t.pnl_usd, 2),
            "pnl_pct": round(t.pnl_pct, 4),
            "exit_reason": t.exit_reason,
            "regime": t.regime,
            "broker": t.broker,
            "hold_hours": t.hold_hours,
            "signal": round(t.signal, 4),
            "confidence": round(t.confidence, 4),
        }

    @staticmethod
    def _is_crypto(sym: str) -> bool:
        sym_upper = sym.upper()
        return sym_upper.startswith("CRYPTO:") or "/USD" in sym_upper or "BTC" in sym_upper


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_generator: Optional[EODDigestGenerator] = None


def get_eod_generator(data_dir: "str | Path" = "data") -> EODDigestGenerator:
    global _generator
    if _generator is None:
        _generator = EODDigestGenerator(data_dir=data_dir)
    return _generator
