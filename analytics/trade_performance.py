"""
analytics/trade_performance.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calculates core strategy performance diagnostics from Apex trade history.

Supported input sources (auto-detected, in priority order):
  1. data/backtest_trades.csv      – output from run_backtest_with_synthetic_data.py
  2. data/trade_diagnostics.jsonl  – live diagnostics; pnl_pct + hold_hours per trade
  3. data/trades.csv               – flat CSV with pnl column
  4. data/apex_saas.db (orders)    – SQLite fallback (no pnl, limited utility)
  5. Any custom CSV path passed by the caller

Metrics produced:
  ● Win Rate              – % of completed trades that closed profitably
  ● Profit Factor         – Gross wins / Gross losses (in absolute $ or pct)
  ● Avg Win / Avg Loss    – Mean magnitude of winning vs. losing trades
  ● Time in Trade         – Average hold duration for winners vs. losers (hours/days)
  ● Maximum Drawdown      – Deepest peak-to-valley drop from equity_curve.csv
                            (falls back to cumulative PnL series from trades)
  ● Expectancy per trade  – Expected value of one random trade
  ● Regime Breakdown      – Win rate + PF per market regime (bull/bear/neutral/…)
  ● Confidence Correlation– Win rate + avg PnL per confidence decile
  ● Rolling Windows       – 7 / 14 / 30-day win rate + net PnL (decay detection)

Usage:
  from analytics.trade_performance import TradePerformanceAnalytics
  report = TradePerformanceAnalytics().run()
  print(report.summary())
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Internal Trade model
# ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    pnl: float                    # $ or pnl_pct (normalised per source)
    hold_hours: Optional[float]   # None when not available
    exit_reason: Optional[str]
    timestamp: Optional[datetime]
    regime: Optional[str] = None
    confidence: Optional[float] = None  # 0-1 signal confidence at entry

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def is_loser(self) -> bool:
        return self.pnl < 0


# ──────────────────────────────────────────────────────────────
# Metrics container
# ──────────────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    win_rate: float = 0.0
    profit_factor: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0

    best_trade: float = 0.0
    worst_trade: float = 0.0

    avg_hold_winners_h: Optional[float] = None
    avg_hold_losers_h: Optional[float] = None

    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0

    expectancy: float = 0.0

    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0

    data_source: str = "unknown"
    pnl_unit: str = "pct"

    by_symbol: dict = field(default_factory=dict)
    loser_exit_reasons: dict = field(default_factory=dict)

    # ── New analytical layers ───────────────────────────────
    by_regime: dict = field(default_factory=dict)
    """{ regime: {count, wins, gross_profit, gross_loss, net, win_rate, pf} }"""

    by_confidence_bucket: dict = field(default_factory=dict)
    """{ bucket_label: {count, wins, win_rate, avg_pnl} }"""

    rolling_windows: list = field(default_factory=list)
    """[ {days, count, wins, win_rate, net_pnl} ]"""

    # ──────────────────────────────────────────────────────────────
    # Summary output
    # ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        RESET  = "\033[0m"
        BOLD   = "\033[1m"
        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        CYAN   = "\033[96m"
        DIM    = "\033[2m"

        def color_val(val: float, good_positive: bool = True) -> str:
            if val > 0:
                c = GREEN if good_positive else RED
            elif val < 0:
                c = RED if good_positive else GREEN
            else:
                c = DIM
            return f"{c}{val:+.4f}{RESET}"

        def pct_color(pct: float) -> str:
            c = GREEN if pct >= 50 else (YELLOW if pct >= 40 else RED)
            return f"{c}{pct:.1f}%{RESET}"

        pf_color  = GREEN if self.profit_factor >= 1.0 else RED
        wr_color  = GREEN if self.win_rate >= 50 else RED
        dd_color  = RED if self.max_drawdown_pct > 10 else (YELLOW if self.max_drawdown_pct > 5 else GREEN)
        unit = "%" if self.pnl_unit == "pct" else "$"

        L: List[str] = [
            "",
            f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗{RESET}",
            f"{BOLD}{CYAN}║         APEX TRADING — PERFORMANCE DIAGNOSTICS       ║{RESET}",
            f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════╝{RESET}",
            f"  Data source : {DIM}{self.data_source}{RESET}   |   PnL unit: {unit}",
            "",
            f"{BOLD}── TRADE SUMMARY ──────────────────────────────────────{RESET}",
            f"  Total closed trades : {BOLD}{self.total_trades}{RESET}",
            f"  Winners             : {GREEN}{self.winning_trades}{RESET}",
            f"  Losers              : {RED}{self.losing_trades}{RESET}",
            f"  Breakeven           : {DIM}{self.breakeven_trades}{RESET}",
            "",
            f"{BOLD}── WIN / LOSS RATES ────────────────────────────────────{RESET}",
            f"  Win Rate            : {wr_color}{self.win_rate:.1f}%{RESET}",
            f"  Profit Factor       : {pf_color}{self.profit_factor:.2f}{RESET}"
            + (f"  {DIM}(< 1.0 = net loss territory){RESET}" if self.profit_factor < 1.0 else ""),
            "",
            f"{BOLD}── RISK / REWARD ───────────────────────────────────────{RESET}",
            f"  Avg Win             : {GREEN}+{self.avg_win:.4f}{unit}{RESET}",
            f"  Avg Loss            : {RED}-{self.avg_loss:.4f}{unit}{RESET}",
            f"  Win/Loss Ratio      : {BOLD}{self.win_loss_ratio:.2f}x{RESET}"
            + (f"  {DIM}(< 1.0 means losers dwarf winners){RESET}" if self.win_loss_ratio < 1.0 else ""),
            f"  Best Trade          : {GREEN}+{self.best_trade:.4f}{unit}{RESET}",
            f"  Worst Trade         : {RED}{self.worst_trade:.4f}{unit}{RESET}",
            "",
            f"{BOLD}── EXPECTANCY ──────────────────────────────────────────{RESET}",
            f"  Expected Value/trade: {color_val(self.expectancy)}{unit}",
            f"  Gross Profit        :  {self.gross_profit:.4f}{unit}",
            f"  Gross Loss          : -{self.gross_loss:.4f}{unit}",
            f"  Net PnL             : {color_val(self.net_pnl)}{unit}",
            "",
        ]

        # Time in Trade
        if self.avg_hold_winners_h is not None or self.avg_hold_losers_h is not None:
            L.append(f"{BOLD}── TIME IN TRADE ───────────────────────────────────────{RESET}")
            if self.avg_hold_winners_h is not None:
                wh = self.avg_hold_winners_h
                L.append(f"  Avg hold (winners)  : {GREEN}{wh:.2f} h{RESET}  ({wh*60:.0f} min)")
            if self.avg_hold_losers_h is not None:
                lh = self.avg_hold_losers_h
                flag = (f"  {YELLOW}⚠ losers held longer{RESET}"
                        if (self.avg_hold_winners_h and lh > self.avg_hold_winners_h) else "")
                L.append(f"  Avg hold (losers)   : {RED}{lh:.2f} h{RESET}  ({lh*60:.0f} min){flag}")
            L.append("")

        # Drawdown
        L += [
            f"{BOLD}── DRAWDOWN ────────────────────────────────────────────{RESET}",
            f"  Max Drawdown        : {dd_color}{self.max_drawdown_pct:.2f}%{RESET}"
            + (f"  ({self.max_drawdown_abs:.4f}{unit})" if self.max_drawdown_abs else ""),
            "",
        ]

        # ── Regime breakdown ────────────────────────────────────────────
        if self.by_regime:
            L += [
                f"{BOLD}── BY REGIME ───────────────────────────────────────────{RESET}",
                f"  {'Regime':<16} {'Trades':>6} {'WinRate':>8} {'NetPnL':>10} {'PF':>6}",
                "  " + "─" * 50,
            ]
            for regime, s in sorted(self.by_regime.items(), key=lambda x: -x[1].get("net", 0)):
                wr = s["win_rate"]
                pf = s.get("pf", 0.0)
                pf_str = f"{pf:.2f}" if pf != float("inf") else "  ∞"
                net_col = GREEN if s["net"] > 0 else RED
                wr_col  = GREEN if wr >= 50 else (YELLOW if wr >= 40 else RED)
                L.append(
                    f"  {regime:<16} {s['count']:>6} {wr_col}{wr:>7.1f}%{RESET} "
                    f"{net_col}{s['net']:>+10.4f}{RESET} {pf_str:>6}"
                )
            L.append("")

        # ── Confidence correlation ──────────────────────────────────────
        if self.by_confidence_bucket:
            L += [
                f"{BOLD}── CONFIDENCE → OUTCOME ────────────────────────────────{RESET}",
                f"  Is higher confidence correlated with better trades?",
                f"  {'Confidence':>12} {'Trades':>6} {'WinRate':>8} {'AvgPnL':>10}",
                "  " + "─" * 44,
            ]
            for label, s in sorted(self.by_confidence_bucket.items()):
                avg_pnl = s["avg_pnl"]
                pnl_col = GREEN if avg_pnl > 0 else RED
                L.append(
                    f"  {label:>12} {s['count']:>6} {pct_color(s['win_rate']):>8} "
                    f"{pnl_col}{avg_pnl:>+10.4f}{RESET}"
                )
            # Predictiveness note
            if len(self.by_confidence_bucket) >= 2:
                buckets = sorted(self.by_confidence_bucket.items())
                low_wr  = buckets[0][1]["win_rate"]
                high_wr = buckets[-1][1]["win_rate"]
                delta   = high_wr - low_wr
                if delta > 10:
                    L.append(f"  {GREEN}✓ Confidence is predictive (+{delta:.1f}pp from lowest → highest bucket){RESET}")
                elif delta < -5:
                    L.append(f"  {RED}✗ Confidence is INVERSELY correlated — high-confidence trades are losses{RESET}")
                else:
                    L.append(f"  {YELLOW}⚠ Confidence shows little predictive power ({delta:+.1f}pp across buckets){RESET}")
            L.append("")

        # ── Rolling windows ─────────────────────────────────────────────
        if self.rolling_windows:
            L += [
                f"{BOLD}── ROLLING PERFORMANCE (is the strategy decaying?) ─────{RESET}",
                f"  {'Window':>8} {'Trades':>6} {'WinRate':>8} {'NetPnL':>10}",
                "  " + "─" * 38,
            ]
            for w in self.rolling_windows:
                net_col = GREEN if w["net_pnl"] > 0 else RED
                window_label = f"Last {w['days']}d"
                L.append(
                    f"  {window_label:>8} {w['count']:>6} {pct_color(w['win_rate']):>8} "
                    f"{net_col}{w['net_pnl']:>+10.4f}{RESET}"
                )
            # Detect decay
            if len(self.rolling_windows) >= 2:
                short = self.rolling_windows[0]   # 7d
                long_ = self.rolling_windows[-1]  # 30d
                if short["count"] >= 3 and long_["count"] >= 5:
                    diff = short["win_rate"] - long_["win_rate"]
                    if diff < -15:
                        L.append(f"  {RED}⚠ DECAY DETECTED: last-{short['days']}d win rate is {abs(diff):.1f}pp below last-{long_['days']}d.{RESET}")
            L.append("")

        # ── Diagnosis ──────────────────────────────────────────────────
        L.append(f"{BOLD}── DIAGNOSIS ───────────────────────────────────────────{RESET}")
        issues = []
        if self.win_rate < 40:
            issues.append(f"  {RED}✗ LOW WIN RATE ({self.win_rate:.1f}%){RESET}: most trades close as losers.")
        if self.profit_factor < 1.0:
            issues.append(f"  {RED}✗ NEGATIVE PROFIT FACTOR ({self.profit_factor:.2f}){RESET}: losers outweigh winners in gross terms.")
        if self.win_loss_ratio < 1.0 and self.win_rate < 60:
            issues.append(
                f"  {YELLOW}⚠ POOR RISK/REWARD ({self.win_loss_ratio:.2f}x){RESET}: "
                "avg winner < avg loser — need higher win rate OR bigger RR."
            )
        if (self.avg_hold_winners_h is not None and self.avg_hold_losers_h is not None
                and self.avg_hold_losers_h > self.avg_hold_winners_h * 1.5):
            issues.append(
                f"  {RED}✗ HOLDING LOSERS TOO LONG{RESET}: losers held "
                f"{self.avg_hold_losers_h:.1f}h vs winners {self.avg_hold_winners_h:.1f}h."
            )
        if self.max_drawdown_pct > 20:
            issues.append(f"  {RED}✗ SEVERE DRAWDOWN ({self.max_drawdown_pct:.1f}%){RESET}: risk of ruin elevated.")
        elif self.max_drawdown_pct > 10:
            issues.append(f"  {YELLOW}⚠ ELEVATED DRAWDOWN ({self.max_drawdown_pct:.1f}%){RESET}: tighten position sizing.")
        if self.expectancy < 0:
            issues.append(
                f"  {RED}✗ NEGATIVE EXPECTANCY ({self.expectancy:.4f}{unit}/trade){RESET}: "
                "system is mathematically expected to lose money per trade."
            )

        if not issues:
            L.append(f"  {GREEN}✓ No major red flags detected in the available trade sample.{RESET}")
        else:
            L += issues

        # Top loser exit reasons
        if self.loser_exit_reasons:
            L += [
                "",
                f"{BOLD}── TOP EXIT REASONS (LOSERS) ───────────────────────────{RESET}",
            ]
            for reason, count in sorted(self.loser_exit_reasons.items(), key=lambda x: -x[1])[:5]:
                L.append(f"  {count:3d}x  {reason}")

        # Per-symbol (compact: only top 10 by trade count)
        if self.by_symbol:
            L += [
                "",
                f"{BOLD}── BY SYMBOL (top 10) ──────────────────────────────────{RESET}",
                f"  {'Symbol':<22} {'Trades':>6} {'WinRate':>8} {'NetPnL':>10} {'PF':>6}",
                "  " + "─" * 56,
            ]
            top10 = sorted(self.by_symbol.items(), key=lambda x: -x[1]["count"])[:10]
            for sym, s in sorted(top10, key=lambda x: -x[1]["net"]):
                wr  = s["wins"] / s["count"] * 100 if s["count"] else 0
                pf  = s["gross_profit"] / s["gross_loss"] if s["gross_loss"] else float("inf")
                pf_str = f"{pf:.2f}" if pf != float("inf") else "  ∞"
                net_col = GREEN if s["net"] > 0 else RED
                L.append(
                    f"  {sym:<22} {s['count']:>6} {pct_color(wr):>8} "
                    f"{net_col}{s['net']:>+10.4f}{RESET} {pf_str:>6}"
                )

        L.append("")
        return "\n".join(L)

    def to_dict(self) -> dict:
        return {
            "total_trades":         self.total_trades,
            "winning_trades":       self.winning_trades,
            "losing_trades":        self.losing_trades,
            "breakeven_trades":     self.breakeven_trades,
            "win_rate_pct":         round(self.win_rate, 2),
            "profit_factor":        round(self.profit_factor, 4),
            "avg_win":              round(self.avg_win, 6),
            "avg_loss":             round(self.avg_loss, 6),
            "win_loss_ratio":       round(self.win_loss_ratio, 4),
            "best_trade":           round(self.best_trade, 6),
            "worst_trade":          round(self.worst_trade, 6),
            "expectancy":           round(self.expectancy, 6),
            "gross_profit":         round(self.gross_profit, 6),
            "gross_loss":           round(self.gross_loss, 6),
            "net_pnl":              round(self.net_pnl, 6),
            "avg_hold_winners_h":   round(self.avg_hold_winners_h, 4) if self.avg_hold_winners_h is not None else None,
            "avg_hold_losers_h":    round(self.avg_hold_losers_h, 4) if self.avg_hold_losers_h is not None else None,
            "max_drawdown_pct":     round(self.max_drawdown_pct, 4),
            "max_drawdown_abs":     round(self.max_drawdown_abs, 4),
            "data_source":          self.data_source,
            "pnl_unit":             self.pnl_unit,
            "by_regime":            self.by_regime,
            "by_confidence_bucket": self.by_confidence_bucket,
            "rolling_windows":      self.rolling_windows,
        }


# ──────────────────────────────────────────────────────────────
# Analytics engine
# ──────────────────────────────────────────────────────────────

class TradePerformanceAnalytics:
    """
    Loads trade history from Apex data directory and computes all performance
    diagnostics including regime-split, confidence-correlation, and rolling windows.

    Parameters
    ----------
    data_dir    : Root of Apex data directory.  Defaults to <project>/data.
    csv_path    : Override: use this CSV file instead of auto-detecting.
    jsonl_path  : Override: use this JSONL diagnostics file.
    equity_csv  : Override: equity curve CSV for drawdown calculation.
    min_trades  : Warn when fewer trades are found.
    rolling_days: Window sizes (days) for rolling performance breakdown.
    conf_buckets: Confidence bucket boundaries for correlation analysis.
    """

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _DATA_DIR     = _PROJECT_ROOT / "data"

    def __init__(
        self,
        data_dir:     Optional[str | Path] = None,
        csv_path:     Optional[str | Path] = None,
        jsonl_path:   Optional[str | Path] = None,
        equity_csv:   Optional[str | Path] = None,
        min_trades:   int = 1,
        rolling_days: List[int] = None,
        conf_buckets: List[float] = None,
    ):
        self.data_dir   = Path(data_dir) if data_dir else self._DATA_DIR
        self.csv_path   = Path(csv_path)   if csv_path   else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path else self.data_dir / "trade_diagnostics.jsonl"
        self.equity_csv = Path(equity_csv) if equity_csv else self.data_dir / "equity_curve.csv"
        self.db_path    = self.data_dir / "apex_saas.db"
        self.min_trades = min_trades
        self.rolling_days  = rolling_days  or [7, 14, 30]
        self.conf_buckets  = conf_buckets  or [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
        # CSV priority: explicit > backtest_trades.csv > trades.csv
        self._auto_csv_candidates = [
            self.data_dir / "backtest_trades.csv",
            self.data_dir / "trades.csv",
        ]

    # ── Public interface ────────────────────────────────────────

    def run(self) -> PerformanceReport:
        trades, source = self._load_trades()
        if not trades:
            report = PerformanceReport(data_source=source or "no data found")
            return report

        report = self._compute_metrics(trades)
        report.data_source = source

        # Override drawdown from equity curve if available
        dd_pct, dd_abs = self._compute_drawdown_from_equity()
        if dd_pct is not None:
            report.max_drawdown_pct = dd_pct
            report.max_drawdown_abs = dd_abs

        # Regime breakdown
        report.by_regime = self._compute_regime_breakdown(trades)

        # Confidence correlation
        report.by_confidence_bucket = self._compute_confidence_correlation(trades)

        # Rolling windows
        report.rolling_windows = self._compute_rolling_windows(trades)

        if len(trades) < self.min_trades:
            print(f"[WARNING] Only {len(trades)} completed trade(s). Metrics may not be meaningful.")

        return report

    # ── Loaders ────────────────────────────────────────────────

    def _load_trades(self) -> Tuple[List[Trade], str]:
        # Explicit CSV override
        if self.csv_path:
            trades, source = self._load_from_csv(self.csv_path)
            if trades:
                return trades, source

        # JSONL (live diagnostics)
        trades, source = self._load_from_jsonl()
        if trades:
            return trades, source

        # Auto-detect CSV candidates (backtest_trades.csv >> trades.csv)
        for candidate in self._auto_csv_candidates:
            trades, source = self._load_from_csv(candidate)
            if trades:
                return trades, source

        # SQLite fallback
        return self._load_from_db()

    def _load_from_jsonl(self) -> Tuple[List[Trade], str]:
        if not self.jsonl_path.exists():
            return [], ""
        trades: List[Trade] = []
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("action") != "entered":
                    continue
                pnl = rec.get("pnl_pct")
                if pnl is None:
                    continue
                hold = rec.get("hold_hours")
                ts = None
                raw_ts = rec.get("ts")
                if raw_ts:
                    try:
                        ts = datetime.fromtimestamp(float(raw_ts))
                    except Exception:
                        pass
                trades.append(Trade(
                    symbol=rec.get("symbol", "UNKNOWN"),
                    pnl=float(pnl),
                    hold_hours=float(hold) if hold is not None else None,
                    exit_reason=rec.get("exit_reason"),
                    timestamp=ts,
                    regime=rec.get("regime"),
                    confidence=rec.get("confidence"),
                ))
        return trades, f"trade_diagnostics.jsonl ({len(trades)} closed trades)"

    def _load_from_csv(self, path: Path) -> Tuple[List[Trade], str]:
        if not path.exists():
            return [], ""
        trades: List[Trade] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return [], f"{path.name} (empty)"
            fnames = [c.lower() for c in reader.fieldnames]
            pnl_col = next(
                (c for c in ("pnl_percent", "pnl_pct", "pnl", "profit", "return", "net_pnl") if c in fnames),
                None,
            )
            if pnl_col is None:
                return [], f"{path.name} (no PnL column found)"
            # Map lower-cased name to actual header name
            col_map = {c.lower(): c for c in reader.fieldnames}
            actual_pnl_col = col_map.get(pnl_col, pnl_col)

            for row in reader:
                raw = row.get(actual_pnl_col, "").strip()
                if not raw or raw in ("", "None", "null"):
                    continue
                try:
                    pnl = float(raw)
                except ValueError:
                    continue

                hold = None
                for hk in ("hold_hours", "hold_days"):
                    hkey = col_map.get(hk)
                    if hkey and row.get(hkey, "").strip():
                        try:
                            v = float(row[hkey])
                            hold = v * 24 if hk == "hold_days" else v
                            break
                        except (ValueError, TypeError):
                            pass

                ts = None
                for tk in ("exit_date", "entry_date", "timestamp", "exit_time", "close_time", "time"):
                    tkey = col_map.get(tk)
                    if tkey and row.get(tkey, "").strip():
                        try:
                            ts = datetime.fromisoformat(row[tkey].strip())
                            break
                        except Exception:
                            pass

                conf = None
                ckey = col_map.get("confidence")
                if ckey and row.get(ckey, "").strip():
                    try:
                        conf = float(row[ckey])
                    except (ValueError, TypeError):
                        pass

                regime = None
                rkey = col_map.get("regime")
                if rkey:
                    regime = row.get(rkey) or None

                exit_reason = None
                erkey = col_map.get("exit_reason")
                if erkey:
                    exit_reason = row.get(erkey) or None

                sym_key = col_map.get("symbol", "symbol")
                trades.append(Trade(
                    symbol=row.get(sym_key, "UNKNOWN"),
                    pnl=pnl,
                    hold_hours=hold,
                    exit_reason=exit_reason,
                    timestamp=ts,
                    regime=regime,
                    confidence=conf,
                ))
        return trades, f"{path.name} ({len(trades)} trades)"

    def _load_from_db(self) -> Tuple[List[Trade], str]:
        if not self.db_path.exists():
            return [], "no data source found"
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute("SELECT count(*) FROM orders WHERE status = 'filled'")
            filled = cur.fetchone()[0]
            conn.close()
            return [], (
                f"apex_saas.db ({filled} filled orders, but PnL not stored in orders table — "
                "run scripts/run_backtest_with_synthetic_data.py to generate data/backtest_trades.csv)"
            )
        except Exception as exc:
            return [], f"apex_saas.db (error: {exc})"

    # ── Core metrics ────────────────────────────────────────────

    def _compute_metrics(self, trades: List[Trade]) -> PerformanceReport:
        r = PerformanceReport(pnl_unit="pct")
        winners = [t for t in trades if t.is_winner]
        losers  = [t for t in trades if t.is_loser]
        be      = [t for t in trades if not t.is_winner and not t.is_loser]

        r.total_trades     = len(trades)
        r.winning_trades   = len(winners)
        r.losing_trades    = len(losers)
        r.breakeven_trades = len(be)

        r.win_rate     = (len(winners) / len(trades) * 100) if trades else 0.0
        r.gross_profit = sum(t.pnl for t in winners)
        r.gross_loss   = abs(sum(t.pnl for t in losers))
        r.net_pnl      = r.gross_profit - r.gross_loss
        r.profit_factor = (r.gross_profit / r.gross_loss) if r.gross_loss > 0 else float("inf")

        r.avg_win      = (r.gross_profit / len(winners)) if winners else 0.0
        r.avg_loss     = (r.gross_loss   / len(losers))  if losers  else 0.0
        r.win_loss_ratio = (r.avg_win / r.avg_loss) if r.avg_loss > 0 else float("inf")

        all_pnl = [t.pnl for t in trades]
        r.best_trade  = max(all_pnl)
        r.worst_trade = min(all_pnl)

        wr_dec  = r.winning_trades / r.total_trades
        los_dec = r.losing_trades  / r.total_trades
        r.expectancy = (wr_dec * r.avg_win) - (los_dec * r.avg_loss)

        w_hold = [t.hold_hours for t in winners if t.hold_hours is not None]
        l_hold = [t.hold_hours for t in losers  if t.hold_hours is not None]
        r.avg_hold_winners_h = (sum(w_hold) / len(w_hold)) if w_hold else None
        r.avg_hold_losers_h  = (sum(l_hold) / len(l_hold)) if l_hold else None

        r.max_drawdown_pct, r.max_drawdown_abs = self._compute_drawdown_from_pnl(trades)

        sym_map: dict = {}
        for t in trades:
            s = sym_map.setdefault(t.symbol, {
                "count": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net": 0.0
            })
            s["count"] += 1
            if t.pnl > 0:
                s["wins"] += 1
                s["gross_profit"] += t.pnl
            elif t.pnl < 0:
                s["gross_loss"] += abs(t.pnl)
            s["net"] += t.pnl
        r.by_symbol = sym_map

        for t in losers:
            if t.exit_reason:
                r.loser_exit_reasons[t.exit_reason] = r.loser_exit_reasons.get(t.exit_reason, 0) + 1

        return r

    # ── Regime breakdown ────────────────────────────────────────

    def _compute_regime_breakdown(self, trades: List[Trade]) -> dict:
        """Win rate, net PnL, profit factor per market regime."""
        reg: Dict[str, dict] = {}
        for t in trades:
            r = t.regime or "unknown"
            s = reg.setdefault(r, {"count": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0, "net": 0.0})
            s["count"] += 1
            if t.pnl > 0:
                s["wins"] += 1
                s["gross_profit"] += t.pnl
            elif t.pnl < 0:
                s["gross_loss"] += abs(t.pnl)
            s["net"] += t.pnl
        for s in reg.values():
            s["win_rate"] = (s["wins"] / s["count"] * 100) if s["count"] else 0.0
            s["pf"] = (s["gross_profit"] / s["gross_loss"]) if s["gross_loss"] > 0 else float("inf")
        return reg

    # ── Confidence correlation ──────────────────────────────────

    def _compute_confidence_correlation(self, trades: List[Trade]) -> dict:
        """
        Bin trades by confidence and compute win rate + avg PnL per bucket.
        Reveals whether the ML confidence score is actually predictive.
        """
        trades_with_conf = [t for t in trades if t.confidence is not None]
        if not trades_with_conf:
            return {}

        buckets: Dict[str, dict] = {}
        bounds = self.conf_buckets
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]
            label = f"{lo:.2f}–{hi:.2f}"
            buckets[label] = {"count": 0, "wins": 0, "total_pnl": 0.0}

        for t in trades_with_conf:
            c = t.confidence
            for i in range(len(bounds) - 1):
                if bounds[i] <= c < bounds[i + 1]:
                    label = f"{bounds[i]:.2f}–{bounds[i+1]:.2f}"
                    buckets[label]["count"] += 1
                    if t.is_winner:
                        buckets[label]["wins"] += 1
                    buckets[label]["total_pnl"] += t.pnl
                    break

        result = {}
        for label, s in buckets.items():
            if s["count"] == 0:
                continue
            result[label] = {
                "count":    s["count"],
                "wins":     s["wins"],
                "win_rate": (s["wins"] / s["count"] * 100),
                "avg_pnl":  s["total_pnl"] / s["count"],
            }
        return result

    # ── Rolling windows ─────────────────────────────────────────

    def _compute_rolling_windows(self, trades: List[Trade]) -> list:
        """
        For each window size in self.rolling_days, compute trade metrics
        for trades with timestamp in the last N calendar days.
        Surfaces if the strategy is decaying recently vs. historically.
        """
        now = datetime.now()
        trades_w_ts = [t for t in trades if t.timestamp is not None]
        if not trades_w_ts:
            return []

        rows = []
        for days in sorted(self.rolling_days):
            cutoff = now - timedelta(days=days)
            window = [t for t in trades_w_ts if t.timestamp >= cutoff]
            if not window:
                rows.append({"days": days, "count": 0, "wins": 0, "win_rate": 0.0, "net_pnl": 0.0})
                continue
            wins    = sum(1 for t in window if t.is_winner)
            net     = sum(t.pnl for t in window)
            win_rate = wins / len(window) * 100
            rows.append({"days": days, "count": len(window), "wins": wins, "win_rate": win_rate, "net_pnl": net})
        return rows

    # ── Drawdown helpers ────────────────────────────────────────

    def _compute_drawdown_from_equity(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.equity_csv.exists():
            return None, None
        equities: List[float] = []
        try:
            with open(self.equity_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    col = next((c for c in ("equity", "balance", "nav", "value") if c in row), None)
                    if col is None:
                        continue
                    raw = row[col].strip()
                    if not raw:
                        continue
                    try:
                        equities.append(float(raw))
                    except ValueError:
                        pass
        except Exception:
            return None, None
        if len(set(equities)) <= 1:
            return 0.0, 0.0
        return self._max_drawdown(equities)

    @staticmethod
    def _compute_drawdown_from_pnl(trades: List[Trade]) -> Tuple[float, float]:
        if not trades:
            return 0.0, 0.0
        sorted_trades = sorted(
            (t for t in trades if t.timestamp is not None),
            key=lambda t: t.timestamp,
        )
        if not sorted_trades:
            sorted_trades = trades
        equity: List[float] = [0.0]
        cum = 0.0
        for t in sorted_trades:
            cum += t.pnl
            equity.append(cum)
        return TradePerformanceAnalytics._max_drawdown(equity)

    @staticmethod
    def _max_drawdown(equity: List[float]) -> Tuple[float, float]:
        if not equity:
            return 0.0, 0.0
        peak = equity[0]
        max_dd_abs = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd_abs:
                max_dd_abs = dd
        max_dd_pct = (max_dd_abs / peak * 100) if peak != 0 else 0.0
        return max_dd_pct, max_dd_abs
