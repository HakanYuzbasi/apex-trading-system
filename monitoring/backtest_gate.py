"""
monitoring/backtest_gate.py — Automated Nightly Backtest + Deployment Gate

Reads trade audit JSONL files, computes rolling walk-forward performance metrics,
and maintains a deployment gate state that can switch the engine between LIVE and
PAPER modes when strategy performance degrades.

Gate logic:
    - Roll over last N days of EXIT trades
    - Compute rolling Sharpe, win-rate, avg P&L
    - Compare current window vs previous window
    - If Sharpe degrades by > SHARPE_DEGRADE_THRESH → flag "degraded"
    - If win-rate < WIN_RATE_FLOOR → flag "low_winrate"
    - If two consecutive degraded runs → set mode = "paper"
    - Recovery: two consecutive healthy runs → set mode = "live"

Gate states:
    "live"      — Normal live trading
    "paper"     — Strategy degraded, running in simulation only
    "unknown"   — Not enough data yet

Usage:
    gate = BacktestGate()
    gate.run_evaluation()           # call nightly or on-demand
    state = gate.get_state()        # dict with mode, metrics, history

Config keys:
    BACKTEST_GATE_ENABLED             = True
    BACKTEST_GATE_WINDOW_DAYS         = 30   # rolling window
    BACKTEST_GATE_COMPARE_DAYS        = 15   # comparison window
    BACKTEST_GATE_MIN_TRADES          = 10   # min trades to evaluate
    BACKTEST_GATE_SHARPE_DEGRADE      = 0.30 # absolute Sharpe drop that triggers flag
    BACKTEST_GATE_WIN_RATE_FLOOR      = 0.40 # win rate below this → flag
    BACKTEST_GATE_CONSEC_DEGRADE      = 2    # consecutive bad runs before paper mode
    BACKTEST_GATE_CONSEC_RECOVER      = 2    # consecutive good runs before live mode
    BACKTEST_GATE_STATE_PATH          = "data/backtest_gate_state.json"
    BACKTEST_GATE_HISTORY_LIMIT       = 30   # max history entries to keep
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEF: Dict = {
    "BACKTEST_GATE_ENABLED":         True,
    "BACKTEST_GATE_WINDOW_DAYS":     30,
    "BACKTEST_GATE_COMPARE_DAYS":    15,
    "BACKTEST_GATE_MIN_TRADES":      10,
    "BACKTEST_GATE_SHARPE_DEGRADE":  0.30,
    "BACKTEST_GATE_WIN_RATE_FLOOR":  0.40,
    "BACKTEST_GATE_CONSEC_DEGRADE":  2,
    "BACKTEST_GATE_CONSEC_RECOVER":  2,
    "BACKTEST_GATE_STATE_PATH":      "data/backtest_gate_state.json",
    "BACKTEST_GATE_HISTORY_LIMIT":   30,
}

_AUDIT_GLOB = "trade_audit_*.jsonl"


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


def _sharpe(pnl_list: List[float]) -> float:
    """Annualised Sharpe from a list of per-trade P&L percentages."""
    if len(pnl_list) < 3:
        return 0.0
    arr = np.array(pnl_list, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-9:
        if mean > 1e-9:
            return 10.0
        if mean < -1e-9:
            return -10.0
        return 0.0
    # Annualise assuming ~252 trading days, ~1 trade/day on average
    return float(mean / std * math.sqrt(252))


def _load_exits(
    audit_dir: Path,
    since_days: int,
    before_days: Optional[int] = None,
) -> List[dict]:
    """
    Load EXIT rows from trade audit JSONL files.
    Returns rows with timestamp in (now - since_days, now - before_days].
    If before_days is None, no upper cutoff is applied.
    """
    now = datetime.now(timezone.utc)
    cutoff_old = now - timedelta(days=since_days)
    cutoff_new = now - timedelta(days=before_days) if before_days is not None else None

    rows: List[dict] = []
    for f in sorted(audit_dir.glob(_AUDIT_GLOB)):
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
                            if ts < cutoff_old:
                                continue
                            if cutoff_new is not None and ts >= cutoff_new:
                                continue
                        except ValueError:
                            pass
                    rows.append(row)
        except Exception as e:
            logger.debug("BacktestGate: error reading %s — %s", f, e)
    return rows


@dataclass
class PeriodMetrics:
    trades: int
    win_rate: float
    avg_pnl_pct: float
    sharpe: float
    flags: List[str] = field(default_factory=list)

    def is_healthy(self, win_floor: float) -> bool:
        return self.win_rate >= win_floor and len(self.flags) == 0


@dataclass
class EvalRecord:
    ts: str
    mode: str               # "live" | "paper" | "unknown"
    current: PeriodMetrics
    previous: Optional[PeriodMetrics]
    sharpe_delta: float     # current.sharpe - previous.sharpe (None → 0.0)
    triggered_flags: List[str] = field(default_factory=list)


class BacktestGate:
    """
    Evaluates recent strategy performance and sets the deployment mode.
    Thread-safe for read; single-writer expected (nightly cron / scheduled call).
    """

    def __init__(self, audit_dir: Optional[str] = None, state_path: Optional[str] = None):
        try:
            from config import ApexConfig
            default_audit = str(ApexConfig.DATA_DIR)
        except Exception:
            default_audit = "data"

        self._audit_dir = Path(audit_dir or default_audit)
        self._state_path = Path(state_path or str(_cfg("BACKTEST_GATE_STATE_PATH")))

        self._mode: str = "unknown"
        self._consec_bad: int = 0
        self._consec_good: int = 0
        self._history: List[dict] = []
        self._last_eval_ts: str = ""
        self._last_metrics: Optional[PeriodMetrics] = None

        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Current deployment mode: 'live' | 'paper' | 'unknown'."""
        return self._mode

    @property
    def is_live(self) -> bool:
        return self._mode == "live"

    @property
    def is_paper(self) -> bool:
        return self._mode == "paper"

    def run_evaluation(self) -> EvalRecord:
        """
        Run a full evaluation pass. Call nightly or on-demand.
        Returns an EvalRecord with results; also updates internal state and saves.
        """
        if not _cfg("BACKTEST_GATE_ENABLED"):
            self._mode = "live"
            return self._make_dummy_record("live")

        window = int(_cfg("BACKTEST_GATE_WINDOW_DAYS"))
        compare = int(_cfg("BACKTEST_GATE_COMPARE_DAYS"))
        min_trades = int(_cfg("BACKTEST_GATE_MIN_TRADES"))
        sharpe_degrade = float(_cfg("BACKTEST_GATE_SHARPE_DEGRADE"))
        win_floor = float(_cfg("BACKTEST_GATE_WIN_RATE_FLOOR"))

        # Current window: last `window` days
        all_exits = _load_exits(self._audit_dir, since_days=window)
        current = self._compute_metrics(all_exits, win_floor)

        # Previous window: `window+compare` to `window` days ago (non-overlapping)
        prev_exits = _load_exits(self._audit_dir, since_days=window + compare, before_days=window)
        previous = self._compute_metrics(prev_exits, win_floor) if prev_exits else None

        sharpe_delta = 0.0
        triggered: List[str] = list(current.flags)

        if previous is not None and previous.trades >= min_trades:
            sharpe_delta = current.sharpe - previous.sharpe
            if sharpe_delta < -sharpe_degrade and "sharpe_degrade" not in triggered:
                triggered.append("sharpe_degrade")

        # Only evaluate if enough data
        if current.trades < min_trades:
            mode = "unknown"
        else:
            if triggered:
                self._consec_bad += 1
                self._consec_good = 0
            else:
                self._consec_good += 1
                self._consec_bad = 0

            consec_degrade = int(_cfg("BACKTEST_GATE_CONSEC_DEGRADE"))
            consec_recover = int(_cfg("BACKTEST_GATE_CONSEC_RECOVER"))

            if self._consec_bad >= consec_degrade:
                mode = "paper"
            elif self._consec_good >= consec_recover:
                mode = "live"
            else:
                mode = self._mode  # keep current mode

        self._mode = mode
        self._last_eval_ts = datetime.now(timezone.utc).isoformat()
        self._last_metrics = current

        record = EvalRecord(
            ts=self._last_eval_ts,
            mode=mode,
            current=current,
            previous=previous,
            sharpe_delta=round(sharpe_delta, 4),
            triggered_flags=triggered,
        )

        self._history.append(asdict(record))
        limit = int(_cfg("BACKTEST_GATE_HISTORY_LIMIT"))
        if len(self._history) > limit:
            self._history = self._history[-limit:]

        self._save()

        if triggered:
            logger.warning(
                "BacktestGate: flags=%s sharpe=%.3f win_rate=%.1f%% trades=%d → mode=%s",
                triggered, current.sharpe, current.win_rate * 100, current.trades, mode,
            )
        else:
            logger.info(
                "BacktestGate: healthy sharpe=%.3f win_rate=%.1f%% trades=%d → mode=%s",
                current.sharpe, current.win_rate * 100, current.trades, mode,
            )

        return record

    def get_state(self) -> dict:
        """Return serialisable gate state for API responses."""
        metrics = asdict(self._last_metrics) if self._last_metrics else {}
        return {
            "mode": self._mode,
            "is_live": self.is_live,
            "last_eval_ts": self._last_eval_ts,
            "consec_bad": self._consec_bad,
            "consec_good": self._consec_good,
            "last_metrics": metrics,
            "history": self._history[-10:],  # last 10 for API
        }

    def get_history(self) -> List[dict]:
        return list(self._history)

    def force_live(self) -> None:
        """Admin override: force live mode (e.g. after manual review)."""
        self._mode = "live"
        self._consec_bad = 0
        self._consec_good = int(_cfg("BACKTEST_GATE_CONSEC_RECOVER"))
        self._save()
        logger.info("BacktestGate: forced to LIVE by admin override")

    def force_paper(self) -> None:
        """Admin override: force paper mode."""
        self._mode = "paper"
        self._consec_bad = int(_cfg("BACKTEST_GATE_CONSEC_DEGRADE"))
        self._consec_good = 0
        self._save()
        logger.info("BacktestGate: forced to PAPER by admin override")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_metrics(self, exits: List[dict], win_floor: float) -> PeriodMetrics:
        if not exits:
            return PeriodMetrics(trades=0, win_rate=0.0, avg_pnl_pct=0.0, sharpe=0.0)

        pnls = [float(r.get("pnl_pct", 0)) for r in exits]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0
        avg_pnl = float(np.mean(pnls)) if pnls else 0.0
        sharpe = _sharpe(pnls)

        flags: List[str] = []
        if win_rate < win_floor:
            flags.append("low_winrate")

        return PeriodMetrics(
            trades=len(exits),
            win_rate=round(win_rate, 4),
            avg_pnl_pct=round(avg_pnl, 5),
            sharpe=round(sharpe, 4),
            flags=flags,
        )

    def _make_dummy_record(self, mode: str) -> EvalRecord:
        dummy = PeriodMetrics(trades=0, win_rate=0.0, avg_pnl_pct=0.0, sharpe=0.0)
        return EvalRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            mode=mode,
            current=dummy,
            previous=None,
            sharpe_delta=0.0,
        )

    def _save(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "mode": self._mode,
                "consec_bad": self._consec_bad,
                "consec_good": self._consec_good,
                "last_eval_ts": self._last_eval_ts,
                "history": self._history,
            }
            tmp = self._state_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.debug("BacktestGate: save failed — %s", e)

    def _load(self) -> None:
        try:
            if not self._state_path.exists():
                return
            with open(self._state_path) as f:
                state = json.load(f)
            self._mode = state.get("mode", "unknown")
            self._consec_bad = int(state.get("consec_bad", 0))
            self._consec_good = int(state.get("consec_good", 0))
            self._last_eval_ts = state.get("last_eval_ts", "")
            self._history = state.get("history", [])
        except Exception as e:
            logger.debug("BacktestGate: load failed — %s", e)
