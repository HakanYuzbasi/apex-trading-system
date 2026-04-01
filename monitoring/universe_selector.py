"""
monitoring/universe_selector.py — Dynamic Equity Universe Selector

Scores the static SYMBOLS universe by recent performance so the engine
focuses compute on high-quality opportunities and deprioritises chronic
underperformers.

Scoring (per symbol, last lookback_days):
  - Win rate component   (40% weight) — % of closed trades with pnl > 0
  - Sharpe component     (35% weight) — annualised Sharpe from trade P&Ls
  - Activity component   (25% weight) — penalises symbols with < min_trades

Output:
  - get_priority_scores() → Dict[str, float]   score ∈ [0, 1]
  - get_active_symbols(universe, min_score)    → List[str] above threshold
  - get_skipped_symbols(universe, min_score)   → List[str] below threshold

Integration (execution_loop.py):
    selector.refresh(data_dir)            # every 300 cycles
    score = selector.get_score(symbol)    # in process_symbol (skip if < MIN)
    _dynamic_equity_priority = selector.get_priority_scores()

Safe-by-default:
  - Returns score=0.5 (neutral) for any symbol with < min_trades
  - Never removes symbols with open positions (those are passed through unconditionally)
  - Runs in a background thread (blocking file I/O)
"""
from __future__ import annotations

import glob
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MIN_TRADES = 5       # fewer trades → neutral score
_DEFAULT_LOOKBACK_DAYS = 21   # 3 weeks of recent data
_NEUTRAL_SCORE = 0.50         # score assigned when insufficient data


class SymbolScore:
    __slots__ = ("symbol", "score", "win_rate", "sharpe", "trade_count")

    def __init__(
        self,
        symbol: str,
        score: float,
        win_rate: float,
        sharpe: float,
        trade_count: int,
    ) -> None:
        self.symbol = symbol
        self.score = score
        self.win_rate = win_rate
        self.sharpe = sharpe
        self.trade_count = trade_count

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 4),
            "win_rate": round(self.win_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "trade_count": self.trade_count,
        }


class UniverseSelector:
    """
    Scores a static equity universe by recent trade performance.

    Thread-safe for reading (get_score / get_priority_scores).
    refresh() should be called from asyncio.to_thread (blocking file I/O).
    """

    def __init__(
        self,
        lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
        min_trades: int = _DEFAULT_MIN_TRADES,
        win_rate_weight: float = 0.40,
        sharpe_weight: float = 0.35,
        activity_weight: float = 0.25,
    ) -> None:
        self._lookback_days = lookback_days
        self._min_trades = min_trades
        self._w_wr = win_rate_weight
        self._w_sh = sharpe_weight
        self._w_ac = activity_weight
        self._scores: Dict[str, SymbolScore] = {}
        self._last_refresh: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self, data_dir: Path) -> int:
        """
        Load EXIT records from trade_audit JSONL and recompute scores.
        Returns number of symbols scored.
        Call from asyncio.to_thread.
        """
        try:
            records = _load_exit_records(data_dir, self._lookback_days)
            self._scores = self._compute_scores(records)
            self._last_refresh = datetime.now(timezone.utc)
            logger.info(
                "UniverseSelector refreshed: %d symbols scored from %d trades",
                len(self._scores), sum(s.trade_count for s in self._scores.values()),
            )
            return len(self._scores)
        except Exception as exc:
            logger.warning("UniverseSelector.refresh error: %s", exc)
            return 0

    def get_score(self, symbol: str) -> float:
        """Return composite score [0, 1] for a symbol. 0.5 if not yet scored."""
        entry = self._scores.get(symbol) or self._scores.get(_strip_prefix(symbol))
        return entry.score if entry is not None else _NEUTRAL_SCORE

    def get_priority_scores(self) -> Dict[str, float]:
        """All scored symbols → their composite scores."""
        return {sym: s.score for sym, s in self._scores.items()}

    def get_active_symbols(self, universe: List[str], min_score: float = 0.30) -> List[str]:
        """Symbols in universe with score >= min_score, sorted by score descending."""
        scored = [(s, self.get_score(s)) for s in universe]
        active = [s for s, sc in scored if sc >= min_score]
        active.sort(key=lambda s: self.get_score(s), reverse=True)
        return active

    def get_skipped_symbols(self, universe: List[str], min_score: float = 0.30) -> List[str]:
        return [s for s in universe if self.get_score(s) < min_score]

    def get_report(self) -> dict:
        sorted_scores = sorted(
            (s.to_dict() for s in self._scores.values()),
            key=lambda x: x["score"],
            reverse=True,
        )
        return {
            "scored_count": len(self._scores),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "lookback_days": self._lookback_days,
            "min_trades": self._min_trades,
            "symbols": sorted_scores,
        }

    # ── Core scoring ──────────────────────────────────────────────────────────

    def _compute_scores(
        self, records: Dict[str, List[float]]
    ) -> Dict[str, SymbolScore]:
        """
        records: symbol → list of pnl_pct values (EXIT trades, sorted recent first).
        """
        scores: Dict[str, SymbolScore] = {}
        for symbol, pnls in records.items():
            n = len(pnls)
            if n < self._min_trades:
                score = _NEUTRAL_SCORE
                win_rate = sum(1 for p in pnls if p > 0) / n if n > 0 else 0.5
                sharpe = 0.0
            else:
                win_rate = sum(1 for p in pnls if p > 0) / n
                sharpe = _compute_sharpe(pnls)
                score = self._composite_score(win_rate, sharpe, n)

            scores[symbol] = SymbolScore(
                symbol=symbol,
                score=score,
                win_rate=win_rate,
                sharpe=sharpe,
                trade_count=n,
            )
        return scores

    def _composite_score(self, win_rate: float, sharpe: float, n_trades: int) -> float:
        # Win rate component: linear from 0 (wr=0%) to 1 (wr=70%+)
        wr_component = min(1.0, win_rate / 0.70)

        # Sharpe component: sigmoid-normalise; Sharpe 2.0 → 0.9, Sharpe -2.0 → 0.1
        _sh_arg = max(-50.0, min(50.0, -sharpe * 0.7))
        sh_component = 1.0 / (1.0 + math.exp(_sh_arg))

        # Activity component: more trades → closer to 1.0 (plateaus at 30 trades)
        ac_component = min(1.0, n_trades / 30.0)

        raw = self._w_wr * wr_component + self._w_sh * sh_component + self._w_ac * ac_component
        return float(max(0.0, min(1.0, raw)))


# ── File I/O helpers ──────────────────────────────────────────────────────────

def _load_exit_records(
    data_dir: Path, lookback_days: int
) -> Dict[str, List[float]]:
    """
    Reads EXIT rows from trade_audit JSONL files and groups pnl_pct by symbol.
    """
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp()
    records: Dict[str, List[float]] = {}

    pattern = str(data_dir / "users" / "*" / "audit" / "trade_audit_*.jsonl")
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("action") != "EXIT":
                        continue
                    ts_str = rec.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(
                            ts_str.replace("Z", "+00:00")
                        ).timestamp()
                    except Exception:
                        continue
                    if ts < cutoff_ts:
                        continue
                    raw_sym = str(rec.get("symbol", "") or "")
                    if not raw_sym:
                        continue
                    sym = _strip_prefix(raw_sym)
                    pnl = float(rec.get("pnl_pct", 0.0) or 0.0)
                    records.setdefault(sym, []).append(pnl)
        except Exception as exc:
            logger.debug("UniverseSelector load error on %s: %s", path, exc)

    return records


def _compute_sharpe(pnls: List[float]) -> float:
    """Annualised Sharpe; returns 0.0 for < 5 trades. Clamped to [-10, 10]."""
    if len(pnls) < 5:
        return 0.0
    n = len(pnls)
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / n
    if var <= 0:
        # All identical returns — treat as very positive/negative but capped
        return 5.0 if mean > 0 else (-5.0 if mean < 0 else 0.0)
    std = math.sqrt(var)
    raw = mean / std * math.sqrt(500)  # ~500 trades/yr @ 2/day
    return float(max(-10.0, min(10.0, raw)))


def _strip_prefix(symbol: str) -> str:
    """Remove CRYPTO: or FX: prefix for normalized comparison."""
    for prefix in ("CRYPTO:", "FX:"):
        if symbol.startswith(prefix):
            return symbol[len(prefix):]
    return symbol
