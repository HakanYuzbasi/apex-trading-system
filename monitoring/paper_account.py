"""
monitoring/paper_account.py — Shadow Paper Account (Implementation Shortfall Tracker)

Runs a zero-risk parallel paper account that receives the same entry/exit signals
as the live account, executes at theoretical mid prices, and tracks P&L delta vs live.

The gap between paper P&L and live P&L = implementation shortfall:
  - Slippage (market impact)
  - Commissions
  - Fill-rate failures
  - Timing lag

Usage:
    from monitoring.paper_account import PaperAccount
    pa = PaperAccount(state_dir=Path("data/paper_account"))

    # At entry
    pa.record_entry("AAPL", side="BUY", price=175.50, notional=5000.0)

    # At exit (both paper and live)
    paper_pnl = pa.record_exit("AAPL", price=177.00)

    # Report live trade P&L for comparison
    pa.record_live_result("AAPL", live_pnl_usd=60.0, paper_pnl_usd=paper_pnl)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAX_CLOSED_HISTORY = 200


@dataclass
class PaperPosition:
    symbol: str
    side: str           # "BUY" | "SELL"
    entry_price: float
    notional: float     # USD notional at entry
    entry_ts: float = field(default_factory=time.time)


@dataclass
class PaperTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    notional: float
    pnl_usd: float
    live_pnl_usd: float         # 0.0 when not yet matched
    shortfall_usd: float        # paper_pnl - live_pnl (positive = paper did better)
    entry_ts: float
    exit_ts: float

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "notional": round(self.notional, 2),
            "pnl_usd": round(self.pnl_usd, 2),
            "live_pnl_usd": round(self.live_pnl_usd, 2),
            "shortfall_usd": round(self.shortfall_usd, 2),
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
        }


class PaperAccount:
    """
    Parallel paper account that mirrors live trades at theoretical mid prices.

    Key metric: implementation_shortfall_usd = paper_total_pnl - live_total_pnl
    Positive shortfall means live execution is lagging theory (expected in real markets).
    """

    def __init__(self, state_dir: Optional[Path] = None):
        self._positions: Dict[str, PaperPosition] = {}
        self._closed: List[PaperTrade] = []        # capped at _MAX_CLOSED_HISTORY
        self._paper_total_pnl: float = 0.0
        self._live_total_pnl: float = 0.0
        self._day_start_ts: float = time.time()

        if state_dir:
            self._state_dir = Path(state_dir)
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load()
        else:
            self._state_dir = None

    # ── Public write API ──────────────────────────────────────────────

    def record_entry(
        self,
        symbol: str,
        side: str,
        price: float,
        notional: float,
    ) -> None:
        """Record a paper entry at theoretical mid price."""
        if price <= 0 or notional <= 0:
            return
        self._positions[symbol] = PaperPosition(
            symbol=symbol,
            side=side.upper(),
            entry_price=float(price),
            notional=float(notional),
        )
        logger.debug("PaperAccount: entered %s %s @ %.4f notional=%.0f", side, symbol, price, notional)

    def record_exit(self, symbol: str, price: float) -> float:
        """
        Record a paper exit at theoretical mid price.

        Returns paper P&L for this trade in USD (0.0 if no open position).
        """
        pos = self._positions.pop(symbol, None)
        if pos is None or price <= 0:
            return 0.0

        shares = pos.notional / pos.entry_price
        if pos.side == "BUY":
            pnl = (price - pos.entry_price) * shares
        else:  # SHORT
            pnl = (pos.entry_price - price) * shares

        trade = PaperTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=float(price),
            notional=pos.notional,
            pnl_usd=float(pnl),
            live_pnl_usd=0.0,
            shortfall_usd=0.0,   # filled in by record_live_result
            entry_ts=pos.entry_ts,
            exit_ts=time.time(),
        )
        self._closed.append(trade)
        if len(self._closed) > _MAX_CLOSED_HISTORY:
            self._closed = self._closed[-_MAX_CLOSED_HISTORY:]
        self._paper_total_pnl += pnl
        logger.debug(
            "PaperAccount: exited %s @ %.4f → paper_pnl=%.2f", symbol, price, pnl
        )
        self._save()
        return float(pnl)

    def record_live_result(
        self, symbol: str, live_pnl_usd: float, paper_pnl_usd: Optional[float] = None
    ) -> float:
        """
        Match a live trade result to the most recent closed paper trade for symbol.

        Returns shortfall (paper - live).  Positive = theory beat reality.
        """
        # Find most recent closed trade for symbol
        for trade in reversed(self._closed):
            if trade.symbol == symbol and trade.live_pnl_usd == 0.0:
                trade.live_pnl_usd = float(live_pnl_usd)
                if paper_pnl_usd is not None:
                    trade.pnl_usd = float(paper_pnl_usd)
                trade.shortfall_usd = trade.pnl_usd - trade.live_pnl_usd
                self._live_total_pnl += float(live_pnl_usd)
                self._save()
                return float(trade.shortfall_usd)
        # No matching paper trade: still track live P&L
        self._live_total_pnl += float(live_pnl_usd)
        return 0.0

    def reset_day(self) -> None:
        """Reset daily P&L counters (call at market open)."""
        self._paper_total_pnl = 0.0
        self._live_total_pnl = 0.0
        self._day_start_ts = time.time()
        logger.info("PaperAccount: daily reset")
        self._save()

    # ── Public query API ─────────────────────────────────────────────

    @property
    def implementation_shortfall_usd(self) -> float:
        """Today's paper P&L minus live P&L (positive = live lagging theory)."""
        return self._paper_total_pnl - self._live_total_pnl

    @property
    def shortfall_pct_of_paper(self) -> float:
        """Shortfall as % of absolute paper P&L."""
        if abs(self._paper_total_pnl) < 1e-6:
            return 0.0
        return self.implementation_shortfall_usd / abs(self._paper_total_pnl)

    def get_win_rates(self) -> Dict[str, float]:
        matched = [t for t in self._closed if t.live_pnl_usd != 0.0]
        if not matched:
            return {"paper": 0.0, "live": 0.0, "n": 0}
        paper_wins = sum(1 for t in matched if t.pnl_usd > 0)
        live_wins  = sum(1 for t in matched if t.live_pnl_usd > 0)
        n = len(matched)
        return {
            "paper": round(paper_wins / n, 4),
            "live": round(live_wins / n, 4),
            "n": n,
        }

    def get_avg_shortfall_per_trade(self) -> float:
        matched = [t for t in self._closed if t.live_pnl_usd != 0.0]
        if not matched:
            return 0.0
        return float(np.mean([t.shortfall_usd for t in matched]))

    def get_snapshot(self) -> Dict:
        wr = self.get_win_rates()
        recent = [t.to_dict() for t in reversed(self._closed[-20:])]
        return {
            "available": True,
            "open_positions": len(self._positions),
            "closed_trades": len(self._closed),
            "paper_total_pnl": round(self._paper_total_pnl, 2),
            "live_total_pnl": round(self._live_total_pnl, 2),
            "implementation_shortfall_usd": round(self.implementation_shortfall_usd, 2),
            "shortfall_pct": round(self.shortfall_pct_of_paper * 100, 2),
            "avg_shortfall_per_trade": round(self.get_avg_shortfall_per_trade(), 2),
            "win_rates": wr,
            "day_start_ts": self._day_start_ts,
            "recent_trades": recent,
        }

    # ── Persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        if self._state_dir is None:
            return
        try:
            state = {
                "paper_total_pnl": self._paper_total_pnl,
                "live_total_pnl": self._live_total_pnl,
                "day_start_ts": self._day_start_ts,
                "closed": [t.to_dict() for t in self._closed[-_MAX_CLOSED_HISTORY:]],
            }
            p = self._state_dir / "paper_account_state.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(p)
        except Exception as exc:
            logger.debug("PaperAccount: save failed: %s", exc)

    def _load(self) -> None:
        if self._state_dir is None:
            return
        try:
            p = self._state_dir / "paper_account_state.json"
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            self._paper_total_pnl = float(raw.get("paper_total_pnl", 0.0))
            self._live_total_pnl = float(raw.get("live_total_pnl", 0.0))
            self._day_start_ts = float(raw.get("day_start_ts", time.time()))
            for td in raw.get("closed", []):
                self._closed.append(PaperTrade(
                    symbol=td.get("symbol", ""),
                    side=td.get("side", "BUY"),
                    entry_price=float(td.get("entry_price", 0)),
                    exit_price=float(td.get("exit_price", 0)),
                    notional=float(td.get("notional", 0)),
                    pnl_usd=float(td.get("pnl_usd", 0)),
                    live_pnl_usd=float(td.get("live_pnl_usd", 0)),
                    shortfall_usd=float(td.get("shortfall_usd", 0)),
                    entry_ts=float(td.get("entry_ts", 0)),
                    exit_ts=float(td.get("exit_ts", 0)),
                ))
        except Exception as exc:
            logger.debug("PaperAccount: load failed: %s", exc)
