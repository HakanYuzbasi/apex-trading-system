"""
monitoring/execution_quality.py — Execution Quality Tracker

Measures how well the system actually executes orders versus its intentions:
  - Slippage (actual fill vs expected price) in bps
  - Fill rate (what fraction of target qty was filled)
  - TWAP effectiveness (for sliced orders)
  - Per-symbol, per-regime, per-broker slippage percentiles

Data feeds back into position sizing (penalise high-slippage symbols) and
order routing decisions.

Usage (from execution_loop):
    tracker.record_fill(symbol, side, expected_price, fill_price, qty,
                        regime, broker, order_type)
    tracker.get_symbol_slippage_bps(symbol)   # rolling P50/P95 in bps
    tracker.get_sizing_penalty(symbol)         # multiplier in (0.5, 1.0]
    tracker.get_report()                       # full diagnostic dict
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_BPS = 10_000.0   # divisor for bps conversion


@dataclass
class FillRecord:
    symbol: str
    side: str                  # BUY / SELL
    expected_price: float
    fill_price: float
    qty: float
    regime: str
    broker: str                # ibkr / alpaca
    order_type: str            # market / limit / twap
    slippage_bps: float        # signed: + means adverse, - means favourable
    ts: float = field(default_factory=time.time)


class ExecutionQualityTracker:
    """
    Rolling slippage ledger with per-symbol sizing-penalty feedback.

    All fills are stored in a capped deque (max_fills). Summary statistics
    are recomputed lazily when get_report() / get_sizing_penalty() is called.
    Optionally persists a daily JSON summary to data_dir.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        max_fills: int = 2000,
        penalty_p95_bps: float = 30.0,   # above this → apply penalty
        penalty_floor: float = 0.70,      # minimum sizing multiplier
        min_fills_for_penalty: int = 5,   # need ≥ N fills before penalising
        persist_interval_fills: int = 100, # persist every N new fills
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self._max_fills = max_fills
        self._penalty_p95_bps = penalty_p95_bps
        self._penalty_floor = penalty_floor
        self._min_fills = min_fills_for_penalty
        self._persist_interval = persist_interval_fills

        # Rolling store
        self._fills: deque[FillRecord] = deque(maxlen=max_fills)
        self._fills_since_persist: int = 0

        # Per-symbol slippage window (last 50 fills per symbol)
        self._sym_slippage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # Daily P&L impact from slippage (cost accounting)
        self._daily_slippage_cost: Dict[str, float] = defaultdict(float)  # date → USD cost
        self._today_key: str = date.today().isoformat()

    # ------------------------------------------------------------------
    # Record a fill
    # ------------------------------------------------------------------

    def record_fill(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        fill_price: float,
        qty: float,
        regime: str = "unknown",
        broker: str = "unknown",
        order_type: str = "market",
    ) -> FillRecord:
        """Record one fill and compute slippage in bps."""
        if expected_price <= 0 or fill_price <= 0 or qty <= 0:
            logger.debug("ExecutionQuality: skipping invalid fill %s exp=%.4f fill=%.4f", symbol, expected_price, fill_price)
            return None

        # Slippage: positive = paid more / received less (adverse)
        if side.upper() == "BUY":
            raw_slip = (fill_price - expected_price) / expected_price
        else:
            raw_slip = (expected_price - fill_price) / expected_price
        slippage_bps = raw_slip * _BPS

        rec = FillRecord(
            symbol=symbol, side=side,
            expected_price=expected_price, fill_price=fill_price,
            qty=qty, regime=regime, broker=broker, order_type=order_type,
            slippage_bps=slippage_bps,
        )
        self._fills.append(rec)
        self._sym_slippage[symbol].append(slippage_bps)

        # Daily cost tracking
        today = date.today().isoformat()
        if today != self._today_key:
            self._today_key = today
        slippage_usd = abs(raw_slip) * fill_price * qty
        self._daily_slippage_cost[today] = self._daily_slippage_cost.get(today, 0.0) + slippage_usd

        self._fills_since_persist += 1
        if self._data_dir and self._fills_since_persist >= self._persist_interval:
            self._persist()
            self._fills_since_persist = 0

        if abs(slippage_bps) > 50:
            logger.warning(
                "⚠️  High slippage: %s %s fill=%.4f expected=%.4f slip=%.1fbps",
                symbol, side, fill_price, expected_price, slippage_bps,
            )
        return rec

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_symbol_slippage_bps(self, symbol: str) -> dict:
        """Return p25/p50/p75/p95 slippage bps for a symbol (adverse only)."""
        data = list(self._sym_slippage.get(symbol, []))
        if not data:
            return {"p50": 0.0, "p95": 0.0, "count": 0, "mean": 0.0}
        arr = np.array(data, dtype=float)
        return {
            "p25":  float(np.percentile(arr, 25)),
            "p50":  float(np.percentile(arr, 50)),
            "p75":  float(np.percentile(arr, 75)),
            "p95":  float(np.percentile(arr, 95)),
            "mean": float(arr.mean()),
            "count": len(arr),
        }

    def get_sizing_penalty(self, symbol: str) -> float:
        """
        Return a multiplier in [penalty_floor, 1.0] to reduce position size
        for high-slippage symbols. 1.0 means no penalty.
        """
        stats = self.get_symbol_slippage_bps(symbol)
        if stats["count"] < self._min_fills:
            return 1.0
        p95 = stats["p95"]
        if p95 <= self._penalty_p95_bps:
            return 1.0
        # Linear ramp: at 2× threshold → floor
        excess = (p95 - self._penalty_p95_bps) / self._penalty_p95_bps
        mult = 1.0 - (1.0 - self._penalty_floor) * min(1.0, excess)
        return round(max(self._penalty_floor, mult), 3)

    def get_broker_summary(self) -> dict:
        """Average slippage bps grouped by broker."""
        by_broker: Dict[str, list] = defaultdict(list)
        for f in self._fills:
            by_broker[f.broker].append(f.slippage_bps)
        return {
            broker: {
                "mean_bps": float(np.mean(vals)),
                "p95_bps": float(np.percentile(vals, 95)),
                "count": len(vals),
            }
            for broker, vals in by_broker.items()
        }

    def get_regime_summary(self) -> dict:
        """Average slippage bps grouped by regime."""
        by_regime: Dict[str, list] = defaultdict(list)
        for f in self._fills:
            by_regime[f.regime].append(f.slippage_bps)
        return {
            regime: {
                "mean_bps": float(np.mean(vals)),
                "p95_bps": float(np.percentile(vals, 95)),
                "count": len(vals),
            }
            for regime, vals in by_regime.items()
        }

    def get_worst_symbols(self, n: int = 10) -> List[dict]:
        """Top N highest median-slippage symbols."""
        results = []
        for sym, dq in self._sym_slippage.items():
            if len(dq) < self._min_fills:
                continue
            arr = np.array(list(dq), dtype=float)
            results.append({
                "symbol": sym,
                "p50_bps": float(np.percentile(arr, 50)),
                "p95_bps": float(np.percentile(arr, 95)),
                "count": len(arr),
                "penalty": self.get_sizing_penalty(sym),
            })
        results.sort(key=lambda x: x["p95_bps"], reverse=True)
        return results[:n]

    def get_daily_slippage_cost(self) -> dict:
        """Total slippage cost in USD per day."""
        return dict(self._daily_slippage_cost)

    def get_report(self) -> dict:
        """Full diagnostic dict for dashboard / API."""
        all_bps = [f.slippage_bps for f in self._fills]
        global_stats: dict = {}
        if all_bps:
            arr = np.array(all_bps, dtype=float)
            global_stats = {
                "mean_bps": float(arr.mean()),
                "p50_bps":  float(np.percentile(arr, 50)),
                "p95_bps":  float(np.percentile(arr, 95)),
                "total_fills": len(arr),
                "adverse_pct": float((arr > 0).mean() * 100),
            }
        return {
            "global": global_stats,
            "by_broker": self.get_broker_summary(),
            "by_regime": self.get_regime_summary(),
            "worst_symbols": self.get_worst_symbols(10),
            "daily_slippage_cost_usd": self.get_daily_slippage_cost(),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        if not self._data_dir:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            today = date.today().isoformat()
            path = self._data_dir / f"execution_quality_{today.replace('-', '')}.json"
            path.write_text(json.dumps(self.get_report(), indent=2))
        except Exception as exc:
            logger.debug("ExecutionQuality persist error: %s", exc)

    def flush(self) -> None:
        """Force-persist; call at EOD."""
        self._persist()
        self._fills_since_persist = 0
