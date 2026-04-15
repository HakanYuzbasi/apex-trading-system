"""
monitoring/missed_opportunity_tracker.py - Missed High-Earner Opportunity Tracker

Tracks signals that were generated but filtered out by thresholds, risk limits,
social governor, or other gates. Measures the hypothetical P&L of missed trades
to identify systematic alpha leakage and overly conservative filters.

Used to answer: "What high earners is our strategy missing?"
"""

import json
import logging
import os
import pandas as pd
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MissedOpportunity:
    """A signal that was generated but filtered out before execution."""
    symbol: str
    signal_date: str
    signal_strength: float
    confidence: float
    direction: str  # "long" or "short"
    regime: str
    filter_reason: str  # Why it was blocked: "threshold", "confidence", "risk_limit", "social_governor", etc.
    entry_price: float
    # Filled in retrospectively
    price_after_5d: Optional[float] = None
    price_after_10d: Optional[float] = None
    price_after_20d: Optional[float] = None
    missed_pnl_5d_pct: Optional[float] = None
    missed_pnl_10d_pct: Optional[float] = None
    missed_pnl_20d_pct: Optional[float] = None
    session_type: str = "unified"
    sector: str = "Unknown"
    asset_class: str = "equity"


@dataclass
class MissedEarnerReport:
    """Aggregated report of missed opportunities."""
    total_missed: int = 0
    total_missed_pnl_5d: float = 0.0
    total_missed_pnl_10d: float = 0.0
    by_filter_reason: Dict[str, int] = field(default_factory=dict)
    by_sector: Dict[str, float] = field(default_factory=dict)
    by_regime: Dict[str, float] = field(default_factory=dict)
    by_asset_class: Dict[str, float] = field(default_factory=dict)
    top_missed_symbols: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""


class MissedOpportunityTracker:
    """Tracks and analyzes missed trading opportunities to identify alpha leakage."""

    MAX_PENDING = 500  # Max unfilled opportunities awaiting retrospective pricing

    def __init__(self, data_dir: Path, session_type: str = "unified"):
        self.data_dir = Path(data_dir)
        self.session_type = session_type
        self._pending: List[MissedOpportunity] = []
        self._completed: List[MissedOpportunity] = []
        self._output_file = self.data_dir / f"missed_opportunities_{session_type}.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def record_missed(
        self,
        symbol: str,
        signal_strength: float,
        confidence: float,
        direction: str,
        regime: str,
        filter_reason: str,
        entry_price: float,
        sector: str = "Unknown",
        asset_class: str = "equity",
    ):
        """Record a signal that was filtered out and did not result in a trade."""
        opp = MissedOpportunity(
            symbol=symbol,
            signal_date=datetime.utcnow().isoformat(),
            signal_strength=signal_strength,
            confidence=confidence,
            direction=direction,
            regime=regime,
            filter_reason=filter_reason,
            entry_price=entry_price,
            session_type=self.session_type,
            sector=sector,
            asset_class=asset_class,
        )
        self._pending.append(opp)
        if len(self._pending) > self.MAX_PENDING:
            self._pending = self._pending[-self.MAX_PENDING:]

    def update_retrospective_prices(self, current_prices: Dict[str, float]):
        """Update missed opportunities with current prices to calculate hypothetical PnL.

        Called periodically (e.g., daily) with a map of symbol -> current_price.
        """
        from datetime import datetime as dt

        still_pending = []
        for opp in self._pending:
            if opp.symbol not in current_prices:
                still_pending.append(opp)
                continue

            try:
                signal_date = dt.fromisoformat(opp.signal_date)
                days_elapsed = (dt.utcnow() - signal_date).days
            except (ValueError, TypeError):
                still_pending.append(opp)
                continue

            current = current_prices[opp.symbol]
            if opp.entry_price <= 0:
                still_pending.append(opp)
                continue

            if opp.direction == "long":
                pnl_pct = (current / opp.entry_price - 1)
            else:
                pnl_pct = (opp.entry_price / current - 1)

            if days_elapsed >= 5 and opp.price_after_5d is None:
                opp.price_after_5d = current
                opp.missed_pnl_5d_pct = pnl_pct

            if days_elapsed >= 10 and opp.price_after_10d is None:
                opp.price_after_10d = current
                opp.missed_pnl_10d_pct = pnl_pct

            if days_elapsed >= 20 and opp.price_after_20d is None:
                opp.price_after_20d = current
                opp.missed_pnl_20d_pct = pnl_pct
                # Fully resolved — move to completed
                self._completed.append(opp)
                continue

            still_pending.append(opp)

        self._pending = still_pending
        self._save()

    def generate_report(self) -> MissedEarnerReport:
        """Generate a summary report of missed high-earner opportunities."""
        report = MissedEarnerReport(
            total_missed=len(self._completed),
            generated_at=datetime.utcnow().isoformat(),
        )

        by_filter: Dict[str, int] = defaultdict(int)
        by_sector: Dict[str, float] = defaultdict(float)
        by_regime: Dict[str, float] = defaultdict(float)
        by_asset: Dict[str, float] = defaultdict(float)
        symbol_pnl: Dict[str, float] = defaultdict(float)

        for opp in self._completed:
            pnl = opp.missed_pnl_10d_pct or opp.missed_pnl_5d_pct or 0.0
            by_filter[opp.filter_reason] += 1
            by_sector[opp.sector] += pnl
            by_regime[opp.regime] += pnl
            by_asset[opp.asset_class] += pnl
            symbol_pnl[opp.symbol] += pnl
            report.total_missed_pnl_5d += (opp.missed_pnl_5d_pct or 0.0)
            report.total_missed_pnl_10d += (opp.missed_pnl_10d_pct or 0.0)

        report.by_filter_reason = dict(by_filter)
        report.by_sector = dict(by_sector)
        report.by_regime = dict(by_regime)
        report.by_asset_class = dict(by_asset)

        # Top missed symbols sorted by missed PnL (descending)
        sorted_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
        report.top_missed_symbols = [
            {"symbol": s, "total_missed_pnl_pct": round(p, 4)}
            for s, p in sorted_symbols[:20]
        ]

        return report

    def _save(self):
        """Persist to disk."""
        try:
            data = {
                "pending": [asdict(o) for o in self._pending[-self.MAX_PENDING:]],
                "completed": [asdict(o) for o in self._completed[-1000:]],
            }
            self._output_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save missed opportunities: {e}")

    def _load(self):
        """Load from disk."""
        if not self._output_file.exists():
            return
        try:
            data = json.loads(self._output_file.read_text())
            self._pending = [MissedOpportunity(**o) for o in data.get("pending", [])]
            self._completed = [MissedOpportunity(**o) for o in data.get("completed", [])]
        except Exception as e:
            logger.warning(f"Failed to load missed opportunities: {e}")


class SectorMomentumScanner:
    """Weekly scanner that identifies sectors/assets with strong momentum
    but zero portfolio exposure — the 'missed high earners'.
    """

    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.lookback_periods = lookback_periods or {
            "1m": 21,
            "3m": 63,
            "6m": 126,
        }

    def scan(
        self,
        price_data: Dict[str, "pd.Series"],
        current_positions: List[str],
        sector_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Scan for high-momentum sectors/assets with no current exposure.

        Returns a list of alerts sorted by combined momentum score.
        """
        import numpy as np

        results = []
        position_sectors = {sector_map.get(s, "Unknown") for s in current_positions}

        for symbol, prices in price_data.items():
            if len(prices) < max(self.lookback_periods.values()):
                continue

            scores = {}
            for label, lookback in self.lookback_periods.items():
                ret = float(prices.iloc[-1] / prices.iloc[-lookback] - 1) if prices.iloc[-lookback] > 0 else 0.0
                scores[f"momentum_{label}"] = ret

            combined = sum(scores.values()) / len(scores) if scores else 0.0
            sector = sector_map.get(symbol, "Unknown")
            has_exposure = symbol in current_positions or sector in position_sectors

            if combined > 0.02 and not has_exposure:  # >2% average momentum, no exposure
                results.append({
                    "symbol": symbol,
                    "sector": sector,
                    "combined_momentum": round(combined, 4),
                    "has_position": False,
                    "sector_has_position": sector in position_sectors,
                    **{k: round(v, 4) for k, v in scores.items()},
                })

        results.sort(key=lambda x: x["combined_momentum"], reverse=True)
        return results[:20]  # Top 20 missed opportunities
