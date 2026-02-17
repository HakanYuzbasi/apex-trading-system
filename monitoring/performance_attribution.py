"""
monitoring/performance_attribution.py

Sleeve-level performance attribution for live/paper trading diagnostics.

Tracks per-position entry context and computes closed-trade attribution:
- gross alpha (price move P&L)
- commission drag
- modeled slippage drag
- net alpha
- holding period
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class EntryAttribution:
    symbol: str
    asset_class: str
    sleeve: str
    side: str
    quantity: float
    entry_price: float
    entry_time: str
    entry_signal: float
    entry_confidence: float
    governor_tier: str
    governor_regime: str
    risk_multiplier: float
    vix_multiplier: float
    governor_size_multiplier: float
    entry_slippage_bps: float
    source: str = "trade_entry"


@dataclass
class ClosedTradeAttribution:
    symbol: str
    asset_class: str
    sleeve: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    holding_hours: float
    entry_signal: float
    entry_confidence: float
    governor_tier: str
    governor_regime: str
    exit_reason: str
    gross_pnl: float
    net_pnl: float
    commissions: float
    entry_slippage_bps: float
    exit_slippage_bps: float
    modeled_entry_slippage_cost: float
    modeled_exit_slippage_cost: float
    modeled_execution_drag: float
    pnl_bps_on_entry_notional: float
    source: str = "trade_exit"


class PerformanceAttributionTracker:
    """Persistent attribution tracker for sleeve-level diagnostics."""

    def __init__(self, data_dir: Path, max_closed_trades: int = 5000):
        self.data_dir = Path(data_dir)
        self.state_file = self.data_dir / "performance_attribution.json"
        self.max_closed_trades = max_closed_trades
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.social_impacts: List[Dict[str, Any]] = []
        self._load_state()

    def record_entry(
        self,
        *,
        symbol: str,
        asset_class: str,
        sleeve: str,
        side: str,
        quantity: float,
        entry_price: float,
        entry_signal: float,
        entry_confidence: float,
        governor_tier: str,
        governor_regime: str,
        risk_multiplier: float = 1.0,
        vix_multiplier: float = 1.0,
        governor_size_multiplier: float = 1.0,
        entry_slippage_bps: float = 0.0,
        entry_time: Optional[datetime] = None,
        source: str = "trade_entry",
    ) -> None:
        """Record/refresh open position attribution context."""
        if quantity <= 0 or entry_price <= 0:
            return

        entry = EntryAttribution(
            symbol=symbol,
            asset_class=str(asset_class).upper(),
            sleeve=sleeve,
            side=side.upper(),
            quantity=float(quantity),
            entry_price=float(entry_price),
            entry_time=(entry_time or datetime.now()).isoformat(),
            entry_signal=float(entry_signal),
            entry_confidence=float(entry_confidence),
            governor_tier=str(governor_tier),
            governor_regime=str(governor_regime),
            risk_multiplier=float(risk_multiplier),
            vix_multiplier=float(vix_multiplier),
            governor_size_multiplier=float(governor_size_multiplier),
            entry_slippage_bps=float(entry_slippage_bps or 0.0),
            source=source,
        )
        self.open_positions[symbol] = asdict(entry)
        self._save_state()

    def record_exit(
        self,
        *,
        symbol: str,
        quantity: float,
        exit_price: float,
        gross_pnl: float,
        net_pnl: float,
        commissions: float,
        exit_reason: str,
        exit_slippage_bps: float = 0.0,
        asset_class_fallback: str = "EQUITY",
        sleeve_fallback: str = "unknown",
        side_fallback: str = "LONG",
        entry_price_fallback: float = 0.0,
        entry_signal_fallback: float = 0.0,
        entry_confidence_fallback: float = 0.0,
        governor_tier_fallback: str = "unknown",
        governor_regime_fallback: str = "unknown",
        entry_time_fallback: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        source: str = "trade_exit",
    ) -> None:
        """Record closed-trade attribution using entry context if available."""
        if quantity <= 0 or exit_price <= 0:
            return

        entry_ctx = self.open_positions.pop(symbol, None) or {}

        entry_price = float(entry_ctx.get("entry_price", entry_price_fallback) or 0.0)
        if entry_price <= 0:
            entry_price = float(exit_price)

        entry_time_raw = entry_ctx.get("entry_time")
        if entry_time_raw:
            try:
                entry_dt = datetime.fromisoformat(str(entry_time_raw))
            except Exception:
                entry_dt = entry_time_fallback or datetime.now()
        else:
            entry_dt = entry_time_fallback or datetime.now()

        exit_dt = exit_time or datetime.now()
        holding_hours = max(0.0, (exit_dt - entry_dt).total_seconds() / 3600.0)

        entry_notional = abs(float(quantity) * float(entry_price))
        exit_notional = abs(float(quantity) * float(exit_price))
        entry_slippage_bps = float(entry_ctx.get("entry_slippage_bps", 0.0) or 0.0)
        exit_slippage_bps = float(exit_slippage_bps or 0.0)

        modeled_entry_slippage_cost = entry_notional * abs(entry_slippage_bps) / 10000.0
        modeled_exit_slippage_cost = exit_notional * abs(exit_slippage_bps) / 10000.0
        modeled_execution_drag = (
            float(commissions) + modeled_entry_slippage_cost + modeled_exit_slippage_cost
        )
        pnl_bps = (
            (float(net_pnl) / entry_notional) * 10000.0
            if entry_notional > 0
            else 0.0
        )

        closed = ClosedTradeAttribution(
            symbol=symbol,
            asset_class=str(entry_ctx.get("asset_class", asset_class_fallback)).upper(),
            sleeve=str(entry_ctx.get("sleeve", sleeve_fallback)),
            side=str(entry_ctx.get("side", side_fallback)).upper(),
            quantity=float(quantity),
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            entry_time=entry_dt.isoformat(),
            exit_time=exit_dt.isoformat(),
            holding_hours=float(holding_hours),
            entry_signal=float(entry_ctx.get("entry_signal", entry_signal_fallback) or 0.0),
            entry_confidence=float(
                entry_ctx.get("entry_confidence", entry_confidence_fallback) or 0.0
            ),
            governor_tier=str(entry_ctx.get("governor_tier", governor_tier_fallback)),
            governor_regime=str(entry_ctx.get("governor_regime", governor_regime_fallback)),
            exit_reason=str(exit_reason or "unknown"),
            gross_pnl=float(gross_pnl),
            net_pnl=float(net_pnl),
            commissions=float(commissions),
            entry_slippage_bps=entry_slippage_bps,
            exit_slippage_bps=exit_slippage_bps,
            modeled_entry_slippage_cost=float(modeled_entry_slippage_cost),
            modeled_exit_slippage_cost=float(modeled_exit_slippage_cost),
            modeled_execution_drag=float(modeled_execution_drag),
            pnl_bps_on_entry_notional=float(pnl_bps),
            source=source,
        )
        self.closed_trades.append(asdict(closed))
        if len(self.closed_trades) > self.max_closed_trades:
            self.closed_trades = self.closed_trades[-self.max_closed_trades :]
        self._save_state()

    def get_summary(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Compute attribution summary for diagnostics/dashboard."""
        cutoff = datetime.now() - timedelta(days=max(1, lookback_days))
        filtered: List[Dict[str, Any]] = []
        for row in self.closed_trades:
            try:
                exit_dt = datetime.fromisoformat(str(row.get("exit_time")))
            except Exception:
                continue
            if exit_dt >= cutoff:
                filtered.append(row)

        social_filtered: List[Dict[str, Any]] = []
        for row in self.social_impacts:
            try:
                ts = datetime.fromisoformat(str(row.get("timestamp"))) if row.get("timestamp") else None
            except Exception:
                ts = None
            if ts and ts >= cutoff:
                social_filtered.append(row)

        summary: Dict[str, Any] = {
            "lookback_days": int(lookback_days),
            "closed_trades": len(filtered),
            "open_positions_tracked": len(self.open_positions),
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "commissions": 0.0,
            "modeled_execution_drag": 0.0,
            "modeled_slippage_drag": 0.0,
            "by_sleeve": {},
            "by_asset_class": {},
            "social_governor": {
                "events": 0,
                "blocked_alpha_opportunity": 0.0,
                "avoided_drawdown_estimate": 0.0,
                "hedge_cost_drag": 0.0,
                "by_asset_class_regime": {},
            },
        }
        if not filtered and not social_filtered:
            return summary

        def _update_bucket(buckets: Dict[str, Dict[str, Any]], key: str, row: Dict[str, Any]) -> None:
            bucket = buckets.setdefault(
                key,
                {
                    "trades": 0,
                    "gross_pnl": 0.0,
                    "net_pnl": 0.0,
                    "commissions": 0.0,
                    "modeled_execution_drag": 0.0,
                    "modeled_slippage_drag": 0.0,
                    "avg_holding_hours": 0.0,
                },
            )
            bucket["trades"] += 1
            bucket["gross_pnl"] += float(row.get("gross_pnl", 0.0) or 0.0)
            bucket["net_pnl"] += float(row.get("net_pnl", 0.0) or 0.0)
            bucket["commissions"] += float(row.get("commissions", 0.0) or 0.0)
            bucket["modeled_execution_drag"] += float(
                row.get("modeled_execution_drag", 0.0) or 0.0
            )
            bucket["modeled_slippage_drag"] += float(
                row.get("modeled_entry_slippage_cost", 0.0) or 0.0
            ) + float(row.get("modeled_exit_slippage_cost", 0.0) or 0.0)
            bucket["avg_holding_hours"] += float(row.get("holding_hours", 0.0) or 0.0)

        for row in filtered:
            summary["gross_pnl"] += float(row.get("gross_pnl", 0.0) or 0.0)
            summary["net_pnl"] += float(row.get("net_pnl", 0.0) or 0.0)
            summary["commissions"] += float(row.get("commissions", 0.0) or 0.0)
            summary["modeled_execution_drag"] += float(
                row.get("modeled_execution_drag", 0.0) or 0.0
            )
            summary["modeled_slippage_drag"] += float(
                row.get("modeled_entry_slippage_cost", 0.0) or 0.0
            ) + float(row.get("modeled_exit_slippage_cost", 0.0) or 0.0)
            _update_bucket(summary["by_sleeve"], str(row.get("sleeve", "unknown")), row)
            _update_bucket(
                summary["by_asset_class"],
                str(row.get("asset_class", "UNKNOWN")).upper(),
                row,
            )

        for buckets in (summary["by_sleeve"], summary["by_asset_class"]):
            for key, bucket in buckets.items():
                trades = max(1, int(bucket["trades"]))
                bucket["avg_holding_hours"] = float(bucket["avg_holding_hours"] / trades)
                bucket["avg_net_pnl"] = float(bucket["net_pnl"] / trades)
                bucket["execution_drag_pct_of_gross"] = (
                    float(bucket["modeled_execution_drag"] / abs(bucket["gross_pnl"]))
                    if abs(bucket["gross_pnl"]) > 1e-9
                    else 0.0
                )

        social_bucket = summary["social_governor"]
        for row in social_filtered:
            asset = str(row.get("asset_class", "UNKNOWN")).upper()
            regime = str(row.get("regime", "default")).lower()
            key = f"{asset}:{regime}"
            bucket = social_bucket["by_asset_class_regime"].setdefault(
                key,
                {
                    "asset_class": asset,
                    "regime": regime,
                    "events": 0,
                    "blocked_alpha_opportunity": 0.0,
                    "avoided_drawdown_estimate": 0.0,
                    "hedge_cost_drag": 0.0,
                },
            )
            blocked = float(row.get("blocked_alpha_opportunity", 0.0) or 0.0)
            avoided = float(row.get("avoided_drawdown_estimate", 0.0) or 0.0)
            hedge = float(row.get("hedge_cost_drag", 0.0) or 0.0)

            bucket["events"] += 1
            bucket["blocked_alpha_opportunity"] += blocked
            bucket["avoided_drawdown_estimate"] += avoided
            bucket["hedge_cost_drag"] += hedge

            social_bucket["events"] += 1
            social_bucket["blocked_alpha_opportunity"] += blocked
            social_bucket["avoided_drawdown_estimate"] += avoided
            social_bucket["hedge_cost_drag"] += hedge

        return summary

    def record_social_governor_impact(
        self,
        *,
        asset_class: str,
        regime: str,
        blocked_alpha_opportunity: float,
        avoided_drawdown_estimate: float,
        hedge_cost_drag: float,
        policy_version: str = "",
        reason: str = "",
        event_id: str = "",
        source: str = "social_governor",
        timestamp: Optional[datetime] = None,
    ) -> None:
        row = {
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "asset_class": str(asset_class).upper(),
            "regime": str(regime).lower() or "default",
            "blocked_alpha_opportunity": float(blocked_alpha_opportunity or 0.0),
            "avoided_drawdown_estimate": float(avoided_drawdown_estimate or 0.0),
            "hedge_cost_drag": float(hedge_cost_drag or 0.0),
            "policy_version": str(policy_version or ""),
            "reason": str(reason or ""),
            "event_id": str(event_id or ""),
            "source": str(source),
        }
        self.social_impacts.append(row)
        if len(self.social_impacts) > self.max_closed_trades:
            self.social_impacts = self.social_impacts[-self.max_closed_trades :]
        self._save_state()

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.open_positions = state.get("open_positions", {}) or {}
            self.closed_trades = state.get("closed_trades", []) or []
            self.social_impacts = state.get("social_impacts", []) or []
            logger.info(
                "📈 Restored attribution state: open=%d closed=%d social=%d",
                len(self.open_positions),
                len(self.closed_trades),
                len(self.social_impacts),
            )
        except Exception as exc:
            logger.error("Failed to load performance attribution state: %s", exc)

    def _save_state(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": datetime.now().isoformat(),
                "open_positions": self.open_positions,
                "closed_trades": self.closed_trades[-self.max_closed_trades :],
                "social_impacts": self.social_impacts[-self.max_closed_trades :],
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save performance attribution state: %s", exc)
