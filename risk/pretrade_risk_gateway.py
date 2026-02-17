"""
risk/pretrade_risk_gateway.py

Institutional pre-trade hard-limit gateway:
- fat-finger protection (shares/notional)
- price band validation
- ADV participation cap
- projected gross exposure cap

The gateway can emit a tamper-evident decision log for every attempted entry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4


@dataclass
class PreTradeLimitConfig:
    enabled: bool = True
    fail_closed: bool = True
    max_order_notional: float = 250_000.0
    max_order_shares: int = 10_000
    max_price_deviation_bps: float = 250.0
    max_participation_rate: float = 0.10
    max_gross_exposure_ratio: float = 2.0


@dataclass
class PreTradeDecision:
    allowed: bool
    reason_code: str
    message: str
    metadata: Dict[str, float | str | bool] = field(default_factory=dict)


class PreTradeRiskGateway:
    """Hard pre-trade risk checks with hash-chained decision audit logging."""

    def __init__(self, config: PreTradeLimitConfig, audit_dir: Path):
        self.config = config
        self.audit_dir = audit_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._last_hash_cache: Dict[str, Optional[str]] = {}

    def evaluate_entry(
        self,
        *,
        symbol: str,
        asset_class: str,
        side: str,
        quantity: int,
        price: float,
        capital: float,
        current_positions: Dict[str, int | float],
        price_cache: Dict[str, float],
        reference_price: Optional[float] = None,
        adv_shares: Optional[float] = None,
    ) -> PreTradeDecision:
        if not self.config.enabled:
            return PreTradeDecision(True, "disabled", "Pre-trade gateway disabled")

        qty = int(abs(quantity))
        px = float(price)
        if qty <= 0 or px <= 0:
            return PreTradeDecision(False, "invalid_order", "Invalid quantity/price", {"qty": qty, "price": px})

        notional = qty * px
        if qty > self.config.max_order_shares:
            return PreTradeDecision(
                False,
                "max_order_shares",
                f"Order shares {qty} exceed limit {self.config.max_order_shares}",
                {"qty": qty, "limit": self.config.max_order_shares},
            )

        if notional > self.config.max_order_notional:
            return PreTradeDecision(
                False,
                "max_order_notional",
                f"Order notional ${notional:,.0f} exceeds limit ${self.config.max_order_notional:,.0f}",
                {"notional": notional, "limit": self.config.max_order_notional},
            )

        if reference_price and reference_price > 0:
            deviation_bps = abs(px / float(reference_price) - 1.0) * 10_000.0
            if deviation_bps > self.config.max_price_deviation_bps:
                return PreTradeDecision(
                    False,
                    "price_band",
                    f"Price deviation {deviation_bps:.1f}bps exceeds limit {self.config.max_price_deviation_bps:.1f}bps",
                    {"deviation_bps": deviation_bps, "limit_bps": self.config.max_price_deviation_bps},
                )

        if adv_shares and adv_shares > 0:
            participation = qty / float(adv_shares)
            if participation > self.config.max_participation_rate:
                return PreTradeDecision(
                    False,
                    "adv_participation",
                    f"Participation {participation:.1%} exceeds limit {self.config.max_participation_rate:.1%}",
                    {"participation": participation, "limit": self.config.max_participation_rate},
                )

        gross_exposure = self._gross_exposure(current_positions, price_cache)
        projected_gross = gross_exposure + notional
        if capital > 0:
            projected_ratio = projected_gross / capital
            if projected_ratio > self.config.max_gross_exposure_ratio:
                return PreTradeDecision(
                    False,
                    "gross_exposure",
                    f"Projected gross exposure {projected_ratio:.2f}x exceeds limit {self.config.max_gross_exposure_ratio:.2f}x",
                    {
                        "projected_ratio": projected_ratio,
                        "limit_ratio": self.config.max_gross_exposure_ratio,
                        "projected_gross": projected_gross,
                    },
                )

        return PreTradeDecision(
            True,
            "allowed",
            "Pre-trade hard-limit checks passed",
            {
                "notional": notional,
                "projected_gross": projected_gross,
                "capital": float(capital),
                "side": side,
                "asset_class": asset_class,
            },
        )

    def record_decision(
        self,
        *,
        symbol: str,
        asset_class: str,
        side: str,
        quantity: int,
        price: float,
        decision: PreTradeDecision,
        actor: str = "strategy_loop",
    ) -> Dict[str, object]:
        file_path = self._audit_file_path()
        prev_hash = self._last_hash(file_path)
        payload: Dict[str, object] = {
            "event_id": f"ptg-{uuid4().hex[:12]}",
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "asset_class": asset_class,
            "side": side,
            "quantity": int(quantity),
            "price": float(price),
            "allowed": bool(decision.allowed),
            "reason_code": decision.reason_code,
            "message": decision.message,
            "metadata": dict(decision.metadata),
            "actor": actor,
            "prev_hash": prev_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        self._last_hash_cache[str(file_path)] = str(payload["hash"])
        return payload

    @staticmethod
    def _gross_exposure(positions: Dict[str, int | float], price_cache: Dict[str, float]) -> float:
        exposure = 0.0
        for symbol, qty in positions.items():
            try:
                q = float(qty)
                p = float(price_cache.get(symbol, 0.0))
            except (TypeError, ValueError):
                continue
            if q == 0 or p <= 0:
                continue
            exposure += abs(q) * p
        return exposure

    def _audit_file_path(self) -> Path:
        date_str = datetime.utcnow().strftime("%Y%m%d")
        return self.audit_dir / f"pretrade_gateway_{date_str}.jsonl"

    def _last_hash(self, filepath: Path) -> Optional[str]:
        cache_key = str(filepath)
        if cache_key in self._last_hash_cache:
            return self._last_hash_cache[cache_key]

        if not filepath.exists():
            self._last_hash_cache[cache_key] = None
            return None

        last_hash: Optional[str] = None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        value = parsed.get("hash")
                        if value:
                            last_hash = str(value)
        except Exception:
            last_hash = None
        self._last_hash_cache[cache_key] = last_hash
        return last_hash
