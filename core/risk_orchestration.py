"""Risk orchestration helpers extracted from the trading runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RiskDecision:
    """Normalized risk decision payload."""

    allowed: bool
    reason: str
    metadata: Dict[str, Any]


class RiskOrchestration:
    """Coordinate layered risk evaluators in a typed, reusable interface."""

    def __init__(self, risk_manager: Any, pretrade_gateway: Optional[Any] = None) -> None:
        self._risk_manager = risk_manager
        self._pretrade_gateway = pretrade_gateway

    def evaluate_entry(self, symbol: str, quantity: float, price: float, context: Dict[str, Any]) -> RiskDecision:
        """Evaluate whether a new entry should be allowed."""
        if self._pretrade_gateway is not None:
            gateway_result = self._pretrade_gateway.evaluate_order(
                symbol=symbol,
                quantity=quantity,
                price=price,
                context=context,
            )
            if not gateway_result.allowed:
                return RiskDecision(
                    allowed=False,
                    reason="pretrade_gateway_block",
                    metadata={"detail": gateway_result.reason},
                )

        if hasattr(self._risk_manager, "validate_trade"):
            ok, reason = self._risk_manager.validate_trade(symbol, quantity, price)
            if not ok:
                return RiskDecision(
                    allowed=False,
                    reason="risk_manager_block",
                    metadata={"detail": reason},
                )

        return RiskDecision(allowed=True, reason="allowed", metadata={})

    def evaluate_exit(self, symbol: str, context: Dict[str, Any]) -> RiskDecision:
        """Evaluate whether an exit should be enforced by risk controls."""
        if hasattr(self._risk_manager, "should_force_exit"):
            force_exit, reason = self._risk_manager.should_force_exit(symbol, context)
            if force_exit:
                return RiskDecision(
                    allowed=True,
                    reason="forced_exit",
                    metadata={"detail": reason},
                )
        return RiskDecision(allowed=True, reason="normal_exit", metadata={})
