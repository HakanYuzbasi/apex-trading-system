"""
Compliance Copilot Service - wraps ComplianceManager for API use.
"""

import logging
from typing import Any, Dict, Optional

from monitoring.compliance_manager import ComplianceManager

logger = logging.getLogger(__name__)

# Module-level singleton so state persists across requests.
_manager: Optional[ComplianceManager] = None


def get_compliance_manager() -> ComplianceManager:
    """Return (or create) the singleton ComplianceManager."""
    global _manager
    if _manager is None:
        _manager = ComplianceManager()
    return _manager


class ComplianceCopilotService:
    """Thin wrapper around ComplianceManager for the REST API layer."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "max_position_pct": 0.02,
        "max_exposure_pct": 0.95,
        "max_shares_per_symbol": 200,
        "allow_short_selling": False,
        "min_stock_price": 5.0,
    }

    def __init__(self, manager: Optional[ComplianceManager] = None):
        self.manager = manager or get_compliance_manager()

    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, int],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a pre-trade compliance check."""
        merged_config = {**self.DEFAULT_CONFIG, **config}
        return self.manager.pre_trade_check(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            portfolio_value=portfolio_value,
            current_positions=current_positions,
            config=merged_config,
        )

    def generate_daily_report(self, date: Optional[str] = None) -> str:
        """Generate a daily compliance report."""
        return self.manager.generate_daily_report(date=date)

    def get_statistics(self) -> Dict[str, Any]:
        """Return compliance statistics."""
        return self.manager.get_statistics()

    def verify_audit_trail(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Verify audit trail integrity for a given date."""
        return self.manager.verify_audit_trail(date=date)
