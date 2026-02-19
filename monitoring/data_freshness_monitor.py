"""
Data Freshness Monitor

Monitors data staleness and alerts when prices or market data become outdated.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from monitoring.alert_aggregator import send_alert, AlertSeverity

logger = logging.getLogger(__name__)


class DataFreshnessMonitor:
    """Monitor data freshness and alert on stale data."""
    
    def __init__(
        self,
        price_staleness_threshold_seconds: int = 60,
        state_staleness_threshold_seconds: int = 30
    ):
        self.price_staleness_threshold = price_staleness_threshold_seconds
        self.state_staleness_threshold = state_staleness_threshold_seconds
        
        # Track last check times
        self.last_price_check: Dict[str, datetime] = {}
        self.last_state_check: Optional[datetime] = None
        
    def check_price_freshness(
        self,
        symbol: str,
        price_timestamp: datetime,
        current_price: float
    ) -> bool:
        """
        Check if price data is fresh.
        
        Returns:
            True if fresh, False if stale
        """
        age_seconds = (datetime.now() - price_timestamp).total_seconds()
        
        if age_seconds > self.price_staleness_threshold:
            send_alert(
                alert_type="STALE_PRICE_DATA",
                message=f"Stale price for {symbol}: {age_seconds:.0f}s old (price: ${current_price:.2f})",
                severity=AlertSeverity.WARNING,
                metadata={
                    "symbol": symbol,
                    "age_seconds": age_seconds,
                    "price": current_price,
                    "threshold": self.price_staleness_threshold
                }
            )
            return False
        
        return True
    
    def check_state_freshness(self, state_timestamp: str) -> bool:
        """
        Check if trading state is fresh.
        
        Returns:
            True if fresh, False if stale
        """
        try:
            state_time = datetime.fromisoformat(state_timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            send_alert(
                alert_type="INVALID_STATE_TIMESTAMP",
                message=f"Invalid state timestamp: {state_timestamp}",
                severity=AlertSeverity.ERROR
            )
            return False
        
        # Handle timezone-aware comparison
        now = datetime.now(state_time.tzinfo) if state_time.tzinfo else datetime.now()
        age_seconds = (now - state_time).total_seconds()
        
        if age_seconds > self.state_staleness_threshold:
            send_alert(
                alert_type="STALE_TRADING_STATE",
                message=f"Trading state is stale: {age_seconds:.0f}s old",
                severity=AlertSeverity.ERROR,
                metadata={
                    "age_seconds": age_seconds,
                    "threshold": self.state_staleness_threshold,
                    "last_update": state_timestamp
                }
            )
            return False
        
        return True
    
    def check_market_data_gap(
        self,
        symbol: str,
        last_update: datetime,
        expected_interval_seconds: int = 60
    ) -> bool:
        """
        Check for gaps in market data updates.
        
        Returns:
            True if no gap, False if gap detected
        """
        last_check = self.last_price_check.get(symbol)
        
        if last_check:
            gap_seconds = (last_update - last_check).total_seconds()
            
            if gap_seconds > expected_interval_seconds * 2:
                send_alert(
                    alert_type="MARKET_DATA_GAP",
                    message=f"Market data gap for {symbol}: {gap_seconds:.0f}s since last update",
                    severity=AlertSeverity.WARNING,
                    metadata={
                        "symbol": symbol,
                        "gap_seconds": gap_seconds,
                        "expected_interval": expected_interval_seconds
                    }
                )
                self.last_price_check[symbol] = last_update
                return False
        
        self.last_price_check[symbol] = last_update
        return True
    
    def validate_price_sanity(
        self,
        symbol: str,
        current_price: float,
        previous_price: Optional[float] = None,
        max_change_pct: float = 20.0
    ) -> bool:
        """
        Validate price is within reasonable bounds.
        
        Returns:
            True if price is sane, False if suspicious
        """
        # Check for zero or negative prices
        if current_price <= 0:
            send_alert(
                alert_type="INVALID_PRICE",
                message=f"Invalid price for {symbol}: ${current_price:.2f}",
                severity=AlertSeverity.CRITICAL,
                metadata={"symbol": symbol, "price": current_price}
            )
            return False
        
        # Check for extreme price changes
        if previous_price and previous_price > 0:
            change_pct = abs((current_price - previous_price) / previous_price * 100)
            
            if change_pct > max_change_pct:
                send_alert(
                    alert_type="EXTREME_PRICE_CHANGE",
                    message=f"Extreme price change for {symbol}: {change_pct:.1f}% "
                           f"(${previous_price:.2f} → ${current_price:.2f})",
                    severity=AlertSeverity.WARNING,
                    metadata={
                        "symbol": symbol,
                        "previous_price": previous_price,
                        "current_price": current_price,
                        "change_pct": change_pct,
                        "threshold_pct": max_change_pct
                    }
                )
                return False
        
        return True


# Global instance
_monitor: Optional[DataFreshnessMonitor] = None


def get_freshness_monitor(
    price_staleness_threshold: int = 60,
    state_staleness_threshold: int = 30
) -> DataFreshnessMonitor:
    """Get or create global freshness monitor."""
    global _monitor
    if _monitor is None:
        _monitor = DataFreshnessMonitor(
            price_staleness_threshold,
            state_staleness_threshold
        )
    return _monitor
