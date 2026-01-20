"""
reconciliation/position_reconciler.py - Position Reconciliation

Scheduled synchronization between local state and IBKR to detect:
- Positions opened/closed outside the system
- Quantity mismatches
- Orphaned positions
- Corporate actions (splits, dividends)

Usage:
    reconciler = PositionReconciler(ibkr_connector)
    result = await reconciler.reconcile()
    if result.has_discrepancies:
        # Handle discrepancies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of position discrepancies."""
    MISSING_LOCAL = "missing_local"      # Position at broker, not in local state
    MISSING_BROKER = "missing_broker"    # Position in local state, not at broker
    QUANTITY_MISMATCH = "quantity_mismatch"  # Different quantities
    SIDE_MISMATCH = "side_mismatch"      # Long vs Short mismatch
    PRICE_DRIFT = "price_drift"          # Entry price significantly different


@dataclass
class Discrepancy:
    """Represents a single position discrepancy."""
    symbol: str
    type: DiscrepancyType
    local_qty: int
    broker_qty: int
    local_value: float
    broker_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_action: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'type': self.type.value,
            'local_qty': self.local_qty,
            'broker_qty': self.broker_qty,
            'local_value': self.local_value,
            'broker_value': self.broker_value,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolution_action': self.resolution_action
        }


@dataclass
class ReconciliationResult:
    """Result of a reconciliation check."""
    timestamp: datetime
    local_positions: Dict[str, int]
    broker_positions: Dict[str, int]
    discrepancies: List[Discrepancy]
    local_total_value: float
    broker_total_value: float
    value_difference: float
    reconciliation_time_ms: float

    @property
    def has_discrepancies(self) -> bool:
        return len(self.discrepancies) > 0

    @property
    def is_critical(self) -> bool:
        """Check if discrepancies are critical (>5% value difference)."""
        if self.broker_total_value == 0:
            return self.local_total_value > 0
        return abs(self.value_difference / self.broker_total_value) > 0.05

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'has_discrepancies': self.has_discrepancies,
            'is_critical': self.is_critical,
            'discrepancy_count': len(self.discrepancies),
            'local_position_count': len(self.local_positions),
            'broker_position_count': len(self.broker_positions),
            'local_total_value': self.local_total_value,
            'broker_total_value': self.broker_total_value,
            'value_difference': self.value_difference,
            'reconciliation_time_ms': self.reconciliation_time_ms,
            'discrepancies': [d.to_dict() for d in self.discrepancies]
        }


class PositionReconciler:
    """
    Reconciles local position state with IBKR broker state.

    Features:
    - Scheduled reconciliation (configurable interval)
    - Discrepancy detection and categorization
    - Auto-resolution for minor discrepancies
    - Alert generation for critical mismatches
    - Audit trail logging
    """

    def __init__(
        self,
        ibkr_connector,
        reconcile_interval_minutes: int = 5,
        auto_resolve_threshold: float = 0.01,  # 1% tolerance
        data_dir: Path = Path('data')
    ):
        """
        Initialize position reconciler.

        Args:
            ibkr_connector: IBKR connector instance
            reconcile_interval_minutes: How often to reconcile
            auto_resolve_threshold: Auto-resolve if difference < this %
            data_dir: Directory for reconciliation logs
        """
        self.ibkr = ibkr_connector
        self.reconcile_interval = reconcile_interval_minutes
        self.auto_resolve_threshold = auto_resolve_threshold
        self.data_dir = data_dir

        # State
        self.last_reconciliation: Optional[ReconciliationResult] = None
        self.reconciliation_history: List[ReconciliationResult] = []
        self.unresolved_discrepancies: List[Discrepancy] = []

        # Callbacks
        self._on_discrepancy_callbacks: List[callable] = []
        self._on_critical_callbacks: List[callable] = []

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📊 Position Reconciler initialized")
        logger.info(f"   Interval: {reconcile_interval_minutes} minutes")
        logger.info(f"   Auto-resolve threshold: {auto_resolve_threshold*100:.1f}%")

    def on_discrepancy(self, callback: callable):
        """Register callback for discrepancy detection."""
        self._on_discrepancy_callbacks.append(callback)

    def on_critical(self, callback: callable):
        """Register callback for critical discrepancies."""
        self._on_critical_callbacks.append(callback)

    async def reconcile(
        self,
        local_positions: Dict[str, int],
        price_cache: Dict[str, float]
    ) -> ReconciliationResult:
        """
        Perform position reconciliation.

        Args:
            local_positions: Current local position state {symbol: qty}
            price_cache: Current prices {symbol: price}

        Returns:
            ReconciliationResult with any discrepancies found
        """
        start_time = datetime.now()
        discrepancies: List[Discrepancy] = []

        try:
            # Get broker positions
            broker_positions = await self.ibkr.get_all_positions()

            # Calculate values
            local_value = sum(
                abs(qty) * price_cache.get(sym, 0)
                for sym, qty in local_positions.items()
                if qty != 0
            )
            broker_value = sum(
                abs(qty) * price_cache.get(sym, 0)
                for sym, qty in broker_positions.items()
                if qty != 0
            )

            # Get all symbols
            all_symbols: Set[str] = set(local_positions.keys()) | set(broker_positions.keys())

            # Check each symbol
            for symbol in all_symbols:
                local_qty = local_positions.get(symbol, 0)
                broker_qty = broker_positions.get(symbol, 0)

                # Skip if both zero
                if local_qty == 0 and broker_qty == 0:
                    continue

                price = price_cache.get(symbol, 0)
                local_sym_value = abs(local_qty) * price
                broker_sym_value = abs(broker_qty) * price

                # Check for discrepancies
                if local_qty == 0 and broker_qty != 0:
                    # Position at broker but not local
                    discrepancies.append(Discrepancy(
                        symbol=symbol,
                        type=DiscrepancyType.MISSING_LOCAL,
                        local_qty=local_qty,
                        broker_qty=broker_qty,
                        local_value=local_sym_value,
                        broker_value=broker_sym_value
                    ))
                    logger.warning(f"⚠️ RECONCILE: {symbol} at broker ({broker_qty}) but not local")

                elif local_qty != 0 and broker_qty == 0:
                    # Position local but not at broker
                    discrepancies.append(Discrepancy(
                        symbol=symbol,
                        type=DiscrepancyType.MISSING_BROKER,
                        local_qty=local_qty,
                        broker_qty=broker_qty,
                        local_value=local_sym_value,
                        broker_value=broker_sym_value
                    ))
                    logger.warning(f"⚠️ RECONCILE: {symbol} local ({local_qty}) but not at broker")

                elif local_qty != broker_qty:
                    # Quantity mismatch
                    if (local_qty > 0) != (broker_qty > 0):
                        disc_type = DiscrepancyType.SIDE_MISMATCH
                        logger.error(f"🚨 RECONCILE: {symbol} SIDE MISMATCH local={local_qty} broker={broker_qty}")
                    else:
                        disc_type = DiscrepancyType.QUANTITY_MISMATCH
                        logger.warning(f"⚠️ RECONCILE: {symbol} qty mismatch local={local_qty} broker={broker_qty}")

                    discrepancies.append(Discrepancy(
                        symbol=symbol,
                        type=disc_type,
                        local_qty=local_qty,
                        broker_qty=broker_qty,
                        local_value=local_sym_value,
                        broker_value=broker_sym_value
                    ))

            # Calculate timing
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Create result
            result = ReconciliationResult(
                timestamp=start_time,
                local_positions=local_positions.copy(),
                broker_positions=broker_positions.copy(),
                discrepancies=discrepancies,
                local_total_value=local_value,
                broker_total_value=broker_value,
                value_difference=local_value - broker_value,
                reconciliation_time_ms=elapsed_ms
            )

            # Store result
            self.last_reconciliation = result
            self.reconciliation_history.append(result)

            # Keep only last 100 reconciliations
            if len(self.reconciliation_history) > 100:
                self.reconciliation_history = self.reconciliation_history[-100:]

            # Log results
            if result.has_discrepancies:
                logger.warning(f"📊 Reconciliation found {len(discrepancies)} discrepancies")
                logger.warning(f"   Value diff: ${result.value_difference:,.2f}")

                # Fire callbacks
                for callback in self._on_discrepancy_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Discrepancy callback error: {e}")

                # Check if critical
                if result.is_critical:
                    logger.error(f"🚨 CRITICAL: Position value mismatch > 5%!")
                    for callback in self._on_critical_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Critical callback error: {e}")
            else:
                logger.debug(f"✅ Reconciliation OK ({elapsed_ms:.1f}ms)")

            # Save to file
            self._save_reconciliation(result)

            return result

        except Exception as e:
            logger.error(f"❌ Reconciliation error: {e}")
            raise

    def auto_resolve(
        self,
        result: ReconciliationResult,
        update_local_callback: callable
    ) -> List[Discrepancy]:
        """
        Attempt to auto-resolve minor discrepancies.

        Args:
            result: Reconciliation result
            update_local_callback: Function to update local state

        Returns:
            List of resolved discrepancies
        """
        resolved = []

        for disc in result.discrepancies:
            # Only auto-resolve quantity mismatches within threshold
            if disc.type == DiscrepancyType.QUANTITY_MISMATCH:
                if disc.broker_value > 0:
                    diff_pct = abs(disc.local_qty - disc.broker_qty) / abs(disc.broker_qty)
                    if diff_pct <= self.auto_resolve_threshold:
                        # Update local to match broker
                        try:
                            update_local_callback(disc.symbol, disc.broker_qty)
                            disc.resolved = True
                            disc.resolution_action = f"Auto-synced to broker qty ({disc.broker_qty})"
                            resolved.append(disc)
                            logger.info(f"✅ Auto-resolved {disc.symbol}: {disc.local_qty} -> {disc.broker_qty}")
                        except Exception as e:
                            logger.error(f"Auto-resolve failed for {disc.symbol}: {e}")

            # Auto-resolve missing local by adding broker position
            elif disc.type == DiscrepancyType.MISSING_LOCAL:
                try:
                    update_local_callback(disc.symbol, disc.broker_qty)
                    disc.resolved = True
                    disc.resolution_action = f"Added from broker ({disc.broker_qty})"
                    resolved.append(disc)
                    logger.info(f"✅ Auto-added {disc.symbol} from broker: {disc.broker_qty}")
                except Exception as e:
                    logger.error(f"Auto-add failed for {disc.symbol}: {e}")

        return resolved

    def _save_reconciliation(self, result: ReconciliationResult):
        """Save reconciliation result to file."""
        try:
            filename = self.data_dir / f"reconciliation_{result.timestamp.strftime('%Y%m%d')}.jsonl"
            with open(filename, 'a') as f:
                f.write(json.dumps(result.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to save reconciliation: {e}")

    def get_status(self) -> Dict:
        """Get reconciler status for dashboard."""
        return {
            'last_reconciliation': self.last_reconciliation.timestamp.isoformat() if self.last_reconciliation else None,
            'last_has_discrepancies': self.last_reconciliation.has_discrepancies if self.last_reconciliation else False,
            'total_reconciliations': len(self.reconciliation_history),
            'unresolved_count': len([d for d in self.unresolved_discrepancies if not d.resolved]),
            'interval_minutes': self.reconcile_interval
        }

    async def run_scheduled(
        self,
        get_local_positions: callable,
        get_price_cache: callable,
        update_local_callback: callable = None
    ):
        """
        Run reconciliation on a schedule.

        Args:
            get_local_positions: Function returning current local positions
            get_price_cache: Function returning current price cache
            update_local_callback: Optional function to update local state
        """
        logger.info(f"📊 Starting scheduled reconciliation (every {self.reconcile_interval} min)")

        while True:
            try:
                local_positions = get_local_positions()
                price_cache = get_price_cache()

                result = await self.reconcile(local_positions, price_cache)

                # Auto-resolve if callback provided
                if update_local_callback and result.has_discrepancies:
                    self.auto_resolve(result, update_local_callback)

            except Exception as e:
                logger.error(f"Scheduled reconciliation error: {e}")

            await asyncio.sleep(self.reconcile_interval * 60)
