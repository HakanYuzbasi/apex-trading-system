from __future__ import annotations

import logging
import asyncio
from typing import Dict, Any, Optional
import numpy as np

from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events.execution import ExecutionEvent
from quant_system.portfolio.ledger import PortfolioLedger
from reconciliation.position_reconciler import PositionReconciler

logger = logging.getLogger("shadow_accounting")

class ShadowAccounting:
    """
    Independent position tracking system that listens to ExecutionEvents 
    and verifies the primary PortfolioLedger against broker reality.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        ledger: PortfolioLedger,
        reconciler: PositionReconciler,
        ibkr_connector: Optional[Any] = None,
        notifier: Optional[Any] = None,
        check_interval_seconds: int = 300,
        single_venue: bool = False,
    ) -> None:
        self._event_bus = event_bus
        self._ledger = ledger
        self._reconciler = reconciler
        self._ibkr = ibkr_connector
        self._notifier = notifier
        self._check_interval = check_interval_seconds
        # When True, skip multi-venue sovereign equity and capital imbalance checks
        # (irrelevant and noisy when running on a single broker).
        self._single_venue = single_venue
        
        self._subscription: Optional[Subscription] = None
        self._stop_event = asyncio.Event()
        # Seed from ledger so shadow doesn't diverge on every restart
        self.shadow_positions: Dict[str, float] = {
            sym: pos.quantity for sym, pos in self._ledger.positions.items()
        }

    def start(self) -> None:
        """Subscribe to execution events and start the mirror."""
        self._subscription = self._event_bus.subscribe("execution", self._on_execution)
        logger.info("🕵️ ShadowAccounting: Monitoring started (%d positions seeded from ledger)", len(self.shadow_positions))

    async def _on_execution(self, event: Any) -> None:
        if not isinstance(event, ExecutionEvent):
            return
            
        if event.execution_status not in {"partial_fill", "filled"}:
            return
            
        # NORMALIZE symbol to avoid duplicate keys (e.g., XRPUSD vs CRYPTO:XRP/USD)
        from core.symbols import normalize_symbol
        symbol = normalize_symbol(event.instrument_id)
        qty = float(event.fill_qty)
        signed_qty = qty if event.side == "buy" else -qty
        
        current = self.shadow_positions.get(symbol, 0.0)
        self.shadow_positions[symbol] = current + signed_qty
        
        logger.debug(f"Shadow update for {symbol}: {current} -> {self.shadow_positions[symbol]}")

    async def _fetch_venue_balance(self, venue: str) -> float:
        """Fetch real-world equity from a specific venue via the reconciler's adapter."""
        if venue == "alpaca":
            if hasattr(self._reconciler.ibkr, "get_account_equity"):
                try:
                    return await self._reconciler.ibkr.get_account_equity()
                except Exception as e:
                    logger.error(f"Failed to fetch real Alpaca equity: {e}")
        elif venue == "ibkr":
            # For IBKR, we use get_net_liquidation() if available on the connector
            # Use the passed-in IBKR connector if available
            if self._ibkr and hasattr(self._ibkr, "get_net_liquidation"):
                try:
                    # is_connected() checks both ib.isConnected() AND offline_mode flag
                    connected = (
                        self._ibkr.is_connected()
                        if hasattr(self._ibkr, "is_connected")
                        else self._ibkr.ib.isConnected()
                    )
                    if not connected:
                        return 0.0
                    
                    # Use async version to avoid 'Event loop is already running' errors
                    summary = await self._ibkr.ib.accountSummaryAsync()
                    for item in summary:
                        if item.tag == 'NetLiquidation':
                            return float(item.value)
                    
                    # Fallback to accountValues if summary is empty
                    acc_values = self._ibkr.ib.accountValues()
                    for v in acc_values:
                        if v.tag == 'NetLiquidation' and v.currency == 'USD':
                            return float(v.value)
                except Exception as e:
                    logger.error(f"Failed to fetch real IBKR equity: {e}")
            
        # Return 0.0 for unconfigured or non-existent mock venues to avoid false alerts
        return 0.0

    async def run_reconciliation_loop(self) -> None:
        """Periodic verification against PortfolioLedger and Broker."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._check_interval)
                await self.verify_integrity()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ShadowAccounting verification error: {e}")

    async def verify_integrity(self) -> None:
        """Compare shadow vs ledger and broker multi-venue sovereign equity."""
        logger.info("🕵️ ShadowAccounting: Running integrity check...")
        
        discrepancies = []
        
        # 1. Compare Shadow vs Local Ledger
        all_symbols = set(self.shadow_positions.keys()) | set(self._ledger.positions.keys())
        
        for symbol in all_symbols:
            shadow_qty = self.shadow_positions.get(symbol, 0.0)
            ledger_qty = self._ledger.get_position(symbol).quantity
            
            if abs(shadow_qty - ledger_qty) > 1e-6:
                logger.warning(f"⚠️ SHADOW DIVERGENCE: {symbol} Shadow={shadow_qty} Ledger={ledger_qty}")
                discrepancies.append(symbol)

        # 2 & 3. Multi-Venue Sovereign Equity + Capital Imbalance checks.
        # Skipped in single-venue mode (Alpaca-only) — meaningless with one broker
        # and a frequent source of false alarms when IBKR is offline.
        if not self._single_venue:
            sovereign_equity = 0.0
            venue_balances = {}
            try:
                venue_balances["alpaca"] = await self._fetch_venue_balance("alpaca")
                venue_balances["ibkr"] = await self._fetch_venue_balance("ibkr")
                active_balances = {v: b for v, b in venue_balances.items() if b > 0.0}
                sovereign_equity = sum(active_balances.values())
                logger.info(
                    "⚖️ Sovereign Equity: Total=$%.2f active_venues=%s",
                    sovereign_equity, list(active_balances.keys()),
                )
            except Exception as e:
                logger.error(f"Failed pulling multi-venue sovereign equity: {e}")

            local_total_equity = self._ledger.total_equity()

            if sovereign_equity > 0 and abs(sovereign_equity - local_total_equity) > 100.0:
                logger.critical(f"🚨 SOVEREIGN EQUITY FAILURE: Sovereign={sovereign_equity} Local={local_total_equity} (Diff > $100)")
                if self._notifier:
                    await self._notifier.notify_text("CRITICAL: Multi-Venue Sovereign Equity deviated by > $100 from internal ledgers.")
                return

            if sovereign_equity > 0 and len(active_balances) >= 2:
                weights = np.array(list(active_balances.values())) / sovereign_equity
                std_dev = np.std(weights)
                if std_dev > 0.15:
                    logger.warning(f"⚖️ CAPITAL IMBALANCE: Venue weights {weights} drifted > 2-sigma ({std_dev:.3f}).")
                    sorted_venues = sorted(active_balances.items(), key=lambda x: x[1])
                    deficit_venue, deficit_bal = sorted_venues[0]
                    surplus_venue, surplus_bal = sorted_venues[-1]
                    transfer_amt = (surplus_bal - deficit_bal) / 2.0
                    logger.info(f"💸 Rebalancer Action: Would emit TransferEvent for ${transfer_amt:.2f} from {surplus_venue} to {deficit_venue}")

        # 4. If discrepant physically on positions, trigger full Broker Reconcile
        if discrepancies:
            msg = f"⚠️ SHADOW ACCOUNTING ALERT: Position divergence detected for {discrepancies}. Triggering emergency reconcile."
            logger.warning(msg)
            if self._notifier:
                await self._notifier.notify_text(msg)
                
            # Perform reconciliation
            # We assume reconcile(local_positions, price_cache)
            local_qtys = {s: int(p.quantity) for s, p in self._ledger.positions.items()}
            # Reconciler needs a price cache, we pull from ledger
            price_cache = self._ledger.last_price_by_instrument
            
            # This triggers Alpaca/Broker call via reconciler.ibkr (generalized connector)
            result = await self._reconciler.reconcile(local_qtys, price_cache)
            
            if result.has_discrepancies:
                logger.error(f"🚨 RECONCILE CONFIRMED: Broker differs from local state for {len(result.discrepancies)} symbols.")
                # Auto-resolve (sync local to broker)
                def force_update(symbol: str, qty: float):
                    pos = self._ledger.get_position(symbol)
                    pos.quantity = float(qty)
                    # We don't adjust avg_price here for simplicity, or we should re-pull it
                    self.shadow_positions[symbol] = float(qty)
                
                self._reconciler.auto_resolve(result, force_update)
                
                if self._notifier:
                    await self._notifier.notify_text(f"✅ EMERGENCY RECONCILE COMPLETE: Local state synced to Broker for {discrepancies}.")
            else:
                logger.info("✅ Broker confirms Ledger is correct. Shadow state reset.")
                for sym in discrepancies:
                    self.shadow_positions[sym] = self._ledger.get_position(sym).quantity

    def stop(self) -> None:
        self._stop_event.set()
        if self._subscription:
            self._event_bus.unsubscribe(self._subscription.token)
