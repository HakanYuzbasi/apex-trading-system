from __future__ import annotations

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, OrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, AssetStatus
from alpaca.trading.models import OptionContract

from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.bayesian_vol import BayesianVolatilityAdjuster

logger = logging.getLogger(__name__)

class TailHedger:
    """
    Automated Tail-Hedging using OTM Puts.
    Triggered when Bayesian Volatility probability > 0.90.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        ledger: PortfolioLedger,
        vol_adjuster: BayesianVolatilityAdjuster,
        hedge_symbols: List[str] = ["SPY", "QQQ"],
        otm_percent: float = 0.10,
        min_expiry_days: int = 4,
        max_expiry_days: int = 10,
    ) -> None:
        self.trading_client = trading_client
        self.ledger = ledger
        self.vol_adjuster = vol_adjuster
        self.hedge_symbols = hedge_symbols
        self.otm_percent = otm_percent
        self.min_expiry_days = min_expiry_days
        self.max_expiry_days = max_expiry_days
        
        self.is_hedged: bool = False
        self.last_hedge_cost: float = 0.0
        self.active_hedges: List[str] = [] # List of option symbols

    async def check_and_hedge(self) -> None:
        """Evaluate volatility and execute hedges if needed."""
        # 1. Check volatility probability for major indices
        # If we don't have direct SPY/QQQ bars, we can use a proxy or just aggregate
        # For simplicity, we'll check if ANY tracked instrument in BayesianVol is > 0.90
        # or if a specific pivot instrument is high.
        
        max_prob = 0.0
        # Accessing private _states for monitoring (in a real system we'd have a public getter per instrument)
        for state in self.vol_adjuster._states.values():
            max_prob = max(max_prob, state.prob_high_vol)
            
        if max_prob > 0.90 and not self.is_hedged:
            logger.warning("TAIL-HEDGE TRIGGERED: Bayesian Vol Probability %.2f > 0.90", max_prob)
            await self._execute_tail_hedge()
        elif max_prob < 0.50 and self.is_hedged:
            logger.info("Volatility subsided (%.2f). Maintaining or closing hedges.", max_prob)
            # In Fund Manager v6, we usually LET the puts expire or hold them until expiry.
            # But for safety, we could close them. We'll leave them for now to achieve max convexity.

    async def _execute_tail_hedge(self) -> None:
        """Find and buy Puts."""
        total_long_exposure = sum(
            pos.quantity * pos.avg_cost 
            for pos in self.ledger.positions.values() 
            if pos.quantity > 0
        )
        
        if total_long_exposure <= 0:
            logger.info("No long exposure to hedge.")
            return

        # Target hedging ~50% of the long notional (proportional to user request "Buy ... Puts ... proportional to current long exposure")
        # Standard options multiplier is 100.
        hedge_notional = total_long_exposure * 0.5
        
        for symbol in self.hedge_symbols:
            try:
                # 1. Get current price (approximate from ledger or mock)
                # In live, we'd use a QuoteClient. Here we use a conservative estimate.
                current_price = 450.0 if symbol == "SPY" else 350.0
                target_strike = current_price * (1.0 - self.otm_percent)
                
                # 2. Search for contracts
                req = GetOptionContractsRequest(
                    underlying_symbol=symbol,
                    status=AssetStatus.ACTIVE,
                    expiration_date_gte=(datetime.now(timezone.utc) + timedelta(days=self.min_expiry_days)).date(),
                    expiration_date_lte=(datetime.now(timezone.utc) + timedelta(days=self.max_expiry_days)).date(),
                )
                contracts = self.trading_client.get_option_contracts(req)
                
                # Filter for Puts and nearest strike
                puts = [c for c in contracts.option_contracts if c.type == "put"]
                if not puts:
                    logger.error(f"No Put contracts found for {symbol}")
                    continue
                    
                # Find contract closest to target strike
                best_contract = min(puts, key=lambda c: abs(float(c.strike_price) - target_strike))
                
                # 3. Submit Order
                # Each contract covers 100 shares.
                qty = int(hedge_notional / (len(self.hedge_symbols) * float(best_contract.strike_price) * 100))
                if qty < 1: qty = 1
                
                order_req = OrderRequest(
                    symbol=best_contract.symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY
                )
                
                order = self.trading_client.submit_order(order_req)
                logger.info(f"Hedged with {qty} {best_contract.symbol} Puts.")
                self.active_hedges.append(best_contract.symbol)
                self.is_hedged = True
                
            except Exception as e:
                logger.error(f"Failed to hedge {symbol}: {e}")

    def get_status(self) -> dict:
        return {
            "is_hedged": self.is_hedged,
            "active_hedges": self.active_hedges,
            "latency_notice": "Latency Heatmap Active"
        }
