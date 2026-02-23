import logging
import asyncio
from typing import Callable, Awaitable, Optional, Dict, Any

logger = logging.getLogger(__name__)

class SmartOrderRouter:
    """
    APEX Institutional Smart Order Router (SOR)
    Implements Adaptive Pegging: Starts at the Mid-Price and gradually increases
    urgency to cross the spread only if necessary.
    """
    def __init__(self, max_urgency_steps: int = 3, step_delay_seconds: int = 10):
        self.max_steps = max_urgency_steps
        self.step_delay = step_delay_seconds

    async def execute_adaptive_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        bid: float,
        ask: float,
        place_order_fn: Callable[[float, str], Awaitable[Optional[str]]],
        check_status_fn: Callable[[str], Awaitable[str]],
        cancel_order_fn: Callable[[str], Awaitable[bool]]
    ) -> bool:
        """
        Executes the order using dynamic limits.
        
        :param place_order_fn: Async callback taking (price, order_type) -> order_id
        :param check_status_fn: Async callback taking (order_id) -> status string ('FILLED', 'OPEN')
        :param cancel_order_fn: Async callback taking (order_id) -> boolean success
        """
        mid_price = (bid + ask) / 2.0
        spread = ask - bid
        
        # Protect against zero-spread or crossed books
        if spread <= 0.0001:
            logger.info(f"SOR [{symbol}] Spread too tight. Sweeping book instantly.")
            await place_order_fn(ask if side.upper() == "BUY" else bid, "MARKET")
            return True

        current_order_id = None
        
        # --- EXECUTION LOOP ---
        for step in range(self.max_steps):
            urgency = step / max(1, (self.max_steps - 1))
            
            # Calculate peg price based on urgency
            if side.upper() == "BUY":
                # Step 0: Mid-price. Step Max: Ask
                limit_price = round(mid_price + (spread * 0.5 * urgency), 4)
            else:
                # Step 0: Mid-price. Step Max: Bid
                limit_price = round(mid_price - (spread * 0.5 * urgency), 4)
                
            phase_name = "Passive Mid-Peg" if step == 0 else f"Urgent Peg ({int(urgency*100)}%)"
            logger.info(f"SOR [{symbol}] {side} {qty} - Phase {step+1}: {phase_name} @ {limit_price}")
            
            # Cancel previous tier if it exists
            if current_order_id:
                await cancel_order_fn(current_order_id)
                
            # Place new limit order
            current_order_id = await place_order_fn(limit_price, "LIMIT")
            if not current_order_id:
                logger.error(f"SOR [{symbol}] Failed to place limit order.")
                return False
                
            # Wait for fill (Patience Timer)
            for _ in range(self.step_delay):
                await asyncio.sleep(1)
                status = await check_status_fn(current_order_id)
                if status.upper() in ["FILLED", "EXECUTED"]:
                    logger.info(f"✅ SOR [{symbol}] Filled elegantly at {limit_price}! Spread captured.")
                    return True
                    
        # --- FALLBACK: SWEEP THE BOOK ---
        logger.warning(f"⏳ SOR [{symbol}] {side} patience exhausted. Sweeping the book to guarantee execution.")
        if current_order_id:
            await cancel_order_fn(current_order_id)
            
        await place_order_fn(0.0, "MARKET")
        return True
