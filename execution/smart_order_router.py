import logging
import asyncio
from typing import Callable, Awaitable, Optional, Dict, Any

logger = logging.getLogger(__name__)


def _is_crypto_symbol(symbol: str) -> bool:
    """Return True if symbol is a crypto asset (trades 24/7 on Alpaca)."""
    try:
        from core.symbols import parse_symbol, AssetClass
        return parse_symbol(symbol).asset_class == AssetClass.CRYPTO
    except Exception:
        return False

class SmartOrderRouter:
    """
    APEX Institutional Smart Order Router (SOR)
    Implements Adaptive Pegging: Starts at the Mid-Price and gradually increases
    urgency to cross the spread only if necessary.
    """
    def __init__(self, max_urgency_steps: int = 3, step_delay_seconds: int = 10):
        self.max_steps = max_urgency_steps
        self.step_delay = step_delay_seconds

    def select_algorithm(self, symbol: str, side: str, quantity: float, urgency: str = "medium", **kwargs) -> Dict[str, Any]:
        """
        Recommends an execution algorithm based on order characteristics.
        Used by simulators and high-level execution managers.
        """
        # Crypto: trades 24/7, no VWAP/TWAP (no defined session volume), much larger daily volume
        if _is_crypto_symbol(symbol):
            # BTC/ETH daily volume ~$10-30B — participation rate math is equity-irrelevant.
            # Use ADAPTIVE (mid-peg) for low urgency; fall back to MARKET for high/critical.
            if urgency in ("critical", "high"):
                return {
                    "algorithm": "MARKET",
                    "estimated_duration_minutes": 0,
                    "reason": "Crypto high-urgency: immediate market execution.",
                }
            return {
                "algorithm": "ADAPTIVE",
                "estimated_duration_minutes": 5,
                "reason": "Crypto 24/7: adaptive mid-peg — no VWAP/TWAP on continuous market.",
            }

        # Equity / Forex: session-based volume, VWAP/TWAP apply
        daily_volume = kwargs.get("daily_volume", 1_000_000)
        participation_rate = quantity / max(1, daily_volume)

        if urgency == "critical" or participation_rate > 0.1:
            return {
                "algorithm": "MARKET",
                "estimated_duration_minutes": 0,
                "reason": "High urgency or large size relative to volume - immediate execution."
            }
        elif urgency == "high" or participation_rate > 0.03:
            return {
                "algorithm": "VWAP",
                "estimated_duration_minutes": 45,
                "reason": "Moderate urgency - VWAP recommended to balance cost and speed."
            }
        elif participation_rate < 0.01 and urgency == "low":
            return {
                "algorithm": "ADAPTIVE",
                "estimated_duration_minutes": 20,
                "reason": "Small order with low urgency - capture spread with adaptive pegging."
            }
        else:
            return {
                "algorithm": "TWAP",
                "estimated_duration_minutes": 30,
                "reason": "Standard order - TWAP recommended for steady execution."
            }

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
