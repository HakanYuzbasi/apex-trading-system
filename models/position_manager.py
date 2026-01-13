# models/position_manager.py - Position tracking and synchronization

from typing import Dict, Any, Optional
import asyncio


class PositionManager:
    """Manage and sync positions with IBKR."""
    
    def __init__(self, ibkr_connector):
        self.ibkr = ibkr_connector
        self.positions: Dict[str, Dict[str, Any]] = {}
        self._sync_lock = asyncio.Lock()
    
    async def sync_positions(self) -> None:
        """Sync positions from IBKR."""
        async with self._sync_lock:
            try:
                positions = await self.ibkr.get_all_positions()
                self.positions = positions or {}
            except Exception as e:
                print(f"Error syncing positions: {e}")
                raise
    
    async def update_position(self, symbol: str, quantity: int, price: float) -> None:
        """Update position with new price."""
        if symbol in self.positions:
            self.positions[symbol]["current_price"] = price
            self.positions[symbol]["quantity"] = quantity
    
    def calculate_pnl(self, symbol: str) -> float:
        """Calculate P&L for position."""
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        return (pos["current_price"] - pos["avg_cost"]) * pos["quantity"]
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position details."""
        return self.positions.get(symbol)
