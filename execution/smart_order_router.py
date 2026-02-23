"""
execution/smart_order_router.py
Venue-aware order router mapping liquidity depth and routing based on fees & execution speed.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class Venue:
    def __init__(self, name: str, fee_bps: float, avg_latency_ms: float):
        self.name = name
        self.fee_bps = fee_bps
        self.avg_latency_ms = avg_latency_ms
        self.book_depth = 0.0 # Dynamic depth tracked via websockets
        
    def update_depth(self, depth: float):
        self.book_depth = depth

class SmartOrderRouter:
    def __init__(self):
        # Configuration mapping for available liquidity venues
        self.venues: Dict[str, Venue] = {
            "IBKR": Venue("IBKR", fee_bps=0.5, avg_latency_ms=25.0),
            "ALPACA": Venue("ALPACA", fee_bps=1.0, avg_latency_ms=50.0),
        }
        logger.info("🛣️ Venue-Aware Smart Order Router initialized.")

    def update_venue_depth(self, venue_name: str, symbol: str, depth: float):
        if venue_name in self.venues:
            self.venues[venue_name].update_depth(depth)

    def route_order(self, symbol: str, qty: float, side: str) -> str:
        """
        Determine optimal venue based on liquidity depth, fees, and latency.
        """
        best_venue = None
        best_score = -float('inf')
        
        for name, venue in self.venues.items():
            # Higher depth is good; fee and latency are bad
            depth_score = venue.book_depth if venue.book_depth > 0 else 10000.0
            
            # Heavy penalty if order size sweeps beyond top of book
            penalty = (qty - depth_score) * 10 if qty > depth_score else 0
            
            score = depth_score - (venue.fee_bps * 100) - venue.avg_latency_ms - penalty
            
            if score > best_score:
                best_score = score
                best_venue = name
                
        selected_venue = best_venue or "IBKR"
        logger.debug(f"🔀 Routed {side} {qty} {symbol} to {selected_venue}")
        return selected_venue
