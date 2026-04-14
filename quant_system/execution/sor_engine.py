import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("sor_engine")

@dataclass
class VenueBook:
    venue: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    taker_fee_pct: float
    maker_rebate_pct: float

class SmartOrderRouter:
    """
    Multi-Venue Smart Order Router (SOR) matching across Alpaca, Coinbase, and Kraken.
    Optimizes Total Cost of Ownership (TCO) using real-time net taking prices.
    """
    def __init__(self):
        self.venues: Dict[str, VenueBook] = {}
        
        # Standardize fee assumptions for 2026 Crypto Tier 1
        self._fee_tiers = {
            "alpaca": {"taker": 0.0015, "maker": 0.0000},
            "coinbase": {"taker": 0.0020, "maker": 0.0005},
            "kraken": {"taker": 0.0012, "maker": 0.0002}
        }

    def update_venue_book(self, venue: str, bid: float, ask: float, bid_size: float, ask_size: float):
        """Update the best bid/ask snapshot for a venue."""
        fees = self._fee_tiers.get(venue, {"taker": 0.002, "maker": 0.0})
        self.venues[venue] = VenueBook(
            venue=venue,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            taker_fee_pct=fees["taker"],
            maker_rebate_pct=fees["maker"]
        )

    def slice_order(self, expected_side: str, total_qty: float) -> List[Dict]:
        """
        Slices a market-taking order across multiple venues to minimize TCO.
        TCO = Price + taker_fee.
        Returns a list of order instructions: [{venue, qty, limit_price}]
        """
        if not self.venues:
            return []

        # Sort venues by net TCO price
        # Taker buying: You pay the Ask. Total cost = ask * (1 + taker_fee_pct)
        # Taker selling: You hit the Bid. Total revenue = bid * (1 - taker_fee_pct) -> want max revenue
        
        available_liquidity = []
        for venue_name, book in self.venues.items():
            if expected_side == "buy":
                net_cost = book.ask * (1 + book.taker_fee_pct)
                available_liquidity.append({
                    "venue": venue_name,
                    "net_price": net_cost,
                    "raw_price": book.ask,
                    "available_qty": book.ask_size
                })
            else:
                net_revenue = book.bid * (1 - book.taker_fee_pct)
                available_liquidity.append({
                    "venue": venue_name,
                    "net_price": net_revenue, # We want to sort descending, but we'll handle sorting later
                    "raw_price": book.bid,
                    "available_qty": book.bid_size
                })

        # Sort mechanics
        if expected_side == "buy":
            available_liquidity.sort(key=lambda x: x["net_price"]) # Cheapest cost first
        else:
            available_liquidity.sort(key=lambda x: x["net_price"], reverse=True) # Highest revenue first

        # Sweep the books
        remaining_qty = total_qty
        child_orders = []

        for liq in available_liquidity:
            if remaining_qty <= 0:
                break
            
            qty_to_take = min(remaining_qty, liq["available_qty"])
            if qty_to_take > 0:
                child_orders.append({
                    "venue": liq["venue"],
                    "qty": qty_to_take,
                    "limit_price": liq["raw_price"] # Send standard limit; exchange handles fees
                })
                remaining_qty -= qty_to_take

        if remaining_qty > 0:
            logger.warning(f"SOR Warning: Insufficient liquidity across {len(self.venues)} venues. {remaining_qty} unfilled.")

        return child_orders
