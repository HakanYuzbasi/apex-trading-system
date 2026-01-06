"""
execution/smart_order_router.py
SMART ORDER ROUTING
- Best venue selection
- Optimal algorithm selection
- Dark pool access
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SmartOrderRouter:
    """
    Route orders to optimal venues and execution algorithms.
    
    Features:
    - Multi-venue price comparison
    - Algorithm selection (Market, VWAP, TWAP, etc.)
    - Dark pool routing for large orders
    - Best execution guarantee
    """
    
    def __init__(self):
        self.venue_history = {}
        self.routing_stats = []
        
        logger.info("✅ Smart Order Router initialized")
    
    def get_best_venue(
        self,
        symbol: str,
        side: str,
        quantity: int
    ) -> Dict:
        """
        Select best execution venue.
        
        Available venues:
        - SMART (IB's smart routing)
        - NASDAQ
        - NYSE
        - ARCA
        - BATS
        - Dark Pools (for large orders)
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order size
        
        Returns:
            Best venue recommendation
        """
        # Simulate venue prices (in reality, query actual venues)
        venues = {
            'SMART': self._get_venue_quote('SMART', symbol, side),
            'NASDAQ': self._get_venue_quote('NASDAQ', symbol, side),
            'NYSE': self._get_venue_quote('NYSE', symbol, side),
            'ARCA': self._get_venue_quote('ARCA', symbol, side),
            'BATS': self._get_venue_quote('BATS', symbol, side),
        }
        
        # For large orders, consider dark pools
        if quantity > 5000:
            venues['DARK'] = self._get_venue_quote('DARK', symbol, side)
        
        # Select best venue based on price
        if side == 'BUY':
            # Best bid (highest for selling, lowest for buying)
            best_venue = min(venues.items(), key=lambda x: x[1]['ask'])
        else:
            best_venue = max(venues.items(), key=lambda x: x[1]['bid'])
        
        venue_name, quote = best_venue
        
        # Calculate savings vs worst venue
        if side == 'BUY':
            worst_price = max(v['ask'] for v in venues.values())
            price_improvement = worst_price - quote['ask']
        else:
            worst_price = min(v['bid'] for v in venues.values())
            price_improvement = quote['bid'] - worst_price
        
        savings = price_improvement * quantity
        
        result = {
            'venue': venue_name,
            'quote': quote,
            'price': quote['ask'] if side == 'BUY' else quote['bid'],
            'price_improvement': price_improvement,
            'estimated_savings': savings,
            'all_venues': venues
        }
        
        logger.info(f"🎯 Best Venue for {symbol}:")
        logger.info(f"   Selected: {venue_name}")
        logger.info(f"   Price: ${result['price']:.2f}")
        logger.info(f"   Savings: ${savings:.2f} vs worst venue")
        
        return result
    
    def _get_venue_quote(self, venue: str, symbol: str, side: str) -> Dict:
        """Simulate getting quote from venue."""
        # In reality, would query actual venue
        # For now, simulate with random spreads
        
        base_price = 180.0  # Placeholder
        
        # Different venues have different spreads
        spreads = {
            'SMART': 0.01,
            'NASDAQ': 0.015,
            'NYSE': 0.012,
            'ARCA': 0.013,
            'BATS': 0.014,
            'DARK': 0.005  # Tighter spread in dark pool
        }
        
        spread = spreads.get(venue, 0.02)
        
        # Add some randomness
        price_noise = np.random.uniform(-0.005, 0.005)
        mid_price = base_price + price_noise
        
        return {
            'bid': mid_price - spread/2,
            'ask': mid_price + spread/2,
            'spread': spread,
            'timestamp': datetime.now()
        }
    
    def select_algorithm(
        self,
        symbol: str,
        side: str,
        quantity: int,
        urgency: str,
        participation_rate: float = None,
        daily_volume: float = 1_000_000
    ) -> Dict:
        """
        Select optimal execution algorithm.
        
        Algorithms:
        - MARKET: Immediate execution
        - LIMIT: Price limit
        - VWAP: Volume-weighted
        - TWAP: Time-weighted
        - ICEBERG: Hidden size
        - POV: Percentage of volume
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order size
            urgency: 'low', 'medium', 'high', 'critical'
            participation_rate: Target participation (for impact estimation)
            daily_volume: Average daily volume
        
        Returns:
            Algorithm recommendation
        """
        if participation_rate is None:
            participation_rate = quantity / daily_volume
        
        # Decision tree for algorithm selection
        if urgency == 'critical':
            # Need immediate execution
            algorithm = 'MARKET'
            params = {}
            reason = "Critical urgency - immediate execution required"
        
        elif participation_rate < 0.01:
            # Small order
            algorithm = 'LIMIT'
            params = {'time_limit_minutes': 5}
            reason = "Small order - limit order acceptable"
        
        elif participation_rate < 0.05:
            # Medium order
            algorithm = 'TWAP'
            params = {'time_horizon_minutes': 30, 'slice_interval_seconds': 60}
            reason = "Medium order - TWAP to minimize impact"
        
        elif participation_rate < 0.10:
            # Large order
            algorithm = 'VWAP'
            params = {'time_horizon_minutes': 60, 'participation_rate': 0.10}
            reason = "Large order - VWAP to match volume"
        
        else:
            # Very large order
            if quantity > 10000:
                algorithm = 'ICEBERG'
                params = {'visible_quantity': 500}
                reason = "Very large order - hide size with iceberg"
            else:
                algorithm = 'POV'
                params = {'target_participation': 0.05, 'time_horizon_minutes': 120}
                reason = "Very large order - POV to manage impact"
        
        result = {
            'algorithm': algorithm,
            'params': params,
            'reason': reason,
            'participation_rate': participation_rate,
            'estimated_duration_minutes': self._estimate_duration(algorithm, params)
        }
        
        logger.info(f"🤖 Algorithm Selection for {symbol}:")
        logger.info(f"   Participation: {participation_rate*100:.2f}%")
        logger.info(f"   Algorithm: {algorithm}")
        logger.info(f"   Reason: {reason}")
        logger.info(f"   Est. Duration: {result['estimated_duration_minutes']:.0f} min")
        
        return result
    
    def _estimate_duration(self, algorithm: str, params: Dict) -> float:
        """Estimate execution duration in minutes."""
        if algorithm == 'MARKET':
            return 0.1  # Immediate
        elif algorithm == 'LIMIT':
            return params.get('time_limit_minutes', 5)
        elif algorithm == 'TWAP':
            return params.get('time_horizon_minutes', 30)
        elif algorithm == 'VWAP':
            return params.get('time_horizon_minutes', 60)
        elif algorithm == 'ICEBERG':
            return 30  # Depends on size
        elif algorithm == 'POV':
            return params.get('time_horizon_minutes', 120)
        else:
            return 10  # Default
    
    def route_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        urgency: str = 'medium',
        price_limit: Optional[float] = None
    ) -> Dict:
        """
        Complete smart routing decision.
        
        Returns:
            {
                'venue': str,
                'algorithm': str,
                'params': dict,
                'estimated_price': float,
                'estimated_duration': float,
                'estimated_cost': float
            }
        """
        # Get best venue
        venue_selection = self.get_best_venue(symbol, side, quantity)
        
        # Select algorithm
        algo_selection = self.select_algorithm(
            symbol, side, quantity, urgency,
            daily_volume=10_000_000  # Would get from market data
        )
        
        # Calculate estimated costs
        estimated_price = venue_selection['price']
        estimated_impact = algo_selection['participation_rate'] * 0.001 * estimated_price
        estimated_cost = 1.0 + estimated_impact  # Commission + impact
        
        routing_decision = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'venue': venue_selection['venue'],
            'algorithm': algo_selection['algorithm'],
            'params': algo_selection['params'],
            'estimated_price': estimated_price,
            'estimated_duration_minutes': algo_selection['estimated_duration_minutes'],
            'estimated_total_cost': estimated_cost,
            'estimated_savings': venue_selection['estimated_savings'],
            'timestamp': datetime.now()
        }
        
        # Record routing decision
        self.routing_stats.append(routing_decision)
        
        logger.info(f"🎯 Smart Routing Decision:")
        logger.info(f"   {side} {quantity} {symbol}")
        logger.info(f"   Route: {routing_decision['venue']} via {routing_decision['algorithm']}")
        logger.info(f"   Est. Price: ${estimated_price:.2f}")
        logger.info(f"   Est. Duration: {routing_decision['estimated_duration_minutes']:.0f} min")
        logger.info(f"   Est. Cost: ${estimated_cost:.2f}")
        
        return routing_decision
    
    def get_routing_statistics(self) -> Dict:
        """Get historical routing statistics."""
        if not self.routing_stats:
            return {}
        
        df = pd.DataFrame(self.routing_stats)
        
        stats = {
            'total_orders': len(df),
            'venue_distribution': df['venue'].value_counts().to_dict(),
            'algorithm_distribution': df['algorithm'].value_counts().to_dict(),
            'total_savings': df['estimated_savings'].sum(),
            'avg_duration': df['estimated_duration_minutes'].mean()
        }
        
        logger.info(f"📊 Smart Routing Statistics:")
        logger.info(f"   Total Orders: {stats['total_orders']}")
        logger.info(f"   Total Savings: ${stats['total_savings']:,.2f}")
        logger.info(f"   Venue Usage: {stats['venue_distribution']}")
        logger.info(f"   Algorithm Usage: {stats['algorithm_distribution']}")
        
        return stats


if __name__ == "__main__":
    # Test smart order router
    import pandas as pd
    logging.basicConfig(level=logging.INFO)
    
    router = SmartOrderRouter()
    
    print("\n" + "="*60)
    print("TEST 1: VENUE SELECTION")
    print("="*60)
    
    venue = router.get_best_venue('AAPL', 'BUY', 1000)
    
    print("\n" + "="*60)
    print("TEST 2: ALGORITHM SELECTION")
    print("="*60)
    
    scenarios = [
        ('Small order', 'AAPL', 'BUY', 100, 'medium'),
        ('Medium order', 'AAPL', 'BUY', 5000, 'medium'),
        ('Large order', 'AAPL', 'BUY', 50000, 'low'),
        ('Critical order', 'AAPL', 'SELL', 1000, 'critical'),
    ]
    
    for name, symbol, side, qty, urgency in scenarios:
        print(f"\n{name}:")
        algo = router.select_algorithm(symbol, side, qty, urgency)
    
    print("\n" + "="*60)
    print("TEST 3: COMPLETE ROUTING")
    print("="*60)
    
    routing = router.route_order('AAPL', 'BUY', 10000, urgency='medium')
    
    print("\n" + "="*60)
    print("TEST 4: ROUTING STATISTICS")
    print("="*60)
    
    # Make several routing decisions
    for i in range(5):
        router.route_order('AAPL', 'BUY', np.random.randint(100, 10000), urgency='medium')
    
    stats = router.get_routing_statistics()
    
    print("\n✅ Smart order router tests complete!")
