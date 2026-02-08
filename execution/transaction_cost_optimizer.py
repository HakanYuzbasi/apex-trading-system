"""
execution/transaction_cost_optimizer.py
OPTIMIZE TRANSACTION COSTS
- Market impact estimation (Almgren-Chriss model)
- Optimal execution timing
- Order consolidation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


class TransactionCostOptimizer:
    """
    Minimize transaction costs through:
    1. Market impact estimation
    2. Optimal timing
    3. Order consolidation
    4. Venue selection
    """
    
    def __init__(self):
        self.cost_history = []
        
        logger.info("✅ Transaction Cost Optimizer initialized")
    
    def estimate_market_impact(
        self,
        symbol: str,
        side: str,
        quantity: int,
        daily_volume: float,
        market_cap: float = 1e9,
        volatility: float = 0.20
    ) -> Dict:
        """
        Estimate market impact using Almgren-Chriss model.
        
        Market Impact = α * σ * (Q/V)^β
        
        where:
        - α = temporary impact coefficient (0.1)
        - σ = volatility
        - Q = order size
        - V = daily volume
        - β = elasticity (0.6-0.8)
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order size
            daily_volume: Average daily volume
            market_cap: Market capitalization
            volatility: Annualized volatility
        
        Returns:
            Impact estimation dict
        """
        if daily_volume == 0:
            logger.warning(f"{symbol}: No volume data, cannot estimate impact")
            return {
                'impact_bps': 0,
                'impact_pct': 0,
                'estimated_slippage': 0,
                'urgency': 'unknown'
            }
        
        # Participation rate
        participation = quantity / daily_volume
        
        # Almgren-Chriss parameters
        alpha = 0.1  # Temporary impact coefficient
        beta = 0.7   # Price elasticity
        
        # Temporary impact (basis points)
        temporary_impact_bps = alpha * volatility * (participation ** beta) * 10000
        
        # Permanent impact (smaller)
        permanent_impact_bps = temporary_impact_bps * 0.3
        
        # Total impact
        total_impact_bps = temporary_impact_bps + permanent_impact_bps
        total_impact_pct = total_impact_bps / 10000
        
        # Classify urgency
        if participation < 0.01:  # < 1% of volume
            urgency = 'low'
            impact_level = 'minimal'
        elif participation < 0.05:  # < 5%
            urgency = 'medium'
            impact_level = 'moderate'
        elif participation < 0.10:  # < 10%
            urgency = 'high'
            impact_level = 'significant'
        else:  # > 10%
            urgency = 'critical'
            impact_level = 'severe'
        
        result = {
            'participation_rate': participation,
            'temporary_impact_bps': temporary_impact_bps,
            'permanent_impact_bps': permanent_impact_bps,
            'total_impact_bps': total_impact_bps,
            'impact_pct': total_impact_pct,
            'urgency': urgency,
            'impact_level': impact_level,
            'recommendation': self._get_execution_recommendation(participation, urgency)
        }
        
        logger.info(f"💸 {symbol} Impact Estimate:")
        logger.info(f"   Participation: {participation*100:.2f}% of daily volume")
        logger.info(f"   Impact: {total_impact_bps:.1f} bps ({impact_level})")
        logger.info(f"   Recommendation: {result['recommendation']}")
        
        return result
    
    def _get_execution_recommendation(self, participation: float, urgency: str) -> str:
        """Get execution algorithm recommendation."""
        if participation < 0.01:
            return "Market order acceptable"
        elif participation < 0.05:
            return "Use TWAP over 30-60 minutes"
        elif participation < 0.10:
            return "Use VWAP over 1-2 hours"
        else:
            return "Use VWAP/POV over multiple hours or days"
    
    def optimize_execution_timing(
        self,
        symbol: str,
        quantity: int,
        urgency: str = 'medium'
    ) -> Dict:
        """
        Determine optimal execution timing based on:
        - Intraday volume patterns
        - Market hours
        - Historical patterns
        
        Args:
            symbol: Trading symbol
            quantity: Order size
            urgency: 'low', 'medium', 'high', 'critical'
        
        Returns:
            Timing recommendation
        """
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Market open: 9:30 AM EST = high volume
        # Mid-day: 11:00-14:00 EST = lower volume
        # Market close: 15:30-16:00 EST = high volume
        
        # Convert to EST (simplified - adjust for your timezone)
        est_hour = hour - 6  # CET to EST
        if est_hour < 0:
            est_hour += 24
        
        timing_score = 1.0  # 0-1 scale
        
        # Optimal times: market open/close
        if 9.5 <= est_hour <= 10.5:  # Market open
            timing_score = 0.9
            reason = "Market open - high liquidity"
            wait_minutes = 0
        
        elif 15.5 <= est_hour <= 16.0:  # Market close
            timing_score = 0.85
            reason = "Market close - high liquidity"
            wait_minutes = 0
        
        elif 11.0 <= est_hour <= 14.0:  # Mid-day
            timing_score = 0.5
            reason = "Mid-day - lower liquidity"
            
            if urgency == 'low':
                wait_minutes = int((15.5 - est_hour) * 60)  # Wait for close
            else:
                wait_minutes = 0
        
        elif est_hour < 9.5:  # Pre-market
            timing_score = 0.3
            reason = "Pre-market - very low liquidity"
            wait_minutes = int((9.5 - est_hour) * 60)
        
        elif est_hour > 16.0:  # After hours
            timing_score = 0.3
            reason = "After hours - wait for next open"
            wait_minutes = int((24 - est_hour + 9.5) * 60)
        
        else:
            timing_score = 0.7
            reason = "Normal trading hours"
            wait_minutes = 0
        
        # Adjust for urgency
        if urgency == 'critical':
            wait_minutes = 0
            recommendation = "Execute immediately"
        elif urgency == 'high' and wait_minutes > 30:
            wait_minutes = 0
            recommendation = "Execute now (high urgency)"
        elif urgency == 'low' and timing_score < 0.7:
            recommendation = f"Wait {wait_minutes} minutes for better timing"
        else:
            recommendation = "Execute now" if wait_minutes == 0 else f"Consider waiting {wait_minutes} min"
        
        result = {
            'timing_score': timing_score,
            'current_time_est': f"{int(est_hour)}:{int((est_hour % 1)*60):02d}",
            'reason': reason,
            'wait_minutes': wait_minutes,
            'execute_at': (now + timedelta(minutes=wait_minutes)).strftime('%H:%M') if wait_minutes > 0 else 'now',
            'recommendation': recommendation
        }
        
        logger.info(f"⏰ Execution Timing for {symbol}:")
        logger.info(f"   Current time: {result['current_time_est']} EST")
        logger.info(f"   Timing score: {timing_score:.2f}")
        logger.info(f"   {reason}")
        logger.info(f"   Recommendation: {recommendation}")
        
        return result
    
    def consolidate_orders(
        self,
        pending_orders: List[Dict],
        min_consolidation_value: float = 1000.0
    ) -> List[Dict]:
        """
        Consolidate small orders to reduce total commissions.
        
        Args:
            pending_orders: List of pending orders
            min_consolidation_value: Minimum value to consolidate
        
        Returns:
            Consolidated order list
        """
        # Group by symbol and side
        order_groups = {}
        
        for order in pending_orders:
            key = (order['symbol'], order['side'])
            
            if key not in order_groups:
                order_groups[key] = []
            
            order_groups[key].append(order)
        
        # Consolidate groups
        consolidated = []
        savings = 0
        
        for (symbol, side), orders in order_groups.items():
            if len(orders) == 1:
                # Single order, no consolidation
                consolidated.append(orders[0])
            
            else:
                # Multiple orders, consolidate
                total_qty = sum(o['quantity'] for o in orders)
                avg_price = np.average(
                    [o['price'] for o in orders],
                    weights=[o['quantity'] for o in orders]
                )
                
                total_value = sum(o['quantity'] * o['price'] for o in orders)
                
                if total_value >= min_consolidation_value:
                    # Consolidate
                    consolidated_order = {
                        'symbol': symbol,
                        'side': side,
                        'quantity': total_qty,
                        'price': avg_price,
                        'commission': 1.0,  # Single commission
                        'source': 'consolidated',
                        'original_orders': len(orders)
                    }
                    
                    consolidated.append(consolidated_order)
                    
                    # Calculate savings
                    original_commissions = len(orders) * 1.0
                    new_commission = 1.0
                    savings += (original_commissions - new_commission)
                
                else:
                    # Too small, keep separate
                    consolidated.extend(orders)
        
        if savings > 0:
            logger.info(f"💰 Order Consolidation:")
            logger.info(f"   Original: {len(pending_orders)} orders")
            logger.info(f"   Consolidated: {len(consolidated)} orders")
            logger.info(f"   Commission savings: ${savings:.2f}")
        
        return consolidated
    
    def estimate_slippage(
        self,
        symbol: str,
        side: str,
        quantity: int,
        bid_ask_spread: float,
        volatility: float,
        market_regime: str = 'neutral'
    ) -> Dict:
        """
        Estimate expected slippage with volatility and regime adjustments.

        Slippage = Spread/2 + Market Impact + Volatility Adjustment

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order size
            bid_ask_spread: Current bid-ask spread
            volatility: Annualized volatility
            market_regime: Market regime ('bull', 'bear', 'neutral', 'high_volatility')

        Returns:
            Slippage estimate with regime adjustments
        """
        # Spread cost (half spread for limit orders)
        spread_cost_bps = (bid_ask_spread / 2) * 10000

        # Impact cost (Almgren-Chriss simplified)
        impact_cost_bps = volatility * np.sqrt(quantity / 1000) * 10

        # ✅ Phase 2.3: Volatility regime adjustment
        # High volatility markets have higher slippage
        vol_multiplier = {
            'strong_bull': 0.8,   # Lower slippage in trending markets
            'bull': 0.9,
            'neutral': 1.0,
            'bear': 1.2,          # Higher slippage in falling markets
            'strong_bear': 1.5,
            'high_volatility': 2.0  # Double slippage in volatile markets
        }.get(market_regime, 1.0)

        # Adjust impact for volatility regime
        adjusted_impact_bps = impact_cost_bps * vol_multiplier

        # Time-of-day adjustment (higher slippage at open/close)
        from datetime import datetime
        import pytz
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        hour = now.hour + now.minute / 60

        if 9.5 <= hour <= 10.5:  # First hour of trading
            time_multiplier = 1.5
        elif 15.5 <= hour <= 16.0:  # Last 30 minutes
            time_multiplier = 1.3
        elif 12.0 <= hour <= 14.0:  # Lunch hour (lower liquidity)
            time_multiplier = 1.2
        else:
            time_multiplier = 1.0

        # Total slippage with all adjustments
        total_slippage_bps = (spread_cost_bps + adjusted_impact_bps) * time_multiplier

        result = {
            'spread_cost_bps': spread_cost_bps,
            'impact_cost_bps': impact_cost_bps,
            'adjusted_impact_bps': adjusted_impact_bps,
            'vol_multiplier': vol_multiplier,
            'time_multiplier': time_multiplier,
            'total_slippage_bps': total_slippage_bps,
            'total_slippage_pct': total_slippage_bps / 10000,
            'market_regime': market_regime
        }

        logger.debug(f"📊 {symbol} Slippage: {total_slippage_bps:.1f} bps (vol_mult={vol_multiplier}, time_mult={time_multiplier:.1f})")

        return result
    
    def calculate_total_transaction_cost(
        self,
        quantity: int,
        price: float,
        commission: float,
        impact_bps: float,
        slippage_bps: float
    ) -> Dict:
        """
        Calculate total transaction cost.
        
        Total Cost = Commission + Market Impact + Slippage
        
        Returns:
            Complete cost breakdown
        """
        notional = quantity * price
        
        # Fixed costs
        commission_cost = commission
        
        # Variable costs (basis points)
        impact_cost = notional * (impact_bps / 10000)
        slippage_cost = notional * (slippage_bps / 10000)
        
        # Total
        total_cost = commission_cost + impact_cost + slippage_cost
        total_cost_bps = (total_cost / notional) * 10000
        
        result = {
            'notional': notional,
            'commission': commission_cost,
            'impact_cost': impact_cost,
            'slippage_cost': slippage_cost,
            'total_cost': total_cost,
            'total_cost_bps': total_cost_bps,
            'total_cost_pct': total_cost_bps / 10000
        }
        
        logger.info(f"💸 Total Transaction Cost: ${total_cost:.2f} ({total_cost_bps:.1f} bps)")
        logger.info(f"   Commission: ${commission_cost:.2f}")
        logger.info(f"   Impact: ${impact_cost:.2f}")
        logger.info(f"   Slippage: ${slippage_cost:.2f}")
        
        self.cost_history.append({
            'timestamp': datetime.now(),
            **result
        })
        
        return result
    
    def get_cost_statistics(self) -> Dict:
        """Get historical cost statistics."""
        if not self.cost_history:
            return {}
        
        df = pd.DataFrame(self.cost_history)
        
        stats = {
            'total_costs': df['total_cost'].sum(),
            'avg_cost_bps': df['total_cost_bps'].mean(),
            'median_cost_bps': df['total_cost_bps'].median(),
            'total_commissions': df['commission'].sum(),
            'total_impact': df['impact_cost'].sum(),
            'total_slippage': df['slippage_cost'].sum(),
            'num_trades': len(df)
        }
        
        logger.info(f"📊 Transaction Cost Statistics:")
        logger.info(f"   Total Costs: ${stats['total_costs']:,.2f}")
        logger.info(f"   Avg Cost: {stats['avg_cost_bps']:.1f} bps")
        logger.info(f"   Commissions: ${stats['total_commissions']:,.2f}")
        logger.info(f"   Market Impact: ${stats['total_impact']:,.2f}")
        logger.info(f"   Slippage: ${stats['total_slippage']:,.2f}")
        
        return stats


if __name__ == "__main__":
    # Test transaction cost optimizer
    setup_logging(level="INFO", log_file=None, json_format=False, console_output=True)
    
    optimizer = TransactionCostOptimizer()
    
    print("\n" + "="*60)
    print("TEST 1: MARKET IMPACT ESTIMATION")
    print("="*60)
    
    impact = optimizer.estimate_market_impact(
        symbol='AAPL',
        side='BUY',
        quantity=10000,
        daily_volume=50_000_000,
        market_cap=3_000_000_000_000,
        volatility=0.25
    )
    
    print("\n" + "="*60)
    print("TEST 2: EXECUTION TIMING")
    print("="*60)
    
    timing = optimizer.optimize_execution_timing(
        symbol='AAPL',
        quantity=10000,
        urgency='medium'
    )
    
    print("\n" + "="*60)
    print("TEST 3: ORDER CONSOLIDATION")
    print("="*60)
    
    pending = [
        {'symbol': 'AAPL', 'side': 'BUY', 'quantity': 50, 'price': 180},
        {'symbol': 'AAPL', 'side': 'BUY', 'quantity': 30, 'price': 181},
        {'symbol': 'AAPL', 'side': 'BUY', 'quantity': 20, 'price': 179},
        {'symbol': 'MSFT', 'side': 'SELL', 'quantity': 100, 'price': 400},
    ]
    
    consolidated = optimizer.consolidate_orders(pending)
    
    print("\n" + "="*60)
    print("TEST 4: TOTAL COST CALCULATION")
    print("="*60)
    
    total_cost = optimizer.calculate_total_transaction_cost(
        quantity=1000,
        price=180.0,
        commission=1.0,
        impact_bps=5.0,
        slippage_bps=3.0
    )
    
    print("\n✅ Transaction cost optimizer tests complete!")
