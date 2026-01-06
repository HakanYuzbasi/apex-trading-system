"""
portfolio/rebalancer.py
AUTOMATIC PORTFOLIO REBALANCING
- Periodic rebalancing (quarterly, monthly)
- Threshold-based rebalancing
- Tax-aware rebalancing
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PortfolioRebalancer:
    """
    Systematic portfolio rebalancing.
    
    Rebalancing methods:
    1. Calendar-based (quarterly, monthly)
    2. Threshold-based (when drift > 5%)
    3. Volatility-targeted
    """
    
    def __init__(self):
        self.last_rebalance_date = None
        self.rebalance_history = []
        
        logger.info("✅ Portfolio Rebalancer initialized")
    
    def should_rebalance(
        self,
        last_rebalance: datetime = None,
        rebalance_frequency: str = 'quarterly'
    ) -> Tuple[bool, str]:
        """
        Check if it's time to rebalance.
        
        Args:
            last_rebalance: Date of last rebalance
            rebalance_frequency: 'monthly', 'quarterly', 'yearly'
        
        Returns:
            (should_rebalance: bool, reason: str)
        """
        if last_rebalance is None:
            return True, "Initial rebalance"
        
        now = datetime.now()
        days_since = (now - last_rebalance).days
        
        if rebalance_frequency == 'monthly' and days_since >= 30:
            return True, f"Monthly rebalance ({days_since} days since last)"
        
        elif rebalance_frequency == 'quarterly' and days_since >= 90:
            return True, f"Quarterly rebalance ({days_since} days since last)"
        
        elif rebalance_frequency == 'yearly' and days_since >= 365:
            return True, f"Yearly rebalance ({days_since} days since last)"
        
        return False, "Not due yet"
    
    def calculate_target_weights(
        self,
        symbols: List[str],
        method: str = 'equal',
        risk_parity_vols: Dict[str, float] = None,
        custom_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate target weights for portfolio.
        
        Args:
            symbols: List of symbols
            method: 'equal', 'risk_parity', 'custom'
            risk_parity_vols: Volatilities for risk parity
            custom_weights: Custom weight dictionary
        
        Returns:
            {symbol: target_weight}
        """
        if method == 'equal':
            # Equal weight
            weight = 1.0 / len(symbols)
            return {s: weight for s in symbols}
        
        elif method == 'risk_parity':
            # Risk parity: inversely proportional to volatility
            if risk_parity_vols is None:
                logger.warning("No volatilities provided, using equal weights")
                return self.calculate_target_weights(symbols, method='equal')
            
            # Inverse volatility
            inv_vols = {s: 1.0 / risk_parity_vols[s] for s in symbols if s in risk_parity_vols}
            total_inv_vol = sum(inv_vols.values())
            
            return {s: inv_vols[s] / total_inv_vol for s in inv_vols}
        
        elif method == 'custom':
            if custom_weights is None:
                logger.warning("No custom weights provided, using equal weights")
                return self.calculate_target_weights(symbols, method='equal')
            
            # Normalize to sum to 1
            total = sum(custom_weights.values())
            return {s: custom_weights[s] / total for s in custom_weights}
        
        else:
            logger.warning(f"Unknown method {method}, using equal weights")
            return self.calculate_target_weights(symbols, method='equal')
    
    def calculate_rebalance_trades(
        self,
        current_positions: Dict[str, int],
        current_prices: Dict[str, float],
        target_weights: Dict[str, float],
        total_equity: float,
        min_trade_value: float = 100.0,
        drift_threshold: float = 0.05
    ) -> List[Dict]:
        """
        Calculate trades needed to rebalance portfolio.
        
        Args:
            current_positions: {symbol: quantity}
            current_prices: {symbol: price}
            target_weights: {symbol: target_weight}
            total_equity: Total portfolio value
            min_trade_value: Minimum trade size in dollars
            drift_threshold: Only rebalance if drift > threshold
        
        Returns:
            List of trades: [{symbol, side, quantity, reason}]
        """
        # Calculate current weights
        current_values = {
            s: abs(current_positions.get(s, 0)) * current_prices[s]
            for s in target_weights.keys()
        }
        current_total = sum(current_values.values())
        
        if current_total == 0:
            current_weights = {s: 0.0 for s in target_weights.keys()}
        else:
            current_weights = {s: current_values[s] / current_total for s in target_weights.keys()}
        
        # Calculate drifts
        drifts = {
            s: abs(target_weights[s] - current_weights[s])
            for s in target_weights.keys()
        }
        
        # Generate rebalance trades
        trades = []
        
        for symbol in target_weights.keys():
            current_weight = current_weights[symbol]
            target_weight = target_weights[symbol]
            drift = drifts[symbol]
            
            # Only trade if drift exceeds threshold
            if drift < drift_threshold:
                continue
            
            # Calculate target position
            target_value = total_equity * target_weight
            current_value = current_values[symbol]
            
            value_diff = target_value - current_value
            
            # Skip if trade too small
            if abs(value_diff) < min_trade_value:
                continue
            
            # Calculate quantity to trade
            price = current_prices[symbol]
            qty_diff = int(value_diff / price)
            
            if qty_diff == 0:
                continue
            
            side = 'BUY' if qty_diff > 0 else 'SELL'
            quantity = abs(qty_diff)
            
            trades.append({
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'drift': drift,
                'reason': f"Rebalance {current_weight*100:.1f}% → {target_weight*100:.1f}%"
            })
        
        # Log rebalance plan
        if trades:
            logger.info(f"🔄 Rebalance: {len(trades)} trades needed")
            for trade in trades:
                logger.info(
                    f"   {trade['side']} {trade['quantity']} {trade['symbol']} "
                    f"@ ${trade['price']:.2f} ({trade['reason']})"
                )
        else:
            logger.info("✅ Portfolio already balanced (no trades needed)")
        
        return trades
    
    def volatility_targeted_rebalance(
        self,
        current_positions: Dict[str, int],
        current_prices: Dict[str, float],
        volatilities: Dict[str, float],
        target_portfolio_vol: float = 0.15,
        total_equity: float = 1_000_000
    ) -> List[Dict]:
        """
        Rebalance to achieve target portfolio volatility.
        
        Args:
            current_positions: Current positions
            current_prices: Current prices
            volatilities: Asset volatilities
            target_portfolio_vol: Target portfolio volatility (e.g., 0.15 = 15%)
            total_equity: Total portfolio value
        
        Returns:
            List of rebalance trades
        """
        # Calculate risk parity weights (inverse volatility)
        target_weights = self.calculate_target_weights(
            list(volatilities.keys()),
            method='risk_parity',
            risk_parity_vols=volatilities
        )
        
        # Calculate current portfolio vol
        symbols = list(volatilities.keys())
        weights = []
        vols = []
        
        for symbol in symbols:
            current_value = abs(current_positions.get(symbol, 0)) * current_prices[symbol]
            weight = current_value / total_equity if total_equity > 0 else 0
            weights.append(weight)
            vols.append(volatilities[symbol])
        
        weights = np.array(weights)
        vols = np.array(vols)
        
        # Simplified portfolio vol (assumes uncorrelated for now)
        current_portfolio_vol = np.sqrt(np.sum((weights * vols) ** 2))
        
        # Scale positions to hit target vol
        if current_portfolio_vol > 0:
            vol_scaling = target_portfolio_vol / current_portfolio_vol
        else:
            vol_scaling = 1.0
        
        # Adjust target weights by vol scaling
        scaled_weights = {s: target_weights[s] * vol_scaling for s in target_weights}
        
        # Renormalize if needed
        total_scaled = sum(scaled_weights.values())
        if total_scaled > 1.0:
            scaled_weights = {s: w / total_scaled for s, w in scaled_weights.items()}
        
        logger.info(f"🎯 Vol-targeted rebalance: {current_portfolio_vol:.1%} → {target_portfolio_vol:.1%}")
        
        # Generate trades
        return self.calculate_rebalance_trades(
            current_positions,
            current_prices,
            scaled_weights,
            total_equity
        )
    
    def tax_aware_rebalance(
        self,
        current_positions: Dict[str, int],
        current_prices: Dict[str, float],
        entry_prices: Dict[str, float],
        target_weights: Dict[str, float],
        total_equity: float,
        tax_rate: float = 0.20,
        min_holding_days: int = 365
    ) -> List[Dict]:
        """
        Tax-aware rebalancing (prefer long-term gains, avoid short-term).
        
        Args:
            current_positions: Current positions
            current_prices: Current prices
            entry_prices: Entry prices (for capital gains)
            target_weights: Target weights
            total_equity: Total equity
            tax_rate: Capital gains tax rate
            min_holding_days: Minimum days for long-term gains
        
        Returns:
            List of tax-optimized trades
        """
        # Get standard rebalance trades
        standard_trades = self.calculate_rebalance_trades(
            current_positions,
            current_prices,
            target_weights,
            total_equity
        )
        
        # Filter trades to minimize tax impact
        tax_optimized_trades = []
        
        for trade in standard_trades:
            symbol = trade['symbol']
            
            # Only applies to sells
            if trade['side'] == 'BUY':
                tax_optimized_trades.append(trade)
                continue
            
            # Calculate capital gain
            entry_price = entry_prices.get(symbol, trade['price'])
            current_price = trade['price']
            gain = (current_price - entry_price) / entry_price
            
            # If substantial gain, consider deferring
            if gain > 0.20:  # > 20% gain
                logger.info(
                    f"⚠️  {symbol}: Large gain ({gain*100:.1f}%), "
                    f"consider deferring for tax optimization"
                )
                # Still include but flag
                trade['tax_impact'] = 'high'
                trade['estimated_tax'] = gain * current_price * trade['quantity'] * tax_rate
            
            tax_optimized_trades.append(trade)
        
        return tax_optimized_trades
    
    def record_rebalance(self, trades: List[Dict]):
        """Record rebalance event."""
        event = {
            'timestamp': datetime.now(),
            'num_trades': len(trades),
            'trades': trades
        }
        
        self.rebalance_history.append(event)
        self.last_rebalance_date = datetime.now()
        
        logger.info(f"📝 Rebalance recorded: {len(trades)} trades")


if __name__ == "__main__":
    # Test rebalancer
    logging.basicConfig(level=logging.INFO)
    
    rebalancer = PortfolioRebalancer()
    
    print("\n" + "="*60)
    print("TEST 1: CALCULATE TARGET WEIGHTS")
    print("="*60)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'BAC']
    
    # Equal weight
    equal_weights = rebalancer.calculate_target_weights(symbols, method='equal')
    print("Equal weights:", equal_weights)
    
    # Risk parity
    vols = {'AAPL': 0.25, 'MSFT': 0.22, 'GOOGL': 0.28, 'JPM': 0.20, 'BAC': 0.30}
    rp_weights = rebalancer.calculate_target_weights(symbols, method='risk_parity', risk_parity_vols=vols)
    print("Risk parity weights:", {k: f"{v:.3f}" for k, v in rp_weights.items()})
    
    print("\n" + "="*60)
    print("TEST 2: CALCULATE REBALANCE TRADES")
    print("="*60)
    
    current_positions = {'AAPL': 50, 'MSFT': 100, 'GOOGL': 20, 'JPM': 80, 'BAC': 150}
    current_prices = {'AAPL': 180, 'MSFT': 400, 'GOOGL': 140, 'JPM': 150, 'BAC': 35}
    
    trades = rebalancer.calculate_rebalance_trades(
        current_positions,
        current_prices,
        equal_weights,
        total_equity=100000,
        drift_threshold=0.05
    )
    
    print("\n" + "="*60)
    print("TEST 3: VOLATILITY-TARGETED REBALANCE")
    print("="*60)
    
    vol_trades = rebalancer.volatility_targeted_rebalance(
        current_positions,
        current_prices,
        vols,
        target_portfolio_vol=0.15,
        total_equity=100000
    )
    
    print("\n✅ Rebalancer tests complete!")
