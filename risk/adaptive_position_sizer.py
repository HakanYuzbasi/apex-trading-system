"""
risk/adaptive_position_sizer.py
DYNAMIC POSITION SIZING BASED ON MARKET CONDITIONS
- Kelly Criterion
- Volatility scaling
- Performance-based adjustment
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class AdaptivePositionSizer:
    """
    Dynamically adjust position size based on:
    - Market volatility
    - Strategy performance (Sharpe ratio)
    - Win rate
    - Signal confidence
    - Current drawdown
    - Market regime
    """
    
    def __init__(self, base_position_size: float = 5000):
        self.base_position_size = base_position_size
        self.volatility_history = []
        
        logger.info(f"✅ Adaptive Position Sizer initialized (base: ${base_position_size:,.0f})")
    
    def calculate_position_size(
        self,
        signal_confidence: float,
        volatility: float,
        sharpe_ratio: float = 0.0,
        win_rate: float = 0.5,
        current_drawdown: float = 0.0,
        regime_multiplier: float = 1.0,
        portfolio_value: float = 1_000_000,
        max_position_pct: float = 0.02
    ) -> Dict:
        """
        Calculate optimal position size.
        
        Args:
            signal_confidence: 0 to 1
            volatility: Current volatility (annualized)
            sharpe_ratio: Strategy Sharpe ratio
            win_rate: Historical win rate (0 to 1)
            current_drawdown: Current drawdown (0 to 1)
            regime_multiplier: From market regime detector
            portfolio_value: Total portfolio value
            max_position_pct: Max % of portfolio per trade
        
        Returns:
            {
                'position_size': float,
                'multiplier': float,
                'components': dict
            }
        """
        multiplier = 1.0
        components = {}
        
        # 1. Signal Confidence Adjustment
        confidence_mult = 0.5 + (signal_confidence * 0.5)  # 0.5x to 1.0x
        multiplier *= confidence_mult
        components['confidence'] = confidence_mult
        
        # 2. Volatility Adjustment (inverse relationship)
        # High vol = smaller size, Low vol = larger size
        vol_percentile = self._get_volatility_percentile(volatility)
        vol_mult = 2.0 - vol_percentile  # 0.5x to 1.5x
        multiplier *= vol_mult
        components['volatility'] = vol_mult
        
        # 3. Sharpe Ratio Adjustment
        if sharpe_ratio > 1.5:
            sharpe_mult = 1.3  # Strategy performing great
        elif sharpe_ratio > 1.0:
            sharpe_mult = 1.15
        elif sharpe_ratio > 0.5:
            sharpe_mult = 1.0
        elif sharpe_ratio > 0:
            sharpe_mult = 0.85
        else:
            sharpe_mult = 0.6  # Strategy struggling
        
        multiplier *= sharpe_mult
        components['sharpe'] = sharpe_mult
        
        # 4. Win Rate Adjustment
        win_rate_mult = 0.5 + win_rate  # 0.5x to 1.5x
        multiplier *= win_rate_mult
        components['win_rate'] = win_rate_mult
        
        # 5. Drawdown Protection
        if current_drawdown > 0.10:  # > 10% drawdown
            dd_mult = 0.3  # Reduce to 30%
        elif current_drawdown > 0.08:
            dd_mult = 0.5  # Reduce to 50%
        elif current_drawdown > 0.05:
            dd_mult = 0.7  # Reduce to 70%
        else:
            dd_mult = 1.0  # No reduction
        
        multiplier *= dd_mult
        components['drawdown'] = dd_mult
        
        # 6. Regime Adjustment
        multiplier *= regime_multiplier
        components['regime'] = regime_multiplier
        
        # 7. Apply caps
        multiplier = np.clip(multiplier, 0.2, 2.5)  # 20% to 250% of base
        
        # Calculate final position size
        position_size = self.base_position_size * multiplier
        
        # Apply portfolio limit
        max_position_size = portfolio_value * max_position_pct
        if position_size > max_position_size:
            position_size = max_position_size
            logger.debug(f"Position capped at {max_position_pct*100:.1f}% of portfolio")
        
        logger.debug(f"💰 Position Size: ${self.base_position_size:,.0f} → ${position_size:,.0f} ({multiplier:.2f}x)")
        
        return {
            'position_size': position_size,
            'multiplier': multiplier,
            'components': components,
            'base_size': self.base_position_size
        }
    
    def _get_volatility_percentile(self, current_vol: float) -> float:
        """
        Get percentile rank of current volatility.
        
        Returns value between 0 and 1:
        - 0 = Lowest volatility ever seen
        - 1 = Highest volatility ever seen
        """
        self.volatility_history.append(current_vol)
        
        # Keep only last 252 trading days (1 year)
        if len(self.volatility_history) > 252:
            self.volatility_history = self.volatility_history[-252:]
        
        if len(self.volatility_history) < 10:
            return 0.5  # Not enough data
        
        # Calculate percentile
        sorted_vol = sorted(self.volatility_history)
        rank = sorted_vol.index(min(sorted_vol, key=lambda x: abs(x - current_vol)))
        percentile = rank / len(sorted_vol)
        
        return percentile
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = W - [(1-W)/R]
        where:
        - W = win rate
        - R = avg_win / avg_loss ratio
        
        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
            kelly_fraction: Fraction of full Kelly (0.25 = quarter Kelly)
        
        Returns:
            Recommended position size as fraction of capital
        """
        if avg_loss == 0 or win_rate == 0:
            return 0.0
        
        # Win/Loss ratio
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fraction (quarter Kelly is safer)
        kelly_pct *= kelly_fraction
        
        # Clip to reasonable range
        kelly_pct = np.clip(kelly_pct, 0.0, 0.05)  # Max 5% of capital
        
        return kelly_pct


if __name__ == "__main__":
    # Test adaptive position sizer
    logging.basicConfig(level=logging.DEBUG)
    
    sizer = AdaptivePositionSizer(base_position_size=5000)
    
    print("\n" + "="*60)
    print("TESTING ADAPTIVE POSITION SIZING")
    print("="*60)
    
    scenarios = [
        {
            'name': 'Perfect Conditions',
            'confidence': 0.9,
            'volatility': 0.12,
            'sharpe': 2.0,
            'win_rate': 0.65,
            'drawdown': 0.02,
            'regime_mult': 1.3
        },
        {
            'name': 'High Volatility',
            'confidence': 0.7,
            'volatility': 0.35,
            'sharpe': 1.0,
            'win_rate': 0.55,
            'drawdown': 0.05,
            'regime_mult': 1.0
        },
        {
            'name': 'Deep Drawdown',
            'confidence': 0.6,
            'volatility': 0.20,
            'sharpe': 0.5,
            'win_rate': 0.45,
            'drawdown': 0.12,
            'regime_mult': 0.7
        },
        {
            'name': 'Crisis Mode',
            'confidence': 0.8,
            'volatility': 0.50,
            'sharpe': -0.5,
            'win_rate': 0.40,
            'drawdown': 0.15,
            'regime_mult': 0.2
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        result = sizer.calculate_position_size(
            signal_confidence=scenario['confidence'],
            volatility=scenario['volatility'],
            sharpe_ratio=scenario['sharpe'],
            win_rate=scenario['win_rate'],
            current_drawdown=scenario['drawdown'],
            regime_multiplier=scenario['regime_mult']
        )
        
        print(f"  Position Size: ${result['position_size']:,.0f} ({result['multiplier']:.2f}x)")
        print(f"  Components:")
        for key, value in result['components'].items():
            print(f"    {key:12s}: {value:.2f}x")
    
    # Test Kelly Criterion
    print("\n" + "="*60)
    print("TESTING KELLY CRITERION")
    print("="*60)
    
    kelly_pct = sizer.kelly_criterion(
        win_rate=0.60,
        avg_win=100,
        avg_loss=50,
        kelly_fraction=0.25
    )
    
    print(f"Kelly Position Size: {kelly_pct*100:.2f}% of capital")
    
    print("\n✅ Adaptive position sizer tests complete!")
