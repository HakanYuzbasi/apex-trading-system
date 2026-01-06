"""
market/market_regime_detector.py
DETECT MARKET REGIMES AND ADAPT STRATEGY
- Bull, Bear, Sideways, Crisis
- Dynamic parameter adjustment
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detect market regime and adjust trading parameters.
    
    Regimes:
    - BULL: Trending up, low volatility
    - BEAR: Trending down, high volatility
    - SIDEWAYS: Range-bound, mean reverting
    - CRISIS: Extreme volatility, high correlation
    """
    
    def __init__(self):
        self.current_regime = 'UNKNOWN'
        self.regime_history = []
        self.regime_confidence = 0.0
        
        logger.info("✅ Market Regime Detector initialized")
    
    def detect_regime(
        self,
        returns: pd.Series,
        volatility: pd.Series = None,
        lookback: int = 60
    ) -> Dict:
        """
        Detect current market regime.
        
        Args:
            returns: Daily returns series
            volatility: Volatility series (optional, will calculate if not provided)
            lookback: Lookback period for analysis
        
        Returns:
            {
                'regime': str,
                'confidence': float,
                'metrics': dict,
                'params': dict
            }
        """
        # Calculate volatility if not provided
        if volatility is None:
            volatility = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        
        # Get recent data
        recent_returns = returns.tail(lookback)
        recent_vol = volatility.tail(lookback)
        
        # Calculate key metrics
        avg_return = recent_returns.mean() * 252  # Annualized
        avg_vol = recent_vol.mean()
        trend_strength = abs(stats.linregress(range(len(recent_returns)), recent_returns).slope)
        skewness = stats.skew(recent_returns)
        
        # Calculate regime indicators
        is_bull = avg_return > 0.10 and avg_vol < 0.20 and skewness > -0.5
        is_bear = avg_return < -0.05 and avg_vol > 0.20
        is_crisis = avg_vol > 0.35
        is_sideways = abs(avg_return) < 0.05 and avg_vol < 0.20
        
        # Determine regime with confidence
        regime_scores = {
            'BULL': 0.0,
            'BEAR': 0.0,
            'SIDEWAYS': 0.0,
            'CRISIS': 0.0
        }
        
        # BULL scoring
        if avg_return > 0:
            regime_scores['BULL'] += min(avg_return / 0.30, 1.0) * 0.4  # Up to 40% weight
        if avg_vol < 0.20:
            regime_scores['BULL'] += (1 - avg_vol / 0.20) * 0.3  # Up to 30% weight
        if skewness > 0:
            regime_scores['BULL'] += min(skewness / 1.0, 1.0) * 0.3  # Up to 30% weight
        
        # BEAR scoring
        if avg_return < 0:
            regime_scores['BEAR'] += min(abs(avg_return) / 0.30, 1.0) * 0.4
        if avg_vol > 0.20:
            regime_scores['BEAR'] += min((avg_vol - 0.20) / 0.30, 1.0) * 0.3
        if skewness < 0:
            regime_scores['BEAR'] += min(abs(skewness) / 1.0, 1.0) * 0.3
        
        # CRISIS scoring
        if avg_vol > 0.35:
            regime_scores['CRISIS'] = min((avg_vol - 0.35) / 0.50, 1.0)
        
        # SIDEWAYS scoring
        if abs(avg_return) < 0.10:
            regime_scores['SIDEWAYS'] += (1 - abs(avg_return) / 0.10) * 0.4
        if avg_vol < 0.20:
            regime_scores['SIDEWAYS'] += (1 - avg_vol / 0.20) * 0.3
        if trend_strength < 0.0001:
            regime_scores['SIDEWAYS'] += 0.3
        
        # Get regime with highest score
        regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[regime]
        
        # Get parameters for this regime
        params = self.get_regime_parameters(regime)
        
        # Log regime detection
        if regime != self.current_regime:
            logger.info(f"📊 Regime Change: {self.current_regime} → {regime} (confidence: {confidence:.2f})")
            logger.info(f"   Metrics: Return={avg_return:.1%}, Vol={avg_vol:.1%}, Skew={skewness:.2f}")
        
        self.current_regime = regime
        self.regime_confidence = confidence
        
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'confidence': confidence,
            'metrics': {
                'avg_return': avg_return,
                'avg_volatility': avg_vol,
                'skewness': skewness,
                'trend_strength': trend_strength
            }
        })
        
        return {
            'regime': regime,
            'confidence': confidence,
            'metrics': {
                'avg_return': avg_return,
                'avg_volatility': avg_vol,
                'skewness': skewness,
                'trend_strength': trend_strength
            },
            'regime_scores': regime_scores,
            'params': params
        }
    
    def get_regime_parameters(self, regime: str) -> Dict:
        """
        Get trading parameters optimized for each regime.
        
        Returns:
            Dictionary of trading parameters
        """
        params = {
            'BULL': {
                'position_size_mult': 1.3,          # Larger positions
                'min_signal_threshold': 0.30,       # Lower threshold (more trades)
                'take_profit_pct': 20.0,            # Let winners run
                'stop_loss_pct': -8.0,              # Wider stops
                'max_holding_days': 60,             # Hold longer
                'max_positions': 20,                # More diversification
                'trade_cooldown': 180,              # 3 minutes
                'adaptive_sizing': True,
                'allow_shorts': False               # Long-only in bull
            },
            'BEAR': {
                'position_size_mult': 0.5,          # Smaller positions
                'min_signal_threshold': 0.60,       # Much higher threshold (fewer trades)
                'take_profit_pct': 8.0,             # Quick profits
                'stop_loss_pct': -3.0,              # Tight stops
                'max_holding_days': 15,             # Exit fast
                'max_positions': 10,                # Less diversification
                'trade_cooldown': 600,              # 10 minutes
                'adaptive_sizing': True,
                'allow_shorts': True                # Shorts allowed
            },
            'SIDEWAYS': {
                'position_size_mult': 1.0,          # Normal size
                'min_signal_threshold': 0.45,       # Standard threshold
                'take_profit_pct': 5.0,             # Quick scalps
                'stop_loss_pct': -4.0,              # Medium stops
                'max_holding_days': 10,             # Short hold
                'max_positions': 15,                # Medium diversification
                'trade_cooldown': 300,              # 5 minutes
                'adaptive_sizing': True,
                'allow_shorts': True                # Both directions
            },
            'CRISIS': {
                'position_size_mult': 0.2,          # Tiny positions!
                'min_signal_threshold': 0.80,       # Only strongest signals
                'take_profit_pct': 5.0,             # Take any profit
                'stop_loss_pct': -2.0,              # Very tight stops
                'max_holding_days': 3,              # Exit very fast
                'max_positions': 5,                 # Minimal exposure
                'trade_cooldown': 900,              # 15 minutes
                'adaptive_sizing': True,
                'allow_shorts': False               # Stay on sidelines
            },
            'UNKNOWN': {
                'position_size_mult': 0.7,
                'min_signal_threshold': 0.50,
                'take_profit_pct': 10.0,
                'stop_loss_pct': -5.0,
                'max_holding_days': 20,
                'max_positions': 10,
                'trade_cooldown': 300,
                'adaptive_sizing': True,
                'allow_shorts': False
            }
        }
        
        return params.get(regime, params['UNKNOWN'])
    
    def should_trade(self, regime: str = None) -> Tuple[bool, str]:
        """
        Determine if we should trade in current regime.
        
        Returns:
            (should_trade: bool, reason: str)
        """
        regime = regime or self.current_regime
        
        if regime == 'CRISIS':
            if self.regime_confidence > 0.7:
                return False, "Crisis regime - staying on sidelines"
        
        if regime == 'UNKNOWN':
            return False, "Regime unclear - waiting for clarity"
        
        return True, f"{regime} regime - trading enabled"
    
    def get_historical_regime_stats(self) -> pd.DataFrame:
        """Get statistics about historical regime changes."""
        if not self.regime_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.regime_history)
        
        # Calculate regime durations
        regime_durations = {}
        for regime in df['regime'].unique():
            regime_df = df[df['regime'] == regime]
            if len(regime_df) > 1:
                duration = (regime_df['timestamp'].iloc[-1] - regime_df['timestamp'].iloc[0]).total_seconds() / 3600
                regime_durations[regime] = duration
        
        logger.info(f"\n📊 Historical Regime Stats:")
        for regime, duration in regime_durations.items():
            logger.info(f"   {regime:15s}: {duration:.1f} hours")
        
        return df


if __name__ == "__main__":
    # Test market regime detector
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    
    # Simulate different regimes
    regime_periods = [
        ('BULL', 0, 300, 0.0005, 0.01),      # Bull market
        ('SIDEWAYS', 300, 500, 0.0, 0.015),   # Sideways
        ('BEAR', 500, 700, -0.0003, 0.025),  # Bear market
        ('BULL', 700, 900, 0.0004, 0.012),   # Recovery
        ('CRISIS', 900, 950, -0.001, 0.05),  # Crisis
        ('BULL', 950, len(dates), 0.0005, 0.01)  # Recovery
    ]
    
    returns = []
    for regime, start, end, mu, sigma in regime_periods:
        period_returns = np.random.normal(mu, sigma, end - start)
        returns.extend(period_returns)
    
    returns_series = pd.Series(returns, index=dates)
    
    # Test detector
    detector = MarketRegimeDetector()
    
    print("\n" + "="*60)
    print("TESTING REGIME DETECTION")
    print("="*60)
    
    # Test at different points
    test_points = [150, 400, 600, 800, 925, 1000]
    
    for point in test_points:
        result = detector.detect_regime(
            returns_series[:point],
            lookback=60
        )
        
        print(f"\nDay {point}:")
        print(f"  Regime: {result['regime']} (confidence: {result['confidence']:.2f})")
        print(f"  Return: {result['metrics']['avg_return']:.2%}")
        print(f"  Volatility: {result['metrics']['avg_volatility']:.2%}")
        print(f"  Parameters:")
        for key, value in result['params'].items():
            print(f"    {key}: {value}")
    
    # Get stats
    stats_df = detector.get_historical_regime_stats()
    
    print("\n✅ Regime detector tests complete!")
