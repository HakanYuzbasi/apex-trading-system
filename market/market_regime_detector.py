"""
market/market_regime_detector.py
================================================================================
APEX TRADING SYSTEM - MARKET REGIME DETECTION
================================================================================

Sophisticated market regime detection with:
- Bull/Bear/Sideways/Crisis classification
- Volatility-based regime identification
- Confidence scoring
- Regime-specific trading parameters
- Trend analysis and classification

The system adapts trading parameters based on detected regime:
- BULL: Aggressive trading, long bias, high position sizing
- BEAR: Conservative trading, short allowed, reduced position sizes
- SIDEWAYS: Mean reversion focus, neutral bias
- CRISIS: Extreme risk management, minimal positions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detect and classify market regimes in real-time.
    
    Regimes:
    1. BULL - Rising trend, low volatility, positive sentiment
       - Return > 5%, Vol < Historical * 1.2
    
    2. BEAR - Falling trend, high volatility, negative sentiment
       - Return < -5% OR Vol > Historical * 1.5
    
    3. SIDEWAYS - Range-bound, normal volatility
       - No strong trend, normal volatility
    
    4. CRISIS - Extreme volatility spike, crash conditions
       - Vol > Historical * 3.0
    
    Usage:
    ```python
    detector = MarketRegimeDetector()
    result = detector.detect_regime(returns_series, lookback=60)
    
    print(result['regime'])  # 'BULL', 'BEAR', etc
    print(result['confidence'])  # 0.85
    params = result['params']  # Regime-specific trading parameters
    ```
    """
    
    def __init__(self):
        """Initialize the detector."""
        self.regime_history = []
        self.volatility_history = []
        
        logger.info("✅ Market Regime Detector initialized")
    
    def detect_regime(
        self,
        returns: pd.Series,
        lookback: int = 60,
        crisis_vol_threshold: float = 3.0,
        bull_return_threshold: float = 0.05,
        bear_return_threshold: float = -0.05,
        bear_vol_multiplier: float = 1.5,
        bull_vol_multiplier: float = 1.2
    ) -> Dict:
        """
        Detect current market regime from returns.
        
        Args:
            returns: Daily returns series (e.g., daily % changes)
            lookback: Number of days to analyze (default: 60 = ~3 months)
            crisis_vol_threshold: Crisis threshold in std devs (default: 3.0)
            bull_return_threshold: Minimum return for bull (default: 5%)
            bear_return_threshold: Maximum return for bear (default: -5%)
            bear_vol_multiplier: Vol multiplier for bear detection (default: 1.5)
            bull_vol_multiplier: Vol multiplier for bull detection (default: 1.2)
        
        Returns:
            {
                'regime': str ('BULL', 'BEAR', 'SIDEWAYS', 'CRISIS'),
                'confidence': float (0-1),
                'volatility': float (annualized),
                'trend': float (-1 to 1),
                'vol_ratio': float (current vol / historical vol),
                'cumulative_return': float,
                'params': dict (regime-specific parameters),
                'timestamp': str (ISO format)
            }
        """
        
        # Validation
        if len(returns) < lookback:
            logger.warning(f"Insufficient data: {len(returns)} < {lookback}")
            return self._default_regime()
        
        # Use recent window
        recent_returns = returns.tail(lookback).copy()
        
        # ═══════════════════════════════════════════════════════════════
        # CALCULATE METRICS
        # ═══════════════════════════════════════════════════════════════
        
        # 1. Cumulative return over lookback period
        cumulative_return = (1 + recent_returns).prod() - 1
        
        # 2. Current volatility (annualized)
        volatility = recent_returns.std() * np.sqrt(252)
        
        # 3. Historical volatility (baseline)
        if len(returns) > 252:
            historical_returns = returns.tail(252)
        else:
            historical_returns = returns
        
        historical_volatility = historical_returns.std() * np.sqrt(252)
        
        # Handle zero volatility
        if historical_volatility == 0:
            historical_volatility = 0.01
        if volatility == 0:
            volatility = 0.01
        
        # 4. Volatility ratio (current vs historical)
        vol_ratio = volatility / historical_volatility
        
        # 5. Trend strength (using moving averages)
        ma_short = recent_returns.rolling(10).mean().iloc[-1]
        ma_long = recent_returns.rolling(30).mean().iloc[-1]
        trend_diff = ma_short - ma_long
        
        # Normalize trend to [-1, 1]
        if volatility > 0:
            trend_normalized = np.clip(trend_diff / (volatility / 10), -1, 1)
        else:
            trend_normalized = 0
        
        # 6. Downside capture (consecutive negative returns)
        negative_returns = (recent_returns < 0).sum()
        downside_ratio = negative_returns / len(recent_returns)
        
        # ═══════════════════════════════════════════════════════════════
        # DETECT REGIME
        # ═══════════════════════════════════════════════════════════════
        
        regime = 'UNKNOWN'
        confidence = 0.0
        regime_reason = ""
        
        # Priority 1: CRISIS (highest priority)
        if vol_ratio > crisis_vol_threshold:
            regime = 'CRISIS'
            confidence = min((vol_ratio / (crisis_vol_threshold * 1.5)), 1.0)
            regime_reason = f"Extreme volatility spike: {vol_ratio:.2f}x historical"
        
        # Priority 2: BEAR (strong downtrend or high volatility + downside)
        elif (cumulative_return < bear_return_threshold or 
              (vol_ratio > bear_vol_multiplier and downside_ratio > 0.45)):
            regime = 'BEAR'
            confidence = min(
                max(
                    abs(cumulative_return) / abs(bear_return_threshold),
                    (vol_ratio - bear_vol_multiplier) / bear_vol_multiplier
                ),
                1.0
            )
            regime_reason = f"Downtrend/High Vol: Return={cumulative_return:.2%}, Vol_ratio={vol_ratio:.2f}x"
        
        # Priority 3: BULL (strong uptrend and controlled volatility)
        elif (cumulative_return > bull_return_threshold and 
              vol_ratio < bull_vol_multiplier):
            regime = 'BULL'
            confidence = min(
                cumulative_return / (bull_return_threshold * 1.5),
                1.0
            )
            regime_reason = f"Uptrend/Low Vol: Return={cumulative_return:.2%}, Vol_ratio={vol_ratio:.2f}x"
        
        # Priority 4: SIDEWAYS (default)
        else:
            regime = 'SIDEWAYS'
            confidence = max(0.5, 1.0 - abs(trend_normalized) * 0.5)
            regime_reason = f"Range-bound: Return={cumulative_return:.2%}, Vol_ratio={vol_ratio:.2f}x"
        
        # Get regime-specific parameters
        params = self._get_regime_params(regime, volatility, trend_normalized)
        
        # Store in history
        regime_data = {
            'timestamp': datetime.now(),
            'regime': regime,
            'confidence': confidence,
            'volatility': volatility,
            'vol_ratio': vol_ratio
        }
        self.regime_history.append(regime_data)
        self.volatility_history.append(volatility)
        
        # Keep only recent history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-500:]
            self.volatility_history = self.volatility_history[-500:]
        
        # Log results
        logger.info(f"\n📊 MARKET REGIME: {regime}")
        logger.info(f"   Confidence: {confidence:.2%}")
        logger.info(f"   {regime_reason}")
        logger.info(f"   Volatility: {volatility*100:.2f}% (historical: {historical_volatility*100:.2f}%)")
        logger.info(f"   Return ({lookback}d): {cumulative_return*100:+.2f}%")
        logger.info(f"   Downside ratio: {downside_ratio:.2%}")
        
        return {
            'regime': regime,
            'confidence': float(confidence),
            'volatility': float(volatility),
            'historical_volatility': float(historical_volatility),
            'trend': float(trend_normalized),
            'vol_ratio': float(vol_ratio),
            'cumulative_return': float(cumulative_return),
            'downside_ratio': float(downside_ratio),
            'params': params,
            'reason': regime_reason,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_regime_params(
        self,
        regime: str,
        volatility: float,
        trend: float
    ) -> Dict:
        """
        Get trading parameters optimized for the detected regime.
        
        Each regime has different optimal parameters based on market conditions.
        """
        
        if regime == 'BULL':
            # Bullish market: Aggressive, long bias, high position sizing
            return {
                'regime_name': 'BULL',
                'description': 'Uptrend with low volatility - Aggressive trading',
                'max_positions': 25,
                'position_size_mult': 1.3,
                'min_signal_threshold': 0.40,
                'take_profit_pct': 15.0,
                'stop_loss_pct': -6.0,
                'allow_shorts': False,
                'short_bias': 0.0,
                'trade_cooldown_seconds': 300,
                'max_sector_exposure': 0.50,
                'rebalance_frequency': 'weekly',
                'trading_hours_est': (9.5, 16.0),
                'max_daily_loss_pct': 3.0,
                'max_drawdown_pct': 15.0,
                'kelly_fraction': 0.25,
                'volatility_target': 0.18
            }
        
        elif regime == 'BEAR':
            # Bearish market: Conservative, allow shorts, reduced positions
            return {
                'regime_name': 'BEAR',
                'description': 'Downtrend or high volatility - Conservative trading',
                'max_positions': 12,
                'position_size_mult': 0.65,
                'min_signal_threshold': 0.55,
                'take_profit_pct': 8.0,
                'stop_loss_pct': -3.0,
                'allow_shorts': True,
                'short_bias': 0.3,
                'trade_cooldown_seconds': 600,
                'max_sector_exposure': 0.35,
                'rebalance_frequency': 'biweekly',
                'trading_hours_est': (10.0, 15.0),
                'max_daily_loss_pct': 2.0,
                'max_drawdown_pct': 10.0,
                'kelly_fraction': 0.15,
                'volatility_target': 0.12
            }
        
        elif regime == 'SIDEWAYS':
            # Sideways market: Mean reversion, neutral
            return {
                'regime_name': 'SIDEWAYS',
                'description': 'Range-bound market - Mean reversion focus',
                'max_positions': 18,
                'position_size_mult': 0.85,
                'min_signal_threshold': 0.48,
                'take_profit_pct': 5.0,
                'stop_loss_pct': -4.0,
                'allow_shorts': True,
                'short_bias': 0.1,
                'trade_cooldown_seconds': 450,
                'max_sector_exposure': 0.42,
                'rebalance_frequency': 'weekly',
                'trading_hours_est': (9.5, 16.0),
                'max_daily_loss_pct': 2.5,
                'max_drawdown_pct': 12.0,
                'kelly_fraction': 0.20,
                'volatility_target': 0.15
            }
        
        elif regime == 'CRISIS':
            # Crisis: Extreme risk management, minimal positions
            return {
                'regime_name': 'CRISIS',
                'description': 'Extreme volatility - Defensive trading',
                'max_positions': 5,
                'position_size_mult': 0.25,
                'min_signal_threshold': 0.70,
                'take_profit_pct': 3.0,
                'stop_loss_pct': -2.0,
                'allow_shorts': True,
                'short_bias': 0.5,
                'trade_cooldown_seconds': 900,
                'max_sector_exposure': 0.20,
                'rebalance_frequency': 'daily',
                'trading_hours_est': (10.0, 14.0),
                'max_daily_loss_pct': 1.0,
                'max_drawdown_pct': 5.0,
                'kelly_fraction': 0.05,
                'volatility_target': 0.08
            }
        
        else:
            # Unknown/Default
            return self._get_default_params()
    
    def _get_default_params(self) -> Dict:
        """Get conservative default parameters."""
        return {
            'regime_name': 'DEFAULT',
            'description': 'Unknown regime - Conservative defaults',
            'max_positions': 15,
            'position_size_mult': 1.0,
            'min_signal_threshold': 0.45,
            'take_profit_pct': 10.0,
            'stop_loss_pct': -5.0,
            'allow_shorts': False,
            'short_bias': 0.0,
            'trade_cooldown_seconds': 300,
            'max_sector_exposure': 0.40,
            'rebalance_frequency': 'weekly',
            'trading_hours_est': (9.5, 16.0),
            'max_daily_loss_pct': 2.5,
            'max_drawdown_pct': 12.0,
            'kelly_fraction': 0.20,
            'volatility_target': 0.15
        }
    
    def _default_regime(self) -> Dict:
        """Return default regime when detection fails."""
        return {
            'regime': 'UNKNOWN',
            'confidence': 0.0,
            'volatility': 0.0,
            'historical_volatility': 0.0,
            'trend': 0.0,
            'vol_ratio': 1.0,
            'cumulative_return': 0.0,
            'downside_ratio': 0.5,
            'params': self._get_default_params(),
            'reason': 'Insufficient data',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_regime_history(self, periods: int = 100) -> pd.DataFrame:
        """Get recent regime history as DataFrame."""
        if not self.regime_history:
            return pd.DataFrame()
        
        recent = self.regime_history[-periods:]
        
        return pd.DataFrame({
            'timestamp': [r['timestamp'] for r in recent],
            'regime': [r['regime'] for r in recent],
            'confidence': [r['confidence'] for r in recent],
            'volatility': [r['volatility'] for r in recent],
            'vol_ratio': [r['vol_ratio'] for r in recent]
        })
    
    def get_regime_distribution(self) -> Dict[str, int]:
        """Get count of each regime in history."""
        if not self.regime_history:
            return {}
        
        distribution = {}
        for regime_data in self.regime_history:
            regime = regime_data['regime']
            distribution[regime] = distribution.get(regime, 0) + 1
        
        return distribution
    
    def get_average_volatility_by_regime(self) -> Dict[str, float]:
        """Get average volatility for each regime."""
        if not self.regime_history:
            return {}
        
        regime_vols = {}
        regime_counts = {}
        
        for regime_data in self.regime_history:
            regime = regime_data['regime']
            vol = regime_data['volatility']
            
            if regime not in regime_vols:
                regime_vols[regime] = 0
                regime_counts[regime] = 0
            
            regime_vols[regime] += vol
            regime_counts[regime] += 1
        
        return {
            regime: regime_vols[regime] / regime_counts[regime]
            for regime in regime_vols
        }


if __name__ == "__main__":
    # Test market regime detector
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("MARKET REGIME DETECTOR - TEST")
    print("="*80 + "\n")
    
    # Generate sample returns
    np.random.seed(42)
    
    # Bull market
    print("TEST 1: BULL MARKET")
    bull_returns = pd.Series(np.random.randn(100) * 0.008 + 0.001)  # +0.1% daily
    
    detector = MarketRegimeDetector()
    bull_result = detector.detect_regime(bull_returns)
    
    print(f"Detected: {bull_result['regime']}")
    print(f"Confidence: {bull_result['confidence']:.2%}")
    assert bull_result['regime'] == 'BULL', "Should detect BULL regime"
    print("✅ PASSED\n")
    
    # Bear market
    print("TEST 2: BEAR MARKET")
    bear_returns = pd.Series(np.random.randn(100) * 0.015 - 0.002)  # -0.2% daily
    
    bear_result = detector.detect_regime(bear_returns)
    
    print(f"Detected: {bear_result['regime']}")
    print(f"Confidence: {bear_result['confidence']:.2%}")
    assert bear_result['regime'] == 'BEAR', "Should detect BEAR regime"
    print("✅ PASSED\n")
    
    # Crisis
    print("TEST 3: CRISIS")
    crisis_returns = pd.Series(np.random.randn(100) * 0.05)  # Huge swings
    
    crisis_result = detector.detect_regime(crisis_returns)
    
    print(f"Detected: {crisis_result['regime']}")
    print(f"Confidence: {crisis_result['confidence']:.2%}")
    assert crisis_result['regime'] == 'CRISIS', "Should detect CRISIS regime"
    print("✅ PASSED\n")
    
    # Regime-specific parameters
    print("TEST 4: REGIME PARAMETERS")
    print(f"BULL params: {bull_result['params']['max_positions']} max positions")
    print(f"BEAR params: {bear_result['params']['max_positions']} max positions")
    print(f"CRISIS params: {crisis_result['params']['max_positions']} max positions")
    assert bull_result['params']['max_positions'] > crisis_result['params']['max_positions']
    print("✅ PASSED\n")
    
    print("="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)