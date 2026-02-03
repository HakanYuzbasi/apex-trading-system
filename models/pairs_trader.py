"""
models/pairs_trader.py - Statistical Arbitrage (Pairs Trading)

Implements cointegration-based pairs trading strategy.
- Finds cointegrated pairs using Engle-Granger test
- Generates z-score signals for mean reversion
- Dynamically updates hedge ratios

Features:
- Automated pair selection
- Rolling z-score calculation
- Half-life estimation for mean reversion speed
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

logger = logging.getLogger(__name__)


@dataclass
class PairAnalysis:
    """Analysis result for a pair."""
    asset_y: str
    asset_x: str
    is_cointegrated: bool
    p_value: float
    hedge_ratio: float
    z_score: float
    half_life: float
    last_spread: float


class PairsTrader:
    """
    Pairs Trading Strategy Manager.
    
    Identifies and trades cointegrated pairs.
    Strategy:
    1. Regress Y on X: Y = beta * X + spread
    2. Check if spread is stationary (cointegrated)
    3. Trade if spread deviates significantly (z-score > 2)
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        min_half_life: int = 1,
        max_half_life: int = 20
    ):
        """
        Initialize Pairs Trader.
        
        Args:
            lookback_window: Data window for regression
            z_entry: Z-score threshold for entry
            z_exit: Z-score threshold for exit
            min_half_life: Minimum mean-reversion half-life (days)
            max_half_life: Maximum mean-reversion half-life (days)
        """
        self.lookback = lookback_window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        
        self.active_pairs: Dict[str, PairAnalysis] = {}
        
        logger.info("PairsTrader initialized")
        
    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        significance: float = 0.05
    ) -> List[Tuple[str, str, float]]:
        """
        Scan all combinations to find cointegrated pairs.
        WARNING: Computationally expensive O(N^2).
        
        Args:
            prices: DataFrame of asset prices (columns=symbols)
            significance: P-value threshold
            
        Returns:
            List of (Asset Y, Asset X, p-value)
        """
        n = prices.shape[1]
        keys = prices.columns
        pairs = []
        
        # Limit universe size for performance if needed
        if n > 50:
            logger.warning(f"Universe size {n} large for exhaustive pair search. Cap suggested.")
        
        for i in range(n):
            for j in range(i + 1, n):
                s1 = prices.iloc[:, i]
                s2 = prices.iloc[:, j]
                
                # Check cointegration Y = a + b*X
                score, pvalue, _ = coint(s1, s2)
                
                if pvalue < significance:
                    # Verify half-life
                    hedge_ratio = self._calculate_hedge_ratio(s1, s2)
                    spread = s1 - hedge_ratio * s2
                    half_life = self._calculate_half_life(spread)
                    
                    if self.min_half_life <= half_life <= self.max_half_life:
                        pairs.append((keys[i], keys[j], pvalue))
                        logger.info(f"🔗 Found pair: {keys[i]}-{keys[j]} (p={pvalue:.4f}, HL={half_life:.1f}d)")
                        
        return pairs

    def analyze_pair(self, asset_y: str, asset_x: str, data: Dict[str, pd.DataFrame]) -> Optional[PairAnalysis]:
        """
        Analyze a specific pair for trading signals.
        
        Args:
            asset_y: Dependent asset symbol
            asset_x: Independent asset symbol
            data: Dictionary of DataFrames with 'Close' columns
            
        Returns:
            PairAnalysis object or None if sufficient data missing
        """
        if asset_y not in data or asset_x not in data:
            return None
            
        y_series = data[asset_y]['Close'].iloc[-self.lookback:]
        x_series = data[asset_x]['Close'].iloc[-self.lookback:]
        
        if len(y_series) < self.lookback or len(x_series) < self.lookback:
            return None
            
        # Re-align indices
        df = pd.DataFrame({'Y': y_series, 'X': x_series}).dropna()
        if len(df) < self.lookback * 0.8:
            return None
            
        # 1. Calculate Hedge Ratio (Beta)
        hedge_ratio = self._calculate_hedge_ratio(df['Y'], df['X'])
        
        # 2. Calculate Spread
        spread = df['Y'] - hedge_ratio * df['X']
        
        # 3. Calculate Z-Score
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        if spread_std == 0:
            z_score = 0
        else:
            z_score = (spread.iloc[-1] - spread_mean) / spread_std
            
        # 4. Check Cointegration (ADF test on spread)
        # Using ADF on spread is approximately Engle-Granger second step
        adf_result = adfuller(spread)
        p_value = adf_result[1]
        is_coint = p_value < 0.05
        
        # 5. Calculate Half-Life
        half_life = self._calculate_half_life(spread)
        
        analysis = PairAnalysis(
            asset_y=asset_y,
            asset_x=asset_x,
            is_cointegrated=is_coint,
            p_value=p_value,
            hedge_ratio=hedge_ratio,
            z_score=z_score,
            half_life=half_life,
            last_spread=spread.iloc[-1]
        )
        
        # Store analysis
        pair_key = f"{asset_y}-{asset_x}"
        self.active_pairs[pair_key] = analysis
        
        return analysis

    def get_signal(self, asset_y: str, asset_x: str) -> Dict[str, Any]:
        """
        Get trading signal for a pair.
        
        Returns:
            Dict with action ('BUY_SPREAD', 'SELL_SPREAD', 'EXIT', 'NONE')
        """
        pair_key = f"{asset_y}-{asset_x}"
        if pair_key not in self.active_pairs:
            return {'action': 'NONE'}
            
        analysis = self.active_pairs[pair_key]
        
        # Not cointegrated anymore? Exit or Ignore
        if not analysis.is_cointegrated or not (self.min_half_life <= analysis.half_life <= self.max_half_life):
            return {
                'action': 'EXIT',
                'reason': f"Broken correlation (p={analysis.p_value:.3f})"
            }
            
        z = analysis.z_score
        
        # Mean Reversion Logic
        if z > self.z_entry:
            # Spread is too HIGH -> Sell Spread (Sell Y, Buy X)
            return {
                'action': 'SELL_SPREAD',
                'description': f"Short {asset_y} / Long {asset_x}",
                'z_score': z,
                'hedge_ratio': analysis.hedge_ratio
            }
        elif z < -self.z_entry:
            # Spread is too LOW -> Buy Spread (Buy Y, Sell X)
            return {
                'action': 'BUY_SPREAD',
                'description': f"Long {asset_y} / Short {asset_x}",
                'z_score': z,
                'hedge_ratio': analysis.hedge_ratio
            }
        elif abs(z) < self.z_exit:
            return {
                'action': 'EXIT',
                'reason': f"Mean reversion target reached (z={z:.2f})"
            }
            
        return {'action': 'HOLD', 'z_score': z}

    def _calculate_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        """Calculate OLS hedge ratio."""
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        return model.params.iloc[1]  # Beta coefficient

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean reversion half-life using Ornstein-Uhlenbeck process.
        dS(t) = -theta * (S(t) - mu) * dt + sigma * dW
        """
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()
        
        df = pd.DataFrame({'y': spread_diff, 'x': spread_lag, 'const': 1}).dropna()
        
        res = sm.OLS(df['y'], df[['x', 'const']]).fit()
        theta = -res.params['x']
        
        if theta <= 0:
            return float('inf')
            
        half_life = np.log(2) / theta
        return half_life
