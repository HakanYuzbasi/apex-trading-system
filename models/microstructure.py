"""
models/microstructure.py - Market Microstructure Features

Advanced features derived from high-fidelity price/volume data.

Features:
1. Volume Profile (POC, VAH, VAL): Distribution of volume at price.
2. Effective Spread (Roll's Measure): Transaction cost estimation.
3. Kyle's Lambda: Proxy for market impact/liquidity.
4. VPIN (Volume-Synchronized Probability of Informed Trading) - Simplified.

These features are critical for:
- Understanding support/resistance (Volume Profile)
- Timing entries (Liquidity)
- Detecting informed trading flow
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfile:
    """Volume Profile for a session."""
    poc: float  # Point of Control (Highest Vol Price)
    vah: float  # Value Area High (70% Vol)
    val: float  # Value Area Low
    total_volume: float
    distribution: Dict[float, float]  # Price -> Vol map


class MicrostructureFeatures:
    """
    Calculator for microstructure features.
    """
    
    def __init__(self):
        pass
        
    def calculate_volume_profile(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        bins: int = 50,
        value_area_pct: float = 0.70
    ) -> VolumeProfile:
        """
        Calculate Volume Profile from intraday (or daily) data.
        
        Args:
            prices: Price series (typically Close or Typical Price)
            volumes: Volume series
            bins: Number of price bins
            value_area_pct: Percentage of volume to include in Value Area (default 70%)
            
        Returns:
            VolumeProfile object
        """
        if len(prices) == 0:
            return VolumeProfile(0, 0, 0, 0, {})
            
        min_p = prices.min()
        max_p = prices.max()
        
        if min_p == max_p:
            return VolumeProfile(min_p, min_p, min_p, volumes.sum(), {min_p: volumes.sum()})
            
        # Create bins
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes, range=(min_p, max_p))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find POC (Point of Control)
        max_idx = np.argmax(hist)
        poc = bin_centers[max_idx]
        total_vol = hist.sum()
        
        # Calculate Value Area (VAH, VAL)
        # Greedy algorithm: start at POC, expand out until 70% vol reached
        target_vol = total_vol * value_area_pct
        current_vol = hist[max_idx]
        
        left = max_idx
        right = max_idx
        
        while current_vol < target_vol:
            # Try expand left
            vol_left = hist[left - 1] if left > 0 else 0
            # Try expand right
            vol_right = hist[right + 1] if right < len(hist) - 1 else 0
            
            if vol_left == 0 and vol_right == 0:
                break
                
            if vol_left > vol_right:
                current_vol += vol_left
                left -= 1
            else:
                current_vol += vol_right
                right += 1
                
        val = bin_centers[left]
        vah = bin_centers[right]
        
        # Create distribution dict (top 20 levels for efficiency)
        dist = {float(bin_centers[i]): float(hist[i]) for i in range(len(hist)) if hist[i] > 0}
        
        return VolumeProfile(
            poc=poc,
            vah=vah,
            val=val,
            total_volume=total_vol,
            distribution=dist
        )

    def roll_effective_spread(self, prices: pd.Series) -> float:
        """
        Calculate Roll's Effective Spread measure using covariance.
        Spread = 2 * sqrt(-Cov(dPt, dPt-1))
        
        Args:
            prices: Close prices
            
        Returns:
            Estimated spread (dollar value). Returns 0 if covariance is positive.
        """
        delta_p = prices.diff()
        # Covariance between price change and previous price change
        cov = delta_p.cov(delta_p.shift(1))
        
        if cov > 0:
            return 0.0
        
        return 2.0 * np.sqrt(-cov)

    def amihud_illiquidity(self, returns: pd.Series, volumes: pd.Series, prices: pd.Series) -> float:
        """
        Amihud Illiquidity Ratio (ILIQ).
        Average of |Return| / (Price * Vol)
        High value = Low Liquidity (Price moves a lot for little volume).
        """
        dollar_vol = prices * volumes
        # Avoid division by zero
        dollar_vol = dollar_vol.replace(0, np.nan)
        
        ratio = returns.abs() / dollar_vol
        return ratio.mean() * 1e6  # Scaled up for readability

    def kyles_lambda(self, price_changes: pd.Series, signed_volume: pd.Series) -> float:
        """
        Kyle's Lambda (Price Impact).
        Slope of regression: dP = lambda * OrderFlow + epsilon
        Requires signed volume (Buy vol - Sell vol).
        If signed volume not available, can approximate using aggressor detection or just use standard volume (less accurate).
        """
        import statsmodels.api as sm
        
        if len(price_changes) < 10:
            return 0.0
            
        df = pd.DataFrame({'dP': price_changes, 'V': signed_volume}).dropna()
        if len(df) < 10:
            return 0.0
            
        model = sm.OLS(df['dP'], df['V']).fit()
        return model.params.iloc[0]

    def estimate_vpin(self, prices: pd.Series, volumes: pd.Series, window: int = 50) -> float:
        """
        Simplified VPIN (Volume-Sychronized Probability of Informed Trading).
        A proxy calculation based on volume imbalance variance.
        """
        # Typically VPIN requires trade-by-trade data bucketed by volume
        # This is a crude time-bar approximation
        
        # Estimate buy/sell volume
        price_diff = prices.diff()
        buy_vol = volumes.where(price_diff > 0, 0)
        sell_vol = volumes.where(price_diff < 0, 0)
        
        # Imbalance
        imbalance = (buy_vol - sell_vol).abs()
        total_vol = buy_vol + sell_vol
        
        return (imbalance.rolling(window).mean() / total_vol.rolling(window).mean()).iloc[-1]
