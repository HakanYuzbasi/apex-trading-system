"""
models/cross_sectional_momentum.py - Cross-Sectional Momentum Factor

Implements Jegadeesh & Titman (1993) cross-sectional momentum:
- Rank stocks by past returns across the universe
- Long top decile, short bottom decile
- Well-documented 0.8-1.2% monthly alpha

Free data: Uses historical prices only (no subscriptions required)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum Factor Generator.
    
    Ranks all stocks in the universe by past returns and generates
    signals based on relative performance (top/bottom quintiles).
    
    Features:
    - Multiple lookback periods (1m, 3m, 6m, 12m)
    - Skip-month adjustment (excludes last month to avoid reversal)
    - Sector-neutral option
    - Volatility-adjusted momentum
    """
    
    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        top_quantile: float = 0.20,
        bottom_quantile: float = 0.20,
        volatility_adjust: bool = True
    ):
        """
        Initialize cross-sectional momentum calculator.
        
        Args:
            lookback_months: Months of returns to consider (default: 12)
            skip_months: Recent months to skip (default: 1, avoids reversal)
            top_quantile: Top percentage for long signals (default: 0.20 = top 20%)
            bottom_quantile: Bottom percentage for short signals (default: 0.20)
            volatility_adjust: Whether to adjust for volatility (Sharpe-like)
        """
        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.top_quantile = top_quantile
        self.bottom_quantile = bottom_quantile
        self.volatility_adjust = volatility_adjust
        
        # Trading days approximation
        self.lookback_days = lookback_months * 21
        self.skip_days = skip_months * 21
        
        # Cache for rankings
        self._last_ranking: Optional[Dict[str, float]] = None
        self._last_ranking_time: Optional[datetime] = None
        self._ranking_cache_minutes: int = 60  # Recalculate hourly
        
        logger.info("CrossSectionalMomentum initialized")
        logger.info(f"  Lookback: {lookback_months}m, Skip: {skip_months}m")
        logger.info(f"  Top {top_quantile*100:.0f}% / Bottom {bottom_quantile*100:.0f}%")
    
    def calculate_universe_momentum(
        self,
        historical_data: Dict[str, pd.DataFrame],
        sectors: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Calculate momentum scores for all stocks in universe.
        
        Args:
            historical_data: Dict of {symbol: DataFrame with 'Close' column}
            sectors: Optional dict of {symbol: sector} for sector-neutral
        
        Returns:
            Dict of {symbol: momentum_score} where score is -1 to 1
        """
        momentum_scores = {}
        raw_momentum = {}
        
        for symbol, data in historical_data.items():
            if 'Close' not in data.columns:
                continue
            
            prices = data['Close']
            
            # Need enough data
            if len(prices) < self.lookback_days + self.skip_days:
                continue
            
            # Calculate momentum return (lookback minus skip period)
            # Example: 12-1 momentum = return from 12 months ago to 1 month ago
            start_idx = -(self.lookback_days + self.skip_days)
            end_idx = -self.skip_days if self.skip_days > 0 else None
            
            start_price = prices.iloc[start_idx]
            end_price = prices.iloc[end_idx] if end_idx else prices.iloc[-1]
            
            if start_price <= 0:
                continue
            
            raw_return = (end_price / start_price) - 1
            
            # Volatility adjustment (optional - creates Sharpe-like measure)
            if self.volatility_adjust:
                returns = prices.pct_change().dropna()
                if len(returns) >= 60:
                    vol = returns.iloc[-60:].std() * np.sqrt(252)
                    if vol > 0:
                        raw_return = raw_return / vol
            
            raw_momentum[symbol] = raw_return
        
        if len(raw_momentum) < 5:
            logger.debug(f"Not enough stocks for cross-sectional ranking ({len(raw_momentum)})")
            return {}
        
        # Rank stocks and assign scores
        momentum_scores = self._rank_and_score(raw_momentum)
        
        # Cache results
        self._last_ranking = momentum_scores
        self._last_ranking_time = datetime.now()
        
        return momentum_scores
    
    def _rank_and_score(self, raw_momentum: Dict[str, float]) -> Dict[str, float]:
        """
        Rank stocks and convert to -1 to 1 scores.
        
        Top quantile: score = 0.5 to 1.0
        Middle: score = -0.5 to 0.5
        Bottom quantile: score = -1.0 to -0.5
        """
        if not raw_momentum:
            return {}
        
        # Sort by momentum
        sorted_symbols = sorted(raw_momentum.keys(), key=lambda s: raw_momentum[s], reverse=True)
        n = len(sorted_symbols)
        
        scores = {}
        for i, symbol in enumerate(sorted_symbols):
            # Percentile rank (0 = best, 1 = worst)
            percentile = i / (n - 1) if n > 1 else 0.5
            
            if percentile <= self.top_quantile:
                # Top quintile: 0.5 to 1.0
                scores[symbol] = 0.5 + 0.5 * (1 - percentile / self.top_quantile)
            elif percentile >= (1 - self.bottom_quantile):
                # Bottom quintile: -1.0 to -0.5
                bottom_percentile = (percentile - (1 - self.bottom_quantile)) / self.bottom_quantile
                scores[symbol] = -0.5 - 0.5 * bottom_percentile
            else:
                # Middle: -0.5 to 0.5 (linear interpolation)
                middle_range = 1 - self.top_quantile - self.bottom_quantile
                middle_percentile = (percentile - self.top_quantile) / middle_range
                scores[symbol] = 0.5 - middle_percentile
        
        return scores
    
    def get_signal(
        self,
        symbol: str,
        historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Get cross-sectional momentum signal for a single symbol.
        
        Args:
            symbol: Stock ticker
            historical_data: Full universe historical data
        
        Returns:
            Dict with 'signal' and 'rank_percentile'
        """
        # Check cache
        if self._should_recalculate():
            self.calculate_universe_momentum(historical_data)
        
        if self._last_ranking is None or symbol not in self._last_ranking:
            return {'signal': 0.0, 'rank_percentile': 0.5, 'universe_size': 0}
        
        score = self._last_ranking[symbol]
        
        # Calculate percentile rank
        all_scores = list(self._last_ranking.values())
        rank = sum(1 for s in all_scores if s > score)
        percentile = rank / len(all_scores) if all_scores else 0.5
        
        return {
            'signal': float(score),
            'rank_percentile': float(percentile),
            'universe_size': len(self._last_ranking)
        }
    
    def _should_recalculate(self) -> bool:
        """Check if rankings should be recalculated."""
        if self._last_ranking is None or self._last_ranking_time is None:
            return True
        
        elapsed = (datetime.now() - self._last_ranking_time).total_seconds() / 60
        return elapsed > self._ranking_cache_minutes
    
    def get_top_momentum_stocks(
        self,
        historical_data: Dict[str, pd.DataFrame],
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top N momentum stocks.
        
        Args:
            historical_data: Universe historical data
            n: Number of top stocks to return
        
        Returns:
            List of (symbol, score) tuples sorted by momentum
        """
        if self._should_recalculate():
            self.calculate_universe_momentum(historical_data)
        
        if self._last_ranking is None:
            return []
        
        sorted_stocks = sorted(
            self._last_ranking.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_stocks[:n]
    
    def get_bottom_momentum_stocks(
        self,
        historical_data: Dict[str, pd.DataFrame],
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get bottom N momentum stocks (for shorting).
        
        Args:
            historical_data: Universe historical data
            n: Number of bottom stocks to return
        
        Returns:
            List of (symbol, score) tuples sorted by momentum (ascending)
        """
        if self._should_recalculate():
            self.calculate_universe_momentum(historical_data)
        
        if self._last_ranking is None:
            return []
        
        sorted_stocks = sorted(
            self._last_ranking.items(),
            key=lambda x: x[1]
        )
        
        return sorted_stocks[:n]
    
    def calculate_sector_neutral_momentum(
        self,
        historical_data: Dict[str, pd.DataFrame],
        sectors: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate sector-neutral cross-sectional momentum.
        
        Ranks stocks within each sector, then combines.
        Reduces sector bias in momentum signals.
        
        Args:
            historical_data: Universe historical data
            sectors: Dict of {symbol: sector}
        
        Returns:
            Dict of {symbol: sector_neutral_score}
        """
        # Group by sector
        sector_data = {}
        for symbol, data in historical_data.items():
            sector = sectors.get(symbol, 'Unknown')
            if sector not in sector_data:
                sector_data[sector] = {}
            sector_data[sector][symbol] = data
        
        # Calculate momentum within each sector
        all_scores = {}
        for sector, data in sector_data.items():
            if len(data) < 3:  # Need at least 3 stocks per sector
                continue
            
            # Temporarily set universe to this sector
            old_ranking = self._last_ranking
            self._last_ranking = None
            
            sector_scores = self.calculate_universe_momentum(data)
            
            # Add sector prefix for debugging
            for symbol, score in sector_scores.items():
                all_scores[symbol] = score
            
            self._last_ranking = old_ranking
        
        logger.info(f"Sector-neutral momentum calculated for {len(all_scores)} stocks")
        return all_scores


def create_momentum_factor(
    lookback: str = '12-1',
    volatility_adjust: bool = True
) -> CrossSectionalMomentum:
    """
    Factory function to create momentum factor.
    
    Args:
        lookback: Lookback specification like '12-1' (12 months minus 1 month)
        volatility_adjust: Whether to adjust for volatility
    
    Returns:
        CrossSectionalMomentum instance
    """
    parts = lookback.split('-')
    lookback_months = int(parts[0])
    skip_months = int(parts[1]) if len(parts) > 1 else 1
    
    return CrossSectionalMomentum(
        lookback_months=lookback_months,
        skip_months=skip_months,
        volatility_adjust=volatility_adjust
    )
