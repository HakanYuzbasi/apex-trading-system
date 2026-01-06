"""
portfolio/correlation_manager.py
PORTFOLIO CORRELATION ANALYSIS & HEDGING
- Track correlation between positions
- Suggest hedges
- Diversification metrics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class CorrelationManager:
    """
    Manage portfolio correlation and suggest hedges.
    
    Features:
    - Calculate pairwise correlations
    - Identify clusters of correlated assets
    - Suggest hedging opportunities
    - Calculate diversification ratio
    """
    
    def __init__(self):
        self.correlation_matrix = pd.DataFrame()
        self.returns_history = {}
        self.correlation_history = []
        
        logger.info("✅ Correlation Manager initialized")
    
    def update_returns(self, symbol: str, returns: pd.Series):
        """Update returns history for a symbol."""
        self.returns_history[symbol] = returns
    
    def calculate_correlation_matrix(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio symbols.
        
        Args:
            symbols: List of symbols (if None, uses all in history)
        
        Returns:
            Correlation matrix DataFrame
        """
        if symbols is None:
            symbols = list(self.returns_history.keys())
        
        if len(symbols) < 2:
            logger.warning("Need at least 2 symbols for correlation")
            return pd.DataFrame()
        
        # Build returns DataFrame
        returns_df = pd.DataFrame({
            symbol: self.returns_history[symbol]
            for symbol in symbols
            if symbol in self.returns_history
        })
        
        # Calculate correlation
        self.correlation_matrix = returns_df.corr()
        
        logger.debug(f"📊 Correlation matrix calculated for {len(symbols)} symbols")
        
        return self.correlation_matrix
    
    def get_portfolio_correlation(self, positions: Dict[str, int]) -> float:
        """
        Calculate average pairwise correlation for current positions.
        
        Args:
            positions: {symbol: quantity}
        
        Returns:
            Average correlation (0 to 1)
        """
        symbols = [s for s in positions.keys() if positions[s] != 0]
        
        if len(symbols) < 2:
            return 0.0
        
        # Get correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            return 0.0
        
        # Calculate average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        correlations = corr_matrix.where(mask).stack().values
        
        avg_correlation = np.mean(correlations)
        
        logger.info(f"📊 Portfolio avg correlation: {avg_correlation:.3f}")
        
        return float(avg_correlation)
    
    def find_hedges(
        self,
        long_positions: List[str],
        available_assets: List[str] = None,
        min_negative_corr: float = -0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Find negatively correlated assets for hedging.
        
        Args:
            long_positions: Symbols we're long
            available_assets: Available assets for hedging
            min_negative_corr: Minimum negative correlation threshold
        
        Returns:
            List of (long_asset, hedge_asset, correlation)
        """
        if available_assets is None:
            available_assets = list(self.returns_history.keys())
        
        # Calculate full correlation matrix
        all_symbols = list(set(long_positions + available_assets))
        corr_matrix = self.calculate_correlation_matrix(all_symbols)
        
        if corr_matrix.empty:
            return []
        
        hedges = []
        
        for long_asset in long_positions:
            if long_asset not in corr_matrix.columns:
                continue
            
            # Find assets with negative correlation
            correlations = corr_matrix[long_asset]
            negative_corr = correlations[correlations < min_negative_corr]
            
            # Exclude self
            negative_corr = negative_corr.drop(long_asset, errors='ignore')
            
            # Sort by most negative (best hedge)
            negative_corr = negative_corr.sort_values()
            
            for hedge_asset, corr in negative_corr.items():
                if hedge_asset in available_assets:
                    hedges.append((long_asset, hedge_asset, float(corr)))
        
        if hedges:
            logger.info(f"🛡️  Found {len(hedges)} hedging opportunities:")
            for long, hedge, corr in hedges[:5]:
                logger.info(f"   {long} → {hedge} (corr: {corr:.3f})")
        
        return hedges
    
    def identify_clusters(
        self,
        symbols: List[str] = None,
        n_clusters: int = 5
    ) -> Dict[int, List[str]]:
        """
        Identify clusters of highly correlated assets using hierarchical clustering.
        
        Args:
            symbols: List of symbols
            n_clusters: Number of clusters to create
        
        Returns:
            {cluster_id: [symbols]}
        """
        if symbols is None:
            symbols = list(self.returns_history.keys())
        
        # Get correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        if corr_matrix.empty or len(symbols) < n_clusters:
            return {}
        
        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix.abs()
        
        # Hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Group symbols by cluster
        clusters = {}
        for i, symbol in enumerate(symbols):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(symbol)
        
        logger.info(f"🔗 Identified {len(clusters)} correlation clusters:")
        for cluster_id, cluster_symbols in clusters.items():
            logger.info(f"   Cluster {cluster_id}: {', '.join(cluster_symbols[:5])}")
        
        return clusters
    
    def calculate_diversification_ratio(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float]
    ) -> float:
        """
        Calculate diversification ratio.
        
        Diversification Ratio = (Weighted avg of volatilities) / (Portfolio volatility)
        
        A ratio > 1 indicates diversification benefit.
        
        Args:
            positions: {symbol: quantity}
            prices: {symbol: price}
        
        Returns:
            Diversification ratio
        """
        symbols = [s for s in positions.keys() if positions[s] != 0]
        
        if len(symbols) < 2:
            return 1.0
        
        # Calculate position values and weights
        values = {s: abs(positions[s]) * prices[s] for s in symbols}
        total_value = sum(values.values())
        weights = {s: values[s] / total_value for s in symbols}
        
        # Calculate individual volatilities
        volatilities = {}
        for symbol in symbols:
            if symbol in self.returns_history:
                vol = self.returns_history[symbol].std() * np.sqrt(252)
                volatilities[symbol] = vol
        
        if not volatilities:
            return 1.0
        
        # Weighted average of volatilities
        weighted_vol = sum(weights[s] * volatilities[s] for s in symbols if s in volatilities)
        
        # Portfolio volatility (accounting for correlations)
        returns_df = pd.DataFrame({
            s: self.returns_history[s] for s in symbols if s in self.returns_history
        })
        
        if returns_df.empty:
            return 1.0
        
        # Calculate weighted portfolio returns
        weighted_returns = sum(
            returns_df[s] * weights[s]
            for s in returns_df.columns
        )
        
        portfolio_vol = weighted_returns.std() * np.sqrt(252)
        
        if portfolio_vol == 0:
            return 1.0
        
        diversification_ratio = weighted_vol / portfolio_vol
        
        logger.info(f"📊 Diversification Ratio: {diversification_ratio:.2f}")
        
        return float(diversification_ratio)
    
    def check_concentration_risk(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        max_correlation: float = 0.7,
        max_cluster_exposure: float = 0.40
    ) -> Dict:
        """
        Check for concentration risk in portfolio.
        
        Returns:
            {
                'concentrated': bool,
                'issues': list of warnings,
                'recommendations': list
            }
        """
        symbols = [s for s in positions.keys() if positions[s] != 0]
        issues = []
        recommendations = []
        
        # Check 1: High average correlation
        avg_corr = self.get_portfolio_correlation(positions)
        if avg_corr > max_correlation:
            issues.append(f"High portfolio correlation: {avg_corr:.2f}")
            recommendations.append("Reduce correlated positions or add hedges")
        
        # Check 2: Cluster concentration
        clusters = self.identify_clusters(symbols)
        
        # Calculate cluster exposures
        values = {s: abs(positions[s]) * prices[s] for s in symbols}
        total_value = sum(values.values())
        
        for cluster_id, cluster_symbols in clusters.items():
            cluster_value = sum(values[s] for s in cluster_symbols if s in values)
            cluster_exposure = cluster_value / total_value
            
            if cluster_exposure > max_cluster_exposure:
                issues.append(
                    f"Cluster {cluster_id} has {cluster_exposure*100:.1f}% exposure "
                    f"(limit: {max_cluster_exposure*100:.1f}%)"
                )
                recommendations.append(f"Reduce exposure to cluster {cluster_id}")
        
        concentrated = len(issues) > 0
        
        if concentrated:
            logger.warning(f"⚠️  Concentration risk detected:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        
        return {
            'concentrated': concentrated,
            'issues': issues,
            'recommendations': recommendations,
            'avg_correlation': avg_corr,
            'clusters': clusters
        }
    
    def suggest_rebalance(
        self,
        current_positions: Dict[str, float],
        target_correlation: float = 0.5
    ) -> List[str]:
        """
        Suggest positions to reduce for better diversification.
        
        Args:
            current_positions: Current portfolio
            target_correlation: Target average correlation
        
        Returns:
            List of symbols to reduce/exit
        """
        symbols = [s for s in current_positions.keys() if current_positions[s] != 0]
        
        if len(symbols) < 3:
            return []
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            return []
        
        # Find symbols with highest average correlation to others
        avg_correlations = {}
        for symbol in symbols:
            others = [s for s in symbols if s != symbol]
            avg_corr = corr_matrix.loc[symbol, others].mean()
            avg_correlations[symbol] = avg_corr
        
        # Sort by highest correlation
        sorted_symbols = sorted(avg_correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Get current avg correlation
        current_avg = self.get_portfolio_correlation(current_positions)
        
        suggestions = []
        
        if current_avg > target_correlation:
            # Suggest reducing most correlated positions
            n_to_reduce = max(1, len(symbols) // 5)  # Reduce 20% of positions
            
            for symbol, corr in sorted_symbols[:n_to_reduce]:
                suggestions.append(symbol)
        
        if suggestions:
            logger.info(f"📉 Suggest reducing: {', '.join(suggestions)}")
        
        return suggestions


if __name__ == "__main__":
    # Test correlation manager
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample correlated returns
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    # Create groups of correlated assets
    tech_returns = np.random.randn(len(dates)) * 0.02
    finance_returns = np.random.randn(len(dates)) * 0.015
    
    symbols = {
        'AAPL': tech_returns + np.random.randn(len(dates)) * 0.005,
        'MSFT': tech_returns + np.random.randn(len(dates)) * 0.005,
        'GOOGL': tech_returns + np.random.randn(len(dates)) * 0.006,
        'JPM': finance_returns + np.random.randn(len(dates)) * 0.005,
        'BAC': finance_returns + np.random.randn(len(dates)) * 0.005,
        'GLD': -tech_returns * 0.3 + np.random.randn(len(dates)) * 0.01,  # Hedge
    }
    
    # Initialize manager
    manager = CorrelationManager()
    
    # Update returns
    for symbol, returns in symbols.items():
        manager.update_returns(symbol, pd.Series(returns, index=dates))
    
    print("\n" + "="*60)
    print("TEST 1: CORRELATION MATRIX")
    print("="*60)
    
    corr_matrix = manager.calculate_correlation_matrix()
    print(corr_matrix)
    
    print("\n" + "="*60)
    print("TEST 2: PORTFOLIO CORRELATION")
    print("="*60)
    
    positions = {'AAPL': 100, 'MSFT': 100, 'GOOGL': 100, 'JPM': 50}
    avg_corr = manager.get_portfolio_correlation(positions)
    print(f"Average portfolio correlation: {avg_corr:.3f}")
    
    print("\n" + "="*60)
    print("TEST 3: FIND HEDGES")
    print("="*60)
    
    hedges = manager.find_hedges(['AAPL', 'MSFT'], list(symbols.keys()))
    
    print("\n" + "="*60)
    print("TEST 4: IDENTIFY CLUSTERS")
    print("="*60)
    
    clusters = manager.identify_clusters()
    
    print("\n" + "="*60)
    print("TEST 5: CONCENTRATION RISK")
    print("="*60)
    
    prices = {s: 100.0 for s in symbols.keys()}
    risk_check = manager.check_concentration_risk(positions, prices)
    
    print("\n✅ Correlation manager tests complete!")
