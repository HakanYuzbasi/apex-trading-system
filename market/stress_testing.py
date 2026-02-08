"""
market/stress_testing.py
STRESS TESTING FRAMEWORK
- Historical crisis scenarios
- Monte Carlo stress tests
- Tail risk analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime, timedelta
from scipy import stats

from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


class StressTestingEngine:
    """
    Test strategy robustness under extreme conditions.
    
    Scenarios:
    1. Historical crises (2008, 2020, etc.)
    2. Monte Carlo simulations
    3. Extreme volatility spikes
    4. Flash crashes
    5. Liquidity crises
    """
    
    def __init__(self):
        self.stress_results = []
        
        logger.info("✅ Stress Testing Engine initialized")
    
    def test_historical_crises(
        self,
        portfolio: Dict[str, int],
        portfolio_value: float,
        positions_data: Dict[str, pd.Series]
    ) -> Dict:
        """
        Test portfolio against historical crisis scenarios.
        
        Scenarios:
        - 2020 COVID Crash (-34% in 1 month)
        - 2008 Financial Crisis (-50% over 6 months)
        - 1987 Black Monday (-22% in 1 day)
        - 2022 Rate Shock (-18% in 3 months)
        - 2011 European Debt Crisis (-19%)
        - 2000 Dot-com Bubble (-78% over 2 years)
        
        Args:
            portfolio: Current positions
            portfolio_value: Current value
            positions_data: Historical returns for positions
        
        Returns:
            Stress test results
        """
        logger.info("🚨 Running Historical Crisis Stress Tests...")
        
        scenarios = {
            '2020_COVID_CRASH': {
                'return': -0.34,
                'volatility': 0.82,
                'duration_days': 30,
                'recovery_days': 150,
                'max_drawdown': -0.34
            },
            '2008_FINANCIAL_CRISIS': {
                'return': -0.57,
                'volatility': 1.5,
                'duration_days': 180,
                'recovery_days': 1460,  # 4 years
                'max_drawdown': -0.57
            },
            '1987_BLACK_MONDAY': {
                'return': -0.22,
                'volatility': 2.0,
                'duration_days': 1,
                'recovery_days': 440,
                'max_drawdown': -0.22
            },
            '2022_RATE_SHOCK': {
                'return': -0.18,
                'volatility': 0.65,
                'duration_days': 90,
                'recovery_days': 180,
                'max_drawdown': -0.25
            },
            '2011_EUROPEAN_CRISIS': {
                'return': -0.19,
                'volatility': 0.75,
                'duration_days': 120,
                'recovery_days': 365,
                'max_drawdown': -0.21
            },
            '2000_DOTCOM_CRASH': {
                'return': -0.78,
                'volatility': 1.2,
                'duration_days': 730,
                'recovery_days': 2555,  # 7 years
                'max_drawdown': -0.78
            }
        }
        
        results = {}
        
        for scenario_name, params in scenarios.items():
            logger.info(f"\n📉 Testing: {scenario_name}")
            
            # Simulate scenario
            result = self._simulate_crisis_scenario(
                portfolio,
                portfolio_value,
                params
            )
            
            results[scenario_name] = result
            
            # Log results
            logger.info(f"   Drawdown: {result['max_drawdown']*100:.1f}%")
            logger.info(f"   Final Value: ${result['final_value']:,.0f}")
            logger.info(f"   Loss: ${result['total_loss']:,.0f}")
            logger.info(f"   Survived: {'✅ YES' if result['survived'] else '❌ NO'}")
            logger.info(f"   Recovery Days: {result['recovery_days']}")
        
        # Summary
        survival_rate = sum(1 for r in results.values() if r['survived']) / len(results)
        avg_drawdown = np.mean([r['max_drawdown'] for r in results.values()])
        worst_scenario = min(results.items(), key=lambda x: x[1]['final_value'])
        
        summary = {
            'scenarios_tested': len(results),
            'survival_rate': survival_rate,
            'avg_drawdown': avg_drawdown,
            'worst_scenario': worst_scenario[0],
            'worst_drawdown': worst_scenario[1]['max_drawdown'],
            'results': results
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 STRESS TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Survival Rate: {survival_rate*100:.0f}%")
        logger.info(f"Average Drawdown: {avg_drawdown*100:.1f}%")
        logger.info(f"Worst Scenario: {worst_scenario[0]} ({worst_scenario[1]['max_drawdown']*100:.1f}% loss)")
        logger.info(f"{'='*60}\n")
        
        return summary
    
    def _simulate_crisis_scenario(
        self,
        portfolio: Dict,
        portfolio_value: float,
        scenario_params: Dict
    ) -> Dict:
        """Simulate a single crisis scenario."""
        
        duration = scenario_params['duration_days']
        total_return = scenario_params['return']
        volatility = scenario_params['volatility']
        
        # Generate daily returns for scenario
        daily_return = total_return / duration
        daily_vol = volatility / np.sqrt(252)
        
        # Simulate path
        daily_returns = np.random.normal(daily_return, daily_vol, duration)
        
        # Calculate equity curve
        equity_curve = [portfolio_value]
        for ret in daily_returns:
            equity_curve.append(equity_curve[-1] * (1 + ret))
        
        equity_curve = np.array(equity_curve)
        
        # Calculate metrics
        final_value = equity_curve[-1]
        total_loss = final_value - portfolio_value
        max_drawdown = (np.min(equity_curve) - portfolio_value) / portfolio_value
        
        # Check if portfolio survived (didn't lose > 50%)
        survived = max_drawdown > -0.50
        
        # Estimate recovery time
        if survived and final_value < portfolio_value:
            recovery_rate = 0.10 / 252  # Assume 10% annual recovery
            days_to_recover = int(np.log(portfolio_value / final_value) / recovery_rate)
        else:
            days_to_recover = scenario_params['recovery_days']
        
        return {
            'final_value': final_value,
            'total_loss': total_loss,
            'max_drawdown': max_drawdown,
            'survived': survived,
            'recovery_days': days_to_recover,
            'equity_curve': equity_curve
        }
    
    def monte_carlo_stress_test(
        self,
        portfolio_value: float,
        num_simulations: int = 1000,
        time_horizon_days: int = 252,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Monte Carlo stress testing.
        
        Generate thousands of random market paths and test portfolio.
        
        Args:
            portfolio_value: Starting value
            num_simulations: Number of paths to simulate
            time_horizon_days: Days to simulate
            confidence_level: Confidence level for VaR
        
        Returns:
            Monte Carlo results
        """
        logger.info(f"🎲 Running Monte Carlo Stress Test ({num_simulations} simulations)...")
        
        # Portfolio parameters (simplified)
        annual_return = 0.10
        annual_volatility = 0.20
        
        daily_return = annual_return / 252
        daily_vol = annual_volatility / np.sqrt(252)
        
        final_values = []
        max_drawdowns = []
        
        for i in range(num_simulations):
            # Generate random path
            daily_returns = np.random.normal(daily_return, daily_vol, time_horizon_days)
            
            # Calculate equity curve
            equity = portfolio_value * np.cumprod(1 + daily_returns)
            
            # Record metrics
            final_values.append(equity[-1])
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            max_drawdowns.append(drawdown.min())
        
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate statistics
        var_percentile = 100 * (1 - confidence_level)
        value_at_risk = np.percentile(final_values - portfolio_value, var_percentile)
        cvar = np.mean(final_values[final_values < np.percentile(final_values, var_percentile)]) - portfolio_value
        
        worst_case = final_values.min()
        best_case = final_values.max()
        median_case = np.median(final_values)
        
        prob_loss = (final_values < portfolio_value).sum() / num_simulations
        prob_severe_loss = (final_values < portfolio_value * 0.80).sum() / num_simulations  # >20% loss
        
        result = {
            'num_simulations': num_simulations,
            'time_horizon_days': time_horizon_days,
            'starting_value': portfolio_value,
            'median_final_value': median_case,
            'best_case': best_case,
            'worst_case': worst_case,
            'value_at_risk': value_at_risk,
            'conditional_var': cvar,
            'prob_loss': prob_loss,
            'prob_severe_loss': prob_severe_loss,
            'avg_max_drawdown': max_drawdowns.mean(),
            'worst_drawdown': max_drawdowns.min(),
            'distribution': {
                'percentile_5': np.percentile(final_values, 5),
                'percentile_25': np.percentile(final_values, 25),
                'percentile_50': median_case,
                'percentile_75': np.percentile(final_values, 75),
                'percentile_95': np.percentile(final_values, 95),
            }
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎲 MONTE CARLO RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Median Outcome: ${median_case:,.0f}")
        logger.info(f"Best Case (95%): ${result['distribution']['percentile_95']:,.0f}")
        logger.info(f"Worst Case (5%): ${result['distribution']['percentile_5']:,.0f}")
        logger.info(f"Value at Risk ({confidence_level*100:.0f}%): ${value_at_risk:,.0f}")
        logger.info(f"Conditional VaR: ${cvar:,.0f}")
        logger.info(f"Probability of Loss: {prob_loss*100:.1f}%")
        logger.info(f"Probability of >20% Loss: {prob_severe_loss*100:.1f}%")
        logger.info(f"Average Max Drawdown: {max_drawdowns.mean()*100:.1f}%")
        logger.info(f"{'='*60}\n")
        
        return result
    
    def test_extreme_volatility(
        self,
        portfolio_value: float,
        normal_volatility: float = 0.20,
        extreme_multiplier: float = 3.0,
        duration_days: int = 30
    ) -> Dict:
        """
        Test portfolio under extreme volatility spike.
        
        Simulates a VIX spike from 20 to 60+.
        """
        logger.info(f"⚡ Testing Extreme Volatility Spike (Vol x{extreme_multiplier})...")
        
        extreme_vol = normal_volatility * extreme_multiplier
        daily_vol = extreme_vol / np.sqrt(252)
        
        # Simulate extreme volatility period
        daily_returns = np.random.normal(-0.0001, daily_vol, duration_days)  # Slight negative drift
        
        equity_curve = portfolio_value * np.cumprod(1 + daily_returns)
        
        final_value = equity_curve[-1]
        max_drawdown = (equity_curve.min() - portfolio_value) / portfolio_value
        
        result = {
            'starting_value': portfolio_value,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'volatility_multiplier': extreme_multiplier,
            'survived': max_drawdown > -0.50,
            'days_underwater': (equity_curve < portfolio_value).sum()
        }
        
        logger.info(f"   Final Value: ${final_value:,.0f}")
        logger.info(f"   Max Drawdown: {max_drawdown*100:.1f}%")
        logger.info(f"   Survived: {'✅ YES' if result['survived'] else '❌ NO'}")
        
        return result
    
    def test_liquidity_crisis(
        self,
        portfolio: Dict[str, int],
        prices: Dict[str, float],
        liquidity_shock_pct: float = 0.10
    ) -> Dict:
        """
        Test ability to exit positions during liquidity crisis.
        
        Simulate extreme bid-ask spreads and limited liquidity.
        """
        logger.info(f"💧 Testing Liquidity Crisis ({liquidity_shock_pct*100:.0f}% slippage)...")
        
        total_value = sum(abs(qty) * prices[sym] for sym, qty in portfolio.items())
        
        # Simulate forced liquidation with slippage
        liquidation_value = 0
        total_slippage = 0
        
        for symbol, qty in portfolio.items():
            if qty == 0:
                continue
            
            price = prices[symbol]
            position_value = abs(qty) * price
            
            # Slippage increases with position size
            slippage_pct = liquidity_shock_pct * (1 + position_value / total_value)
            slippage = position_value * slippage_pct
            
            liquidation_value += (position_value - slippage)
            total_slippage += slippage
        
        result = {
            'total_value': total_value,
            'liquidation_value': liquidation_value,
            'total_slippage': total_slippage,
            'slippage_pct': total_slippage / total_value,
            'survived': liquidation_value > total_value * 0.50
        }
        
        logger.info(f"   Total Value: ${total_value:,.0f}")
        logger.info(f"   Liquidation Value: ${liquidation_value:,.0f}")
        logger.info(f"   Total Slippage: ${total_slippage:,.0f} ({result['slippage_pct']*100:.1f}%)")
        
        return result


if __name__ == "__main__":
    # Test stress testing engine
    setup_logging(level="INFO", log_file=None, json_format=False, console_output=True)
    
    engine = StressTestingEngine()
    
    print("\n" + "="*60)
    print("TEST 1: HISTORICAL CRISES")
    print("="*60)
    
    portfolio = {'AAPL': 100, 'MSFT': 50, 'GOOGL': 30}
    portfolio_value = 100000
    
    crisis_results = engine.test_historical_crises(portfolio, portfolio_value, {})
    
    print("\n" + "="*60)
    print("TEST 2: MONTE CARLO")
    print("="*60)
    
    mc_results = engine.monte_carlo_stress_test(
        portfolio_value=100000,
        num_simulations=1000,
        time_horizon_days=252
    )
    
    print("\n" + "="*60)
    print("TEST 3: EXTREME VOLATILITY")
    print("="*60)
    
    vol_results = engine.test_extreme_volatility(
        portfolio_value=100000,
        extreme_multiplier=3.0
    )
    
    print("\n" + "="*60)
    print("TEST 4: LIQUIDITY CRISIS")
    print("="*60)
    
    prices = {'AAPL': 180, 'MSFT': 400, 'GOOGL': 140}
    liq_results = engine.test_liquidity_crisis(portfolio, prices)
    
    print("\n✅ Stress testing engine tests complete!")
