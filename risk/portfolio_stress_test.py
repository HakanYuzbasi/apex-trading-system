"""
risk/portfolio_stress_test.py - Portfolio Stress Testing

Simulates extreme market scenarios to assess portfolio resilience:
- Market crashes (2008, 2020 COVID, flash crashes)
- Volatility spikes (VIX 3x increase)
- Correlation breakdown (all correlations spike to 0.95)
- Sector rotation (concentrated losses)
- Liquidity crisis (bid-ask spread widening)

Provides actionable insights on portfolio vulnerabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of stress scenarios."""
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_SPIKE = "correlation_spike"
    SECTOR_ROTATION = "sector_rotation"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    RATE_SHOCK = "rate_shock"
    CUSTOM = "custom"


@dataclass
class StressScenario:
    """Definition of a stress scenario."""
    name: str
    scenario_type: ScenarioType
    description: str

    # Market-wide parameters
    market_return: float = 0.0          # Overall market return (e.g., -0.40 for 40% drop)
    volatility_multiplier: float = 1.0  # Multiplier for volatility
    correlation_override: Optional[float] = None  # Force all correlations to this value

    # Sector-specific impacts
    sector_impacts: Dict[str, float] = field(default_factory=dict)

    # Liquidity parameters
    spread_multiplier: float = 1.0      # Bid-ask spread multiplier
    volume_multiplier: float = 1.0      # Trading volume multiplier

    # Time parameters
    duration_days: int = 1              # How long the scenario lasts
    recovery_days: int = 0              # Days to partial recovery


@dataclass
class StressTestResult:
    """Results of a stress test."""
    scenario_name: str
    scenario_type: ScenarioType

    # Portfolio impact
    portfolio_pnl: float                # Dollar P&L
    portfolio_return: float             # Percentage return
    max_drawdown: float                 # Maximum drawdown during scenario

    # Risk metrics under stress
    var_95_stressed: float              # VaR under stressed conditions
    expected_shortfall: float           # Expected shortfall (CVaR)

    # Position-level impacts
    position_impacts: Dict[str, float]  # P&L by position
    worst_positions: List[Tuple[str, float]]  # Top losers
    best_positions: List[Tuple[str, float]]   # Top gainers (hedges)

    # Sector analysis
    sector_impacts: Dict[str, float]

    # Risk limit breaches
    breached_limits: List[str]
    margin_call_amount: float           # Estimated margin call

    # Liquidity impact
    estimated_liquidation_cost: float   # Cost to exit all positions

    # Recommendations
    recommendations: List[str]


class PortfolioStressTest:
    """
    Portfolio stress testing engine.

    Usage:
        stress_test = PortfolioStressTest(
            positions={"AAPL": 100, "MSFT": 50},
            prices={"AAPL": 150.0, "MSFT": 300.0}
        )

        # Run predefined scenarios
        results = stress_test.run_all_scenarios()

        # Or run specific scenario
        result = stress_test.run_scenario(StressScenario(
            name="Custom Crash",
            scenario_type=ScenarioType.CUSTOM,
            market_return=-0.30
        ))
    """

    # Predefined stress scenarios based on historical events
    PREDEFINED_SCENARIOS = {
        "2008_financial_crisis": StressScenario(
            name="2008 Financial Crisis",
            scenario_type=ScenarioType.MARKET_CRASH,
            description="Lehman-style systemic crisis with 40% market drop",
            market_return=-0.40,
            volatility_multiplier=3.5,
            correlation_override=0.85,
            sector_impacts={
                "Financials": -0.60,
                "Consumer": -0.45,
                "Technology": -0.35,
                "Industrials": -0.40,
                "Materials": -0.45,
                "Healthcare": -0.25,
                "Energy": -0.50
            },
            spread_multiplier=5.0,
            volume_multiplier=3.0,
            duration_days=5
        ),

        "2020_covid_crash": StressScenario(
            name="2020 COVID Crash",
            scenario_type=ScenarioType.MARKET_CRASH,
            description="Rapid 35% market decline like March 2020",
            market_return=-0.35,
            volatility_multiplier=4.0,
            correlation_override=0.90,
            sector_impacts={
                "Travel": -0.70,
                "Energy": -0.60,
                "Financials": -0.40,
                "Consumer": -0.35,
                "Technology": -0.25,
                "Healthcare": -0.15
            },
            spread_multiplier=3.0,
            duration_days=3
        ),

        "flash_crash": StressScenario(
            name="Flash Crash",
            scenario_type=ScenarioType.FLASH_CRASH,
            description="May 2010 style flash crash - rapid 10% drop and recovery",
            market_return=-0.10,
            volatility_multiplier=6.0,
            spread_multiplier=10.0,
            volume_multiplier=5.0,
            duration_days=1,
            recovery_days=1
        ),

        "vix_spike": StressScenario(
            name="VIX Spike",
            scenario_type=ScenarioType.VOLATILITY_SPIKE,
            description="VIX triples from normal levels",
            market_return=-0.08,
            volatility_multiplier=3.0,
            correlation_override=0.75,
            spread_multiplier=2.0,
            duration_days=5
        ),

        "correlation_breakdown": StressScenario(
            name="Correlation Breakdown",
            scenario_type=ScenarioType.CORRELATION_SPIKE,
            description="All correlations spike to 0.95 - diversification fails",
            market_return=-0.15,
            volatility_multiplier=2.0,
            correlation_override=0.95,
            duration_days=3
        ),

        "tech_sector_crash": StressScenario(
            name="Tech Sector Crash",
            scenario_type=ScenarioType.SECTOR_ROTATION,
            description="Technology sector drops 30% while others flat",
            market_return=-0.08,
            sector_impacts={
                "Technology": -0.30,
                "Communication": -0.25,
                "Consumer": -0.05,
                "Financials": 0.02,
                "Healthcare": 0.03,
                "Energy": 0.05
            },
            duration_days=5
        ),

        "liquidity_crisis": StressScenario(
            name="Liquidity Crisis",
            scenario_type=ScenarioType.LIQUIDITY_CRISIS,
            description="Spreads widen 5x, volume drops 80%",
            market_return=-0.05,
            spread_multiplier=5.0,
            volume_multiplier=0.2,
            duration_days=3
        ),

        "rate_shock": StressScenario(
            name="Rate Shock",
            scenario_type=ScenarioType.RATE_SHOCK,
            description="Sudden 100bps rate increase",
            market_return=-0.12,
            sector_impacts={
                "Technology": -0.18,
                "Consumer": -0.15,
                "Financials": -0.08,
                "Utilities": -0.20,
                "Real Estate": -0.25
            },
            duration_days=5
        )
    }

    # Sector mappings
    SECTOR_MAP = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
        'GOOGL': 'Technology', 'META': 'Communication', 'TSLA': 'Consumer',
        'AVGO': 'Technology', 'ORCL': 'Technology', 'CSCO': 'Technology',
        'ADBE': 'Technology', 'CRM': 'Technology', 'ACN': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'IBM': 'Technology',
        'QCOM': 'Technology', 'TXN': 'Technology', 'AMAT': 'Technology',
        'MU': 'Technology', 'LRCX': 'Technology',
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'GS': 'Financials', 'MS': 'Financials', 'C': 'Financials',
        'BLK': 'Financials', 'AXP': 'Financials', 'SCHW': 'Financials',
        'USB': 'Financials',
        'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare',
        'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'TMO': 'Healthcare',
        'ABT': 'Healthcare', 'DHR': 'Healthcare', 'PFE': 'Healthcare',
        'BMY': 'Healthcare',
        'AMZN': 'Consumer', 'WMT': 'Consumer', 'HD': 'Consumer',
        'MCD': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer',
        'LOW': 'Consumer', 'TGT': 'Consumer', 'DG': 'Consumer',
        'DLTR': 'Consumer',
        'DIS': 'Communication', 'NFLX': 'Communication', 'CHTR': 'Communication',
        'RTX': 'Industrials', 'CAT': 'Industrials',
        'NEM': 'Materials', 'FCX': 'Materials'
    }

    def __init__(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
        capital: float = 1_000_000.0,
        margin_requirement: float = 0.25
    ):
        """
        Initialize stress test engine.

        Args:
            positions: Dict of symbol -> quantity (positive=long, negative=short)
            prices: Dict of symbol -> current price
            historical_data: Optional historical data for more accurate modeling
            capital: Total portfolio capital
            margin_requirement: Margin requirement ratio
        """
        self.positions = positions
        self.prices = prices
        self.historical_data = historical_data or {}
        self.capital = capital
        self.margin_requirement = margin_requirement

        # Calculate position values
        self.position_values = {
            sym: qty * prices.get(sym, 0)
            for sym, qty in positions.items()
            if qty != 0
        }

        self.portfolio_value = sum(self.position_values.values())

        # Calculate sector exposures
        self.sector_exposures = self._calculate_sector_exposures()

        # Estimate volatilities
        self.volatilities = self._estimate_volatilities()

        logger.info(f"PortfolioStressTest initialized: "
                   f"{len(positions)} positions, "
                   f"${self.portfolio_value:,.0f} exposure")

    def _calculate_sector_exposures(self) -> Dict[str, float]:
        """Calculate exposure by sector."""
        exposures = {}
        for sym, value in self.position_values.items():
            sector = self.SECTOR_MAP.get(sym, 'Other')
            exposures[sector] = exposures.get(sector, 0) + value
        return exposures

    def _estimate_volatilities(self) -> Dict[str, float]:
        """Estimate volatility for each position."""
        volatilities = {}

        for sym in self.positions.keys():
            if sym in self.historical_data and len(self.historical_data[sym]) >= 20:
                df = self.historical_data[sym]
                close = df['close'] if 'close' in df.columns else df.get('Close', pd.Series([]))
                if len(close) >= 20:
                    returns = close.pct_change().dropna()
                    volatilities[sym] = returns.tail(20).std() * np.sqrt(252)
                else:
                    volatilities[sym] = 0.25  # Default 25% annual vol
            else:
                # Default volatility by sector
                sector = self.SECTOR_MAP.get(sym, 'Other')
                sector_vols = {
                    'Technology': 0.35, 'Financials': 0.30,
                    'Healthcare': 0.25, 'Consumer': 0.28,
                    'Communication': 0.32, 'Industrials': 0.25,
                    'Materials': 0.30, 'Energy': 0.35,
                    'Other': 0.30
                }
                volatilities[sym] = sector_vols.get(sector, 0.30)

        return volatilities

    def run_scenario(self, scenario: StressScenario) -> StressTestResult:
        """
        Run a single stress scenario.

        Args:
            scenario: StressScenario to simulate

        Returns:
            StressTestResult with detailed analysis
        """
        logger.info(f"Running stress scenario: {scenario.name}")

        # Calculate position-level impacts
        position_impacts = {}

        for sym, qty in self.positions.items():
            if qty == 0:
                continue

            current_price = self.prices.get(sym, 0)
            if current_price == 0:
                continue

            # Get sector and determine return
            sector = self.SECTOR_MAP.get(sym, 'Other')

            # Sector-specific return or market return
            if sector in scenario.sector_impacts:
                base_return = scenario.sector_impacts[sector]
            else:
                base_return = scenario.market_return

            # Add volatility-scaled random component
            vol = self.volatilities.get(sym, 0.25)
            vol_adjustment = (scenario.volatility_multiplier - 1) * vol * 0.5

            # Total return for this position
            position_return = base_return - vol_adjustment

            # Calculate P&L
            position_value = qty * current_price
            position_pnl = position_value * position_return

            # For short positions, P&L is inverted
            if qty < 0:
                position_pnl = -position_pnl

            position_impacts[sym] = position_pnl

        # Portfolio metrics
        portfolio_pnl = sum(position_impacts.values())
        portfolio_return = portfolio_pnl / self.capital if self.capital > 0 else 0

        # Calculate sector-level impacts
        sector_impacts = {}
        for sym, pnl in position_impacts.items():
            sector = self.SECTOR_MAP.get(sym, 'Other')
            sector_impacts[sector] = sector_impacts.get(sector, 0) + pnl

        # Identify worst and best positions
        sorted_positions = sorted(position_impacts.items(), key=lambda x: x[1])
        worst_positions = sorted_positions[:5]
        best_positions = sorted_positions[-5:][::-1]

        # Calculate stressed VaR
        var_95 = self._calculate_stressed_var(scenario)
        expected_shortfall = var_95 * 1.3  # Approximate ES

        # Calculate liquidation cost
        liquidation_cost = self._estimate_liquidation_cost(scenario)

        # Check for limit breaches
        breached_limits = self._check_limit_breaches(
            portfolio_pnl, portfolio_return, scenario
        )

        # Estimate margin call
        margin_call = self._estimate_margin_call(portfolio_pnl)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            portfolio_return, worst_positions, sector_impacts, scenario
        )

        return StressTestResult(
            scenario_name=scenario.name,
            scenario_type=scenario.scenario_type,
            portfolio_pnl=portfolio_pnl,
            portfolio_return=portfolio_return,
            max_drawdown=abs(min(portfolio_return, 0)),
            var_95_stressed=var_95,
            expected_shortfall=expected_shortfall,
            position_impacts=position_impacts,
            worst_positions=worst_positions,
            best_positions=best_positions,
            sector_impacts=sector_impacts,
            breached_limits=breached_limits,
            margin_call_amount=margin_call,
            estimated_liquidation_cost=liquidation_cost,
            recommendations=recommendations
        )

    def run_all_scenarios(self) -> Dict[str, StressTestResult]:
        """Run all predefined scenarios."""
        results = {}

        for scenario_id, scenario in self.PREDEFINED_SCENARIOS.items():
            results[scenario_id] = self.run_scenario(scenario)

        return results

    def run_custom_crash(self, crash_magnitude: float) -> StressTestResult:
        """
        Run a simple market crash scenario.

        Args:
            crash_magnitude: Market drop (e.g., -0.40 for 40% crash)
        """
        scenario = StressScenario(
            name=f"Custom {abs(crash_magnitude)*100:.0f}% Crash",
            scenario_type=ScenarioType.MARKET_CRASH,
            description=f"Market drops {abs(crash_magnitude)*100:.0f}%",
            market_return=crash_magnitude,
            volatility_multiplier=2.5,
            correlation_override=0.85
        )
        return self.run_scenario(scenario)

    def _calculate_stressed_var(self, scenario: StressScenario) -> float:
        """Calculate VaR under stressed conditions."""
        # Simple parametric VaR with stressed volatility
        portfolio_vol = 0

        for sym, value in self.position_values.items():
            weight = value / self.portfolio_value if self.portfolio_value != 0 else 0
            vol = self.volatilities.get(sym, 0.25)
            stressed_vol = vol * scenario.volatility_multiplier
            portfolio_vol += (weight * stressed_vol) ** 2

        portfolio_vol = np.sqrt(portfolio_vol)

        # 95% VaR (1.65 standard deviations)
        var_95 = self.portfolio_value * portfolio_vol * 1.65 * np.sqrt(scenario.duration_days / 252)

        return var_95

    def _estimate_liquidation_cost(self, scenario: StressScenario) -> float:
        """Estimate cost to liquidate all positions."""
        total_cost = 0

        for sym, value in self.position_values.items():
            # Base spread cost (5 bps) * spread multiplier
            spread_cost = abs(value) * 0.0005 * scenario.spread_multiplier

            # Market impact (larger for low volume)
            impact_cost = abs(value) * 0.001 / scenario.volume_multiplier

            total_cost += spread_cost + impact_cost

        return total_cost

    def _check_limit_breaches(
        self,
        portfolio_pnl: float,
        portfolio_return: float,
        scenario: StressScenario
    ) -> List[str]:
        """Check which risk limits would be breached."""
        breaches = []

        if portfolio_return < -0.03:
            breaches.append("Daily loss limit (-3%)")

        if portfolio_return < -0.05:
            breaches.append("Circuit breaker threshold (-5%)")

        if portfolio_return < -0.08:
            breaches.append("Emergency halt threshold (-8%)")

        # Check sector concentration under stress
        for sector, exposure in self.sector_exposures.items():
            sector_pct = exposure / self.portfolio_value if self.portfolio_value != 0 else 0
            if sector_pct > 0.40:
                breaches.append(f"Sector concentration: {sector} ({sector_pct:.0%})")

        return breaches

    def _estimate_margin_call(self, portfolio_pnl: float) -> float:
        """Estimate margin call amount."""
        new_equity = self.capital + portfolio_pnl
        required_margin = abs(self.portfolio_value) * self.margin_requirement

        if new_equity < required_margin:
            return required_margin - new_equity
        return 0

    def _generate_recommendations(
        self,
        portfolio_return: float,
        worst_positions: List[Tuple[str, float]],
        sector_impacts: Dict[str, float],
        scenario: StressScenario
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Severity-based recommendations
        if portfolio_return < -0.30:
            recommendations.append(
                "CRITICAL: Portfolio would lose >30% - consider significant hedging"
            )
        elif portfolio_return < -0.15:
            recommendations.append(
                "WARNING: Portfolio would lose >15% - review risk exposure"
            )

        # Position-specific recommendations
        if worst_positions:
            worst_sym, worst_loss = worst_positions[0]
            if worst_loss < -self.capital * 0.05:
                recommendations.append(
                    f"Consider reducing {worst_sym} position (potential loss: ${abs(worst_loss):,.0f})"
                )

        # Sector recommendations
        worst_sector = min(sector_impacts.items(), key=lambda x: x[1], default=(None, 0))
        if worst_sector[0] and worst_sector[1] < -self.capital * 0.10:
            recommendations.append(
                f"High {worst_sector[0]} sector exposure - consider hedging or rebalancing"
            )

        # Correlation recommendations
        if scenario.correlation_override and scenario.correlation_override > 0.8:
            recommendations.append(
                "Diversification benefit reduced - consider uncorrelated assets (bonds, gold)"
            )

        # Liquidity recommendations
        if scenario.spread_multiplier > 3:
            recommendations.append(
                "Liquidity stress scenario - ensure adequate cash reserves"
            )

        return recommendations

    def print_summary(self, results: Dict[str, StressTestResult]):
        """Print a summary of stress test results."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PORTFOLIO STRESS TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Portfolio Value: ${self.portfolio_value:,.0f}")
        logger.info(f"Capital: ${self.capital:,.0f}")
        logger.info("")

        # Sort by severity
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].portfolio_return
        )

        logger.info(f"{'Scenario':<30} {'Return':>10} {'P&L':>15} {'VaR 95%':>12}")
        logger.info("-" * 70)

        for scenario_id, result in sorted_results:
            logger.info(
                f"{result.scenario_name:<30} "
                f"{result.portfolio_return:>9.1%} "
                f"${result.portfolio_pnl:>13,.0f} "
                f"${result.var_95_stressed:>10,.0f}"
            )

        logger.info("")

        # Worst case analysis
        worst = sorted_results[0][1]
        logger.info(f"WORST CASE: {worst.scenario_name}")
        logger.info(f"  Portfolio Return: {worst.portfolio_return:.1%}")
        logger.info(f"  Portfolio P&L: ${worst.portfolio_pnl:,.0f}")

        if worst.breached_limits:
            logger.info(f"  Breached Limits: {', '.join(worst.breached_limits)}")

        if worst.margin_call_amount > 0:
            logger.info(f"  Margin Call: ${worst.margin_call_amount:,.0f}")

        logger.info("")
        logger.info("RECOMMENDATIONS:")
        for rec in worst.recommendations:
            logger.info(f"  - {rec}")

        logger.info("=" * 80)
