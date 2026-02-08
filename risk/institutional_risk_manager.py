"""
risk/institutional_risk_manager.py

Institutional-Grade Risk Management

Key Features:
- Position sizing with volatility targeting
- Correlation-aware exposure limits
- Drawdown-based risk reduction
- Sector and factor concentration limits
- VaR and Expected Shortfall calculations
- Circuit breaker with graduated response

Author: Institutional Quant Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk regime levels."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    OPEN = "open"           # Trading allowed
    WARNING = "warning"     # Reduced risk
    TRIPPED = "tripped"     # Trading halted
    COOLDOWN = "cooldown"   # Recovery period


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Position limits
    max_position_pct: float = 0.05       # 5% of capital per position
    min_position_pct: float = 0.01       # 1% minimum position
    max_positions: int = 15

    # Sector limits
    max_sector_pct: float = 0.30         # 30% per sector
    max_single_name_pct: float = 0.05    # 5% single name

    # Volatility targeting
    target_portfolio_vol: float = 0.12   # 12% annual vol target
    vol_lookback_days: int = 20
    vol_scaling_enabled: bool = True

    # Correlation limits
    max_correlation: float = 0.70
    correlation_lookback_days: int = 60

    # Drawdown limits
    max_daily_loss_pct: float = 0.02     # 2% daily loss limit
    max_drawdown_pct: float = 0.10       # 10% max drawdown
    dd_risk_reduction_start: float = 0.05  # Start reducing at 5%
    dd_risk_reduction_factor: float = 0.5  # Reduce risk by 50% at threshold

    # Circuit breaker
    cb_daily_loss_trigger: float = 0.02   # 2%
    cb_drawdown_trigger: float = 0.08     # 8%
    cb_consecutive_losses: int = 5
    cb_cooldown_hours: int = 24

    # Transaction costs
    commission_per_trade: float = 1.0     # $1 per trade
    slippage_bps: float = 5.0             # 5 basis points


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    quantity: int
    market_value: float
    weight: float
    volatility: float
    var_95: float  # 1-day VaR at 95%
    beta: float
    sector: str


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    timestamp: datetime
    total_value: float
    cash: float

    # Position metrics
    num_positions: int
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float

    # Risk metrics
    portfolio_volatility: float
    var_95: float           # Portfolio VaR
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float

    # Concentration
    largest_position_pct: float
    sector_concentration: Dict[str, float]
    herfindahl_index: float  # Concentration measure

    # Risk regime
    risk_level: RiskLevel
    risk_multiplier: float


@dataclass
class SizingResult:
    """Position sizing result."""
    symbol: str
    target_shares: int
    target_value: float
    target_weight: float

    # Sizing factors
    base_size: float
    vol_adjusted_size: float
    correlation_penalty: float
    drawdown_adjustment: float
    sector_limit_applied: bool

    # Risk metrics
    position_var: float
    marginal_var: float

    # Constraints hit
    constraints: List[str] = field(default_factory=list)


class InstitutionalRiskManager:
    """
    Institutional-grade risk management.

    Features:
    - Volatility-targeted position sizing
    - Correlation-aware constraints
    - Drawdown-based risk reduction
    - VaR and Expected Shortfall
    - Graduated circuit breaker
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

        # State tracking
        self.starting_capital: float = 0.0
        self.peak_capital: float = 0.0
        self.current_capital: float = 0.0

        # Performance tracking
        self.daily_pnl: float = 0.0
        self.daily_start_capital: float = 0.0
        self.trade_results: List[float] = []
        self.equity_history: List[Tuple[datetime, float]] = []

        # Circuit breaker
        self.cb_state = CircuitBreakerState.OPEN
        self.cb_trip_time: Optional[datetime] = None
        self.consecutive_losses: int = 0

        # Cache
        self._volatility_cache: Dict[str, float] = {}
        self._correlation_cache: Dict[str, Dict[str, float]] = {}
        self._last_cache_update: Optional[datetime] = None

        logger.info("InstitutionalRiskManager initialized")
        logger.info(f"  Max position: {self.config.max_position_pct:.1%}")
        logger.info(f"  Max sector: {self.config.max_sector_pct:.1%}")
        logger.info(f"  Target vol: {self.config.target_portfolio_vol:.1%}")
        logger.info(f"  Max drawdown: {self.config.max_drawdown_pct:.1%}")

    def initialize(self, capital: float):
        """Initialize with starting capital."""
        self.starting_capital = capital
        self.peak_capital = capital
        self.current_capital = capital
        self.daily_start_capital = capital
        logger.info(f"Risk manager initialized with ${capital:,.2f}")

    def update_capital(self, capital: float):
        """Update current capital and track metrics."""
        self.current_capital = capital
        self.peak_capital = max(self.peak_capital, capital)

        # Record equity history
        self.equity_history.append((datetime.now(), capital))

        # Keep last 252 days
        if len(self.equity_history) > 252 * 24:  # Assuming hourly updates
            self.equity_history = self.equity_history[-252*24:]

    def start_new_day(self):
        """Reset daily tracking."""
        self.daily_start_capital = self.current_capital
        self.daily_pnl = 0.0
        logger.debug(f"New trading day started. Capital: ${self.current_capital:,.2f}")

    def record_trade_result(self, pnl: float):
        """Record trade P&L for circuit breaker tracking."""
        self.trade_results.append(pnl)
        self.daily_pnl += pnl

        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check circuit breaker
        self._check_circuit_breaker()

    # ═══════════════════════════════════════════════════════════════
    # POSITION SIZING
    # ═══════════════════════════════════════════════════════════════

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float,
        signal_confidence: float,
        current_positions: Dict[str, int],
        price_cache: Dict[str, float],
        sector: str = "Unknown",
        historical_prices: Optional[pd.Series] = None
    ) -> SizingResult:
        """
        Calculate optimal position size with institutional constraints.

        Args:
            symbol: Stock ticker
            price: Current price
            signal_strength: Signal value [-1, 1]
            signal_confidence: Signal confidence [0, 1]
            current_positions: Dict of symbol -> quantity
            price_cache: Dict of symbol -> price
            sector: Symbol's sector
            historical_prices: Optional price history for vol calc

        Returns:
            SizingResult with target position and constraints
        """
        constraints_hit = []

        # 1. BASE SIZE - percentage of capital
        base_value = self.current_capital * self.config.max_position_pct
        base_shares = int(base_value / price) if price > 0 else 0

        # 2. SIGNAL-ADJUSTED SIZE
        # Scale by signal strength and confidence with a floor
        # Use sqrt to avoid crushing small signals: signal=0.35,conf=0.40 → 0.62 instead of 0.14
        signal_factor = np.sqrt(abs(signal_strength) * signal_confidence)
        signal_factor = max(signal_factor, 0.5)  # Floor at 50% of base size
        signal_adjusted_value = base_value * signal_factor

        # 3. VOLATILITY ADJUSTMENT
        vol_adjusted_value = signal_adjusted_value
        position_vol = 0.0

        if self.config.vol_scaling_enabled and historical_prices is not None:
            position_vol = self._calculate_volatility(symbol, historical_prices)

            if position_vol > 0:
                # Target: position_vol * weight = target_contribution
                # Solve for weight: weight = target_contribution / position_vol
                target_vol_contribution = self.config.target_portfolio_vol / max(self.config.max_positions, 1)
                vol_weight = target_vol_contribution / position_vol

                vol_adjusted_value = self.current_capital * min(vol_weight, self.config.max_position_pct)

                if vol_adjusted_value < signal_adjusted_value:
                    constraints_hit.append(f"vol_scaled:{position_vol:.2%}")

        # 4. CORRELATION PENALTY
        correlation_penalty = 1.0
        if current_positions:
            avg_correlation = self._calculate_avg_correlation(symbol, list(current_positions.keys()))
            if avg_correlation > 0.3:
                # Reduce size for highly correlated positions
                correlation_penalty = max(0.5, 1 - avg_correlation)
                constraints_hit.append(f"corr_penalty:{correlation_penalty:.2f}")

        corr_adjusted_value = vol_adjusted_value * correlation_penalty

        # 5. DRAWDOWN ADJUSTMENT
        drawdown_mult = self._get_drawdown_multiplier()
        dd_adjusted_value = corr_adjusted_value * drawdown_mult

        if drawdown_mult < 1.0:
            constraints_hit.append(f"dd_mult:{drawdown_mult:.2f}")

        # 6. SECTOR LIMIT CHECK
        sector_limit_applied = False
        current_sector_exposure = self._calculate_sector_exposure(
            current_positions, price_cache, sector
        )

        if current_sector_exposure >= self.config.max_sector_pct:
            dd_adjusted_value = 0  # Can't add to this sector
            sector_limit_applied = True
            constraints_hit.append("sector_limit")

        # 7. CALCULATE FINAL SHARES
        final_value = max(0, dd_adjusted_value)
        final_shares = int(final_value / price) if price > 0 else 0

        # Ensure minimum position size
        min_value = self.current_capital * self.config.min_position_pct
        if final_value > 0 and final_value < min_value:
            final_value = 0
            final_shares = 0
            constraints_hit.append("below_minimum")

        # Calculate weight
        weight = final_value / self.current_capital if self.current_capital > 0 else 0

        # Calculate VaR metrics
        position_var = final_value * position_vol * 1.65 / np.sqrt(252) if position_vol > 0 else 0

        return SizingResult(
            symbol=symbol,
            target_shares=final_shares,
            target_value=final_value,
            target_weight=weight,
            base_size=base_value,
            vol_adjusted_size=vol_adjusted_value,
            correlation_penalty=correlation_penalty,
            drawdown_adjustment=drawdown_mult,
            sector_limit_applied=sector_limit_applied,
            position_var=position_var,
            marginal_var=position_var,  # Simplified
            constraints=constraints_hit
        )

    def _calculate_volatility(self, symbol: str, prices: pd.Series) -> float:
        """Calculate annualized volatility."""
        # Check cache
        if symbol in self._volatility_cache:
            return self._volatility_cache[symbol]

        if len(prices) < self.config.vol_lookback_days:
            return 0.20  # Default 20% vol

        returns = prices.pct_change().dropna()
        vol = float(returns.iloc[-self.config.vol_lookback_days:].std() * np.sqrt(252))

        self._volatility_cache[symbol] = vol
        return vol

    def _calculate_avg_correlation(self, symbol: str, existing_symbols: List[str]) -> float:
        """Calculate average correlation with existing positions."""
        if not existing_symbols:
            return 0.0

        # Simplified: return cached or default
        if symbol in self._correlation_cache:
            correlations = [
                self._correlation_cache[symbol].get(s, 0.3)
                for s in existing_symbols
            ]
            return float(np.mean(correlations))

        return 0.3  # Default moderate correlation

    def _calculate_sector_exposure(
        self,
        positions: Dict[str, int],
        price_cache: Dict[str, float],
        target_sector: str
    ) -> float:
        """Calculate current sector exposure."""
        if not positions:
            return 0.0

        sector_value = 0.0
        total_value = 0.0

        for symbol, qty in positions.items():
            if qty == 0:
                continue

            price = price_cache.get(symbol, 0)
            value = abs(qty * price)
            total_value += value

            # Simplified: would need sector mapping
            # For now, assume 10% default
            # In production, use proper sector classification

        return sector_value / total_value if total_value > 0 else 0.0

    def _get_drawdown_multiplier(self) -> float:
        """Get risk multiplier based on current drawdown."""
        if self.peak_capital <= 0:
            return 1.0

        current_dd = (self.peak_capital - self.current_capital) / self.peak_capital

        if current_dd < self.config.dd_risk_reduction_start:
            return 1.0

        if current_dd >= self.config.max_drawdown_pct:
            return 0.0  # Stop trading

        # Linear reduction
        dd_range = self.config.max_drawdown_pct - self.config.dd_risk_reduction_start
        dd_excess = current_dd - self.config.dd_risk_reduction_start
        reduction = (dd_excess / dd_range) * (1 - self.config.dd_risk_reduction_factor)

        return max(self.config.dd_risk_reduction_factor, 1 - reduction)

    # ═══════════════════════════════════════════════════════════════
    # CIRCUIT BREAKER
    # ═══════════════════════════════════════════════════════════════

    def can_trade(self) -> Tuple[bool, str, float]:
        """
        Check if trading is allowed with graduated response.

        Returns:
            Tuple of (can_trade, reason, risk_multiplier)
            - risk_multiplier: 1.0 = full size, 0.5 = half size, 0.0 = no trading
        """
        # Check circuit breaker state
        if self.cb_state == CircuitBreakerState.TRIPPED:
            return False, "Circuit breaker tripped", 0.0

        if self.cb_state == CircuitBreakerState.COOLDOWN:
            if self.cb_trip_time:
                cooldown_end = self.cb_trip_time + timedelta(hours=self.config.cb_cooldown_hours)
                if datetime.now() < cooldown_end:
                    remaining = cooldown_end - datetime.now()
                    return False, f"Cooldown: {remaining.seconds // 60} minutes remaining", 0.0
                else:
                    self.cb_state = CircuitBreakerState.OPEN
                    logger.info("Circuit breaker cooldown complete - trading resumed")

        # Check daily loss
        if self.daily_start_capital > 0:
            daily_return = (self.current_capital - self.daily_start_capital) / self.daily_start_capital

            if daily_return <= -self.config.cb_daily_loss_trigger:
                self._trip_circuit_breaker("Daily loss limit")
                return False, f"Daily loss: {daily_return:.2%}", 0.0

        # Check drawdown
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

            if drawdown >= self.config.cb_drawdown_trigger:
                self._trip_circuit_breaker("Drawdown limit")
                return False, f"Drawdown: {drawdown:.2%}", 0.0

        # Check consecutive losses
        if self.consecutive_losses >= self.config.cb_consecutive_losses:
            self._trip_circuit_breaker("Consecutive losses")
            return False, f"Consecutive losses: {self.consecutive_losses}", 0.0

        # Warning state - graduated response: reduce position sizes by 50%
        if self.cb_state == CircuitBreakerState.WARNING:
            return True, "Warning: Elevated risk - 50% position size", 0.5

        return True, "OK", 1.0

    def _check_circuit_breaker(self):
        """Check and update circuit breaker state."""
        if self.cb_state == CircuitBreakerState.TRIPPED:
            return

        # Check thresholds
        daily_return = 0.0
        if self.daily_start_capital > 0:
            daily_return = (self.current_capital - self.daily_start_capital) / self.daily_start_capital

        drawdown = 0.0
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        # Trip conditions
        if (daily_return <= -self.config.cb_daily_loss_trigger or
            drawdown >= self.config.cb_drawdown_trigger or
            self.consecutive_losses >= self.config.cb_consecutive_losses):

            self._trip_circuit_breaker("Threshold breached")

        # Warning conditions (50% of trip threshold)
        elif (daily_return <= -self.config.cb_daily_loss_trigger * 0.5 or
              drawdown >= self.config.cb_drawdown_trigger * 0.5):

            if self.cb_state != CircuitBreakerState.WARNING:
                self.cb_state = CircuitBreakerState.WARNING
                logger.warning("⚠️ Circuit breaker WARNING - Risk elevated")

    def _trip_circuit_breaker(self, reason: str):
        """Trip the circuit breaker."""
        self.cb_state = CircuitBreakerState.TRIPPED
        self.cb_trip_time = datetime.now()
        logger.error(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
        logger.error(f"   Trading halted for {self.config.cb_cooldown_hours} hours")

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker."""
        self.cb_state = CircuitBreakerState.OPEN
        self.cb_trip_time = None
        self.consecutive_losses = 0
        logger.info("Circuit breaker manually reset")

    # ═══════════════════════════════════════════════════════════════
    # RISK METRICS
    # ═══════════════════════════════════════════════════════════════

    def calculate_portfolio_risk(
        self,
        positions: Dict[str, int],
        price_cache: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame]
    ) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics."""
        timestamp = datetime.now()

        # Calculate exposures
        long_exposure = 0.0
        short_exposure = 0.0
        position_values = {}

        for symbol, qty in positions.items():
            if qty == 0:
                continue

            price = price_cache.get(symbol, 0)
            value = qty * price

            if qty > 0:
                long_exposure += value
            else:
                short_exposure += abs(value)

            position_values[symbol] = abs(value)

        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        # Calculate weights
        total_value = self.current_capital
        weights = {s: v / total_value for s, v in position_values.items()} if total_value > 0 else {}

        # Largest position
        largest_pct = max(weights.values()) if weights else 0.0

        # Herfindahl index (concentration)
        hhi = sum(w ** 2 for w in weights.values()) if weights else 0.0

        # Portfolio volatility and returns for VaR/CVaR
        portfolio_vol = 0.0
        portfolio_returns = None
        
        if historical_data and weights:
            returns_dict = {}
            for symbol, weight in weights.items():
                if symbol in historical_data:
                    df = historical_data[symbol]
                    if 'Close' in df.columns and len(df) >= 60:
                        returns_dict[symbol] = df['Close'].pct_change().dropna()

            if returns_dict:
                # Align returns to common dates
                returns_df = pd.DataFrame(returns_dict)
                returns_df = returns_df.dropna()
                
                if len(returns_df) >= 20:
                    # Calculate weighted portfolio returns
                    weight_array = np.array([weights.get(s, 0) for s in returns_df.columns])
                    portfolio_returns = (returns_df.values @ weight_array)
                    portfolio_vol = float(np.std(portfolio_returns) * np.sqrt(252))

        # VaR (95%, 1-day) - Historical simulation
        var_95 = self._calculate_var_historical(portfolio_returns, total_value, 0.95)
        
        # Expected Shortfall (CVaR) - Average of losses beyond VaR
        es = self._calculate_cvar_historical(portfolio_returns, total_value, 0.95)

        # Drawdown
        max_dd = 0.0
        current_dd = 0.0
        if self.peak_capital > 0:
            current_dd = (self.peak_capital - self.current_capital) / self.peak_capital
            max_dd = current_dd  # Would need history for true max

        # Risk level
        risk_level = self._determine_risk_level(current_dd, portfolio_vol)
        risk_mult = self._get_drawdown_multiplier()

        return PortfolioRisk(
            timestamp=timestamp,
            total_value=total_value,
            cash=total_value - gross_exposure,
            num_positions=len([q for q in positions.values() if q != 0]),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            portfolio_volatility=portfolio_vol,
            var_95=var_95,
            expected_shortfall=es,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            largest_position_pct=largest_pct,
            sector_concentration={},  # Would need sector data
            herfindahl_index=hhi,
            risk_level=risk_level,
            risk_multiplier=risk_mult
        )

    def _calculate_var_historical(
        self,
        returns: Optional[np.ndarray],
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk using historical simulation.

        Args:
            returns: Array of historical portfolio returns
            portfolio_value: Current portfolio value
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR dollar amount (positive = potential loss)
        """
        if returns is None or len(returns) < 20:
            # Fallback to parametric VaR with assumed 20% volatility
            return portfolio_value * 0.20 * 1.65 / np.sqrt(252)

        # Historical VaR: percentile of losses
        percentile = (1 - confidence) * 100
        var_return = np.percentile(returns, percentile)
        var_dollar = abs(var_return) * portfolio_value

        return float(var_dollar)

    def _calculate_cvar_historical(
        self,
        returns: Optional[np.ndarray],
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        CVaR is the average loss in the worst (1-confidence)% of cases.
        More stable than VaR and captures tail risk better.

        Args:
            returns: Array of historical portfolio returns
            portfolio_value: Current portfolio value
            confidence: Confidence level

        Returns:
            CVaR dollar amount (positive = potential loss)
        """
        if returns is None or len(returns) < 20:
            # Fallback: CVaR ≈ 1.25x VaR for normal distribution
            return portfolio_value * 0.20 * 2.06 / np.sqrt(252)

        # Find VaR threshold
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(returns, percentile)

        # CVaR = average of returns worse than VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return self._calculate_var_historical(returns, portfolio_value, confidence) * 1.25

        cvar_return = np.mean(tail_returns)
        cvar_dollar = abs(cvar_return) * portfolio_value

        return float(cvar_dollar)

    def calculate_factor_exposures(
        self,
        positions: Dict[str, int],
        price_cache: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio factor exposures.

        Factors:
        - Market Beta: Sensitivity to market movements
        - Size: Exposure to small vs large cap
        - Value: Exposure to high vs low book-to-market
        - Momentum: Exposure to past winners vs losers

        Args:
            positions: Current positions
            price_cache: Current prices
            historical_data: Historical price data
            market_returns: Market benchmark returns (e.g., SPY)

        Returns:
            Dict of factor exposures
        """
        exposures = {
            'market_beta': 0.0,
            'size_factor': 0.0,
            'momentum_factor': 0.0,
            'volatility_factor': 0.0
        }

        if not positions or not historical_data:
            return exposures

        # Calculate weights
        total_value = sum(
            abs(qty * price_cache.get(sym, 0))
            for sym, qty in positions.items() if qty != 0
        )

        if total_value == 0:
            return exposures

        weighted_beta = 0.0
        weighted_momentum = 0.0
        weighted_vol = 0.0

        for symbol, qty in positions.items():
            if qty == 0:
                continue

            price = price_cache.get(symbol, 0)
            weight = abs(qty * price) / total_value
            sign = 1 if qty > 0 else -1  # Long/short direction

            if symbol in historical_data:
                df = historical_data[symbol]
                if 'Close' in df.columns and len(df) >= 60:
                    prices = df['Close']
                    returns = prices.pct_change().dropna()

                    # Beta calculation (vs market if available)
                    if market_returns is not None and len(market_returns) >= 60:
                        aligned = pd.concat([returns, market_returns], axis=1).dropna()
                        if len(aligned) >= 20:
                            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
                            weighted_beta += weight * beta * sign
                    else:
                        weighted_beta += weight * 1.0 * sign  # Assume beta=1

                    # Momentum: 12-1 month return
                    if len(prices) >= 252:
                        mom_12m = (prices.iloc[-21] / prices.iloc[-252]) - 1
                        weighted_momentum += weight * mom_12m * sign
                    elif len(prices) >= 60:
                        mom_60d = (prices.iloc[-1] / prices.iloc[-60]) - 1
                        weighted_momentum += weight * mom_60d * sign

                    # Volatility exposure
                    vol = returns.std() * np.sqrt(252)
                    weighted_vol += weight * vol

        exposures['market_beta'] = float(weighted_beta)
        exposures['momentum_factor'] = float(weighted_momentum)
        exposures['volatility_factor'] = float(weighted_vol)

        return exposures

    def check_factor_limits(
        self,
        factor_exposures: Dict[str, float],
        limits: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if factor exposures are within limits.

        Args:
            factor_exposures: Current factor exposures
            limits: Dict of factor -> (min, max) limits

        Returns:
            Tuple of (within_limits, list of violations)
        """
        if limits is None:
            limits = {
                'market_beta': (0.5, 1.5),      # Net beta between 0.5 and 1.5
                'momentum_factor': (-0.5, 0.5),  # Limit momentum tilt
                'volatility_factor': (0.1, 0.3)  # Target vol range
            }

        violations = []

        for factor, exposure in factor_exposures.items():
            if factor in limits:
                min_limit, max_limit = limits[factor]
                if exposure < min_limit:
                    violations.append(f"{factor}: {exposure:.2f} < {min_limit:.2f}")
                elif exposure > max_limit:
                    violations.append(f"{factor}: {exposure:.2f} > {max_limit:.2f}")

        return len(violations) == 0, violations

    def _determine_risk_level(self, drawdown: float, volatility: float) -> RiskLevel:
        """Determine current risk level."""
        if drawdown >= 0.08 or volatility > 0.40:
            return RiskLevel.CRITICAL
        elif drawdown >= 0.05 or volatility > 0.30:
            return RiskLevel.HIGH
        elif drawdown >= 0.03 or volatility > 0.20:
            return RiskLevel.ELEVATED
        else:
            return RiskLevel.NORMAL

    def check_exit_conditions(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        quantity: int,
        entry_time: datetime,
        signal: float
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Returns:
            Tuple of (should_exit, reason)
        """
        if quantity == 0:
            return False, ""

        is_long = quantity > 0

        # Calculate P&L
        if is_long:
            pnl_pct = (current_price / entry_price - 1) * 100
        else:
            pnl_pct = (entry_price / current_price - 1) * 100

        holding_days = (datetime.now() - entry_time).days

        # Stop loss: -5%
        if pnl_pct < -5:
            return True, f"Stop loss ({pnl_pct:+.1f}%)"

        # Take profit: +15%
        if pnl_pct > 15:
            return True, f"Take profit ({pnl_pct:+.1f}%)"

        # Signal reversal (strong)
        if is_long and signal < -0.40:
            return True, f"Signal reversal ({signal:.2f})"
        if not is_long and signal > 0.40:
            return True, f"Signal reversal ({signal:.2f})"

        # Time-based exit (weak signal after holding period)
        if holding_days > 10:
            if is_long and signal < -0.20:
                return True, f"Weak signal after {holding_days}d"
            if not is_long and signal > 0.20:
                return True, f"Weak signal after {holding_days}d"

        # Max holding period
        if holding_days > 30:
            return True, f"Max holding ({holding_days}d)"

        return False, ""

    # ═══════════════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════════════

    def get_risk_report(self) -> Dict:
        """Generate risk report for monitoring."""
        daily_return = 0.0
        if self.daily_start_capital > 0:
            daily_return = (self.current_capital - self.daily_start_capital) / self.daily_start_capital

        total_return = 0.0
        if self.starting_capital > 0:
            total_return = (self.current_capital - self.starting_capital) / self.starting_capital

        drawdown = 0.0
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        return {
            'timestamp': datetime.now().isoformat(),
            'capital': {
                'starting': self.starting_capital,
                'current': self.current_capital,
                'peak': self.peak_capital,
                'daily_start': self.daily_start_capital
            },
            'returns': {
                'daily': daily_return,
                'total': total_return,
                'daily_pnl': self.daily_pnl
            },
            'risk': {
                'drawdown': drawdown,
                'drawdown_pct': drawdown * 100,
                'risk_multiplier': self._get_drawdown_multiplier(),
                'consecutive_losses': self.consecutive_losses
            },
            'circuit_breaker': {
                'state': self.cb_state.value,
                'trip_time': self.cb_trip_time.isoformat() if self.cb_trip_time else None
            },
            'config': {
                'max_position_pct': self.config.max_position_pct,
                'max_sector_pct': self.config.max_sector_pct,
                'target_vol': self.config.target_portfolio_vol,
                'max_drawdown': self.config.max_drawdown_pct
            }
        }


def create_risk_manager(
    config: Optional[Dict] = None
) -> InstitutionalRiskManager:
    """Factory function for risk manager."""
    if config:
        risk_config = RiskConfig(**config)
    else:
        risk_config = RiskConfig()

    return InstitutionalRiskManager(config=risk_config)
