"""
risk/god_level_risk_manager.py - God Level Risk Management
Advanced risk management with:
- Dynamic ATR-based position sizing
- Kelly criterion optimization
- Correlation-aware exposure limits
- Adaptive stop-loss and take-profit
- Drawdown-based position reduction
- Sector and factor exposure management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class PositionRisk:
    """Risk metrics for a single position."""
    symbol: str
    entry_price: float
    current_price: float
    shares: int
    stop_loss: float
    take_profit: float
    trailing_stop: float
    atr: float
    risk_amount: float
    position_value: float
    pnl: float
    pnl_percent: float
    days_held: int
    max_favorable_excursion: float
    max_adverse_excursion: float


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_value: float
    cash: float
    invested: float
    daily_var: float  # Value at Risk
    max_drawdown: float
    current_drawdown: float
    sharpe_estimate: float
    correlation_risk: float
    sector_concentration: Dict[str, float]
    beta: float


class GodLevelRiskManager:
    """
    God-level risk management with adaptive position sizing,
    correlation management, and dynamic risk limits.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_pct: float = 0.05,
        max_portfolio_risk: float = 0.02,
        max_correlation: float = 0.7,
        max_sector_exposure: float = 0.30,
        max_drawdown_limit: float = 0.15
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital

        # Risk parameters
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.max_sector_exposure = max_sector_exposure
        self.max_drawdown_limit = max_drawdown_limit

        # Tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.historical_returns: List[float] = []
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.sector_map: Dict[str, str] = {}
        self.daily_pnl: List[float] = []

        # Adaptive parameters
        self.risk_multiplier = 1.0  # Reduces when in drawdown
        self.recent_win_rate = 0.5

        logger.info("God Level Risk Manager initialized")

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        signal_strength: float,
        confidence: float,
        prices: pd.Series,
        regime: str = "neutral"
    ) -> Dict:
        """
        Calculate optimal position size using multiple methods.

        Returns dict with:
        - shares: Recommended number of shares
        - position_value: Dollar value of position
        - stop_loss: Recommended stop loss price
        - take_profit: Recommended take profit price
        - risk_per_share: Dollar risk per share
        - risk_reward_ratio: Expected risk/reward
        """
        # Calculate ATR for volatility-adjusted sizing
        atr = self._calculate_atr(prices, 14)
        atr_pct = atr / entry_price if entry_price > 0 else 0.02

        # 1. ATR-based stop loss (adaptive to volatility)
        atr_multiplier = self._get_atr_multiplier(regime, signal_strength)
        stop_distance_pct = min(atr_pct * atr_multiplier, 0.08)  # Max 8% stop

        # 2. Calculate risk per share
        stop_loss = entry_price * (1 - stop_distance_pct) if signal_strength > 0 else entry_price * (1 + stop_distance_pct)
        risk_per_share = abs(entry_price - stop_loss)

        # 3. Kelly criterion for optimal bet size
        kelly_fraction = self._calculate_kelly_fraction(signal_strength, confidence)

        # 4. Volatility-adjusted position size
        vol_adjusted_size = self._volatility_adjusted_size(atr_pct)

        # 5. Correlation check
        correlation_penalty = self._check_correlation_limit(symbol, prices)

        # 6. Drawdown adjustment
        drawdown_multiplier = self._get_drawdown_multiplier()

        # 7. Regime adjustment
        regime_multiplier = self._get_regime_multiplier(regime)

        # Combine all factors
        base_risk_budget = self.current_capital * self.max_portfolio_risk
        adjusted_risk_budget = (
            base_risk_budget
            * kelly_fraction
            * vol_adjusted_size
            * correlation_penalty
            * drawdown_multiplier
            * regime_multiplier
            * self.risk_multiplier
        )

        # Calculate shares
        if risk_per_share > 0:
            shares = int(adjusted_risk_budget / risk_per_share)
        else:
            shares = 0

        # Apply position limits
        max_position_value = self.current_capital * self.max_position_pct
        max_shares_by_value = int(max_position_value / entry_price) if entry_price > 0 else 0
        shares = min(shares, max_shares_by_value)

        # Minimum viable position
        min_shares = 1
        shares = max(shares, min_shares) if shares > 0 else 0

        # Calculate take profit (risk-reward based)
        risk_reward_ratio = self._calculate_risk_reward(signal_strength, confidence, regime)
        profit_distance_pct = stop_distance_pct * risk_reward_ratio

        if signal_strength > 0:
            take_profit = entry_price * (1 + profit_distance_pct)
        else:
            take_profit = entry_price * (1 - profit_distance_pct)

        # Trailing stop (tighter for higher confidence)
        trailing_stop_pct = stop_distance_pct * (0.5 + 0.5 * (1 - confidence))

        position_value = shares * entry_price
        total_risk = shares * risk_per_share

        return {
            'shares': shares,
            'position_value': position_value,
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'trailing_stop_pct': round(trailing_stop_pct, 4),
            'risk_per_share': round(risk_per_share, 2),
            'total_risk': round(total_risk, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'atr': round(atr, 2),
            'atr_pct': round(atr_pct, 4),
            'kelly_fraction': round(kelly_fraction, 3),
            'position_pct': round(position_value / self.current_capital, 4) if self.current_capital > 0 else 0
        }

    def _calculate_atr(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(prices) < period + 1:
            return prices.iloc[-1] * 0.02  # Default 2%

        # Approximate ATR using close-only data
        daily_range = prices.diff().abs()
        atr = daily_range.rolling(period).mean().iloc[-1]

        return float(atr) if not np.isnan(atr) else prices.iloc[-1] * 0.02

    def calculate_stops_for_existing_position(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_qty: int,
        prices: pd.Series,
        regime: str = "neutral"
    ) -> Dict:
        """
        Calculate stop loss and take profit for an existing position.
        Used for positions loaded from IBKR that don't have stops set.

        Args:
            symbol: Stock symbol
            entry_price: Original entry price (or current if unknown)
            current_price: Current market price
            position_qty: Position quantity (positive=LONG, negative=SHORT)
            prices: Historical price series for ATR calculation
            regime: Current market regime

        Returns:
            Dict with stop_loss, take_profit, trailing_stop_pct
        """
        is_long = position_qty > 0

        # Calculate ATR
        atr = self._calculate_atr(prices, 14)
        atr_pct = atr / entry_price if entry_price > 0 else 0.02

        # Use conservative ATR multiplier for existing positions (we don't know original signal)
        atr_multiplier = self._get_atr_multiplier(regime, 0.5)  # Assume moderate signal
        stop_distance_pct = min(atr_pct * atr_multiplier, 0.08)  # Max 8% stop

        # Calculate stop loss based on position direction
        if is_long:
            # For LONG: stop below entry, but if price has moved up, trail from current
            base_stop = entry_price * (1 - stop_distance_pct)
            # If we're in profit, use trailing stop from current price
            if current_price > entry_price:
                trailing_stop = current_price * (1 - stop_distance_pct)
                stop_loss = max(base_stop, trailing_stop)
            else:
                stop_loss = base_stop
        else:
            # For SHORT: stop above entry, but if price has moved down, trail from current
            base_stop = entry_price * (1 + stop_distance_pct)
            if current_price < entry_price:
                trailing_stop = current_price * (1 + stop_distance_pct)
                stop_loss = min(base_stop, trailing_stop)
            else:
                stop_loss = base_stop

        # Calculate take profit (1.5x risk-reward for existing positions)
        risk_reward_ratio = 1.5
        profit_distance_pct = stop_distance_pct * risk_reward_ratio

        if is_long:
            take_profit = entry_price * (1 + profit_distance_pct)
        else:
            take_profit = entry_price * (1 - profit_distance_pct)

        # Trailing stop percentage
        trailing_stop_pct = stop_distance_pct * 0.75  # Tighter trailing

        logger.info(f"   📊 {symbol}: Set ATR-based stops - SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}, Trail: {trailing_stop_pct*100:.1f}%")

        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'trailing_stop_pct': round(trailing_stop_pct, 4),
            'atr': round(atr, 2),
            'atr_pct': round(atr_pct, 4)
        }

    def _get_atr_multiplier(self, regime: str, signal_strength: float) -> float:
        """Get ATR multiplier for stop loss based on regime and signal."""
        base_multiplier = {
            'strong_bull': 2.5,
            'bull': 2.0,
            'neutral': 1.5,
            'bear': 2.0,
            'strong_bear': 2.5,
            'high_volatility': 3.0
        }.get(regime, 2.0)

        # Tighter stops for weaker signals
        signal_adjustment = 0.5 + abs(signal_strength) * 0.5

        return base_multiplier * signal_adjustment

    def _calculate_kelly_fraction(self, signal_strength: float, confidence: float) -> float:
        """
        Calculate Kelly criterion fraction.
        Kelly = (W * p - L * q) / W
        Where:
        - p = probability of winning (confidence)
        - q = probability of losing (1 - p)
        - W = average win size
        - L = average loss size
        """
        # Estimate win probability from confidence
        win_prob = 0.5 + (confidence * abs(signal_strength)) * 0.3  # 50-80% range
        win_prob = min(max(win_prob, 0.4), 0.75)  # Clamp to realistic range

        # Estimate win/loss ratio from signal strength
        win_loss_ratio = 1.0 + abs(signal_strength) * 0.5  # 1.0-1.5 range

        # Kelly formula
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

        # Half-Kelly for safety (full Kelly is too aggressive)
        kelly = kelly * 0.5

        # Clamp to reasonable range
        return float(max(min(kelly, 0.25), 0.02))

    def _volatility_adjusted_size(self, atr_pct: float) -> float:
        """Reduce position size for high volatility stocks."""
        # Target: 2% daily volatility
        target_vol = 0.02

        if atr_pct <= 0:
            return 1.0

        vol_ratio = target_vol / atr_pct

        # Clamp adjustment
        return float(max(min(vol_ratio, 2.0), 0.25))

    def _check_correlation_limit(self, symbol: str, prices: pd.Series) -> float:
        """Check if adding this position would exceed correlation limits."""
        if self.correlation_matrix is None or symbol not in self.correlation_matrix.columns:
            return 1.0

        # Check correlation with existing positions
        for existing_symbol in self.positions.keys():
            if existing_symbol in self.correlation_matrix.columns:
                corr = self.correlation_matrix.loc[symbol, existing_symbol]
                if abs(corr) > self.max_correlation:
                    return 0.5  # Reduce size by 50% if highly correlated

        return 1.0

    def _get_drawdown_multiplier(self) -> float:
        """Reduce position size during drawdowns."""
        if self.peak_capital <= 0:
            return 1.0

        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        if current_drawdown <= 0.05:
            return 1.0
        elif current_drawdown <= 0.10:
            return 0.75
        elif current_drawdown <= 0.15:
            return 0.50
        else:
            return 0.25  # Severely reduce during large drawdowns

    def _get_regime_multiplier(self, regime: str) -> float:
        """Adjust position size based on market regime."""
        return {
            'strong_bull': 1.2,
            'bull': 1.1,
            'neutral': 0.9,
            'bear': 0.8,
            'strong_bear': 0.7,
            'high_volatility': 0.6
        }.get(regime, 1.0)

    def _calculate_risk_reward(self, signal_strength: float, confidence: float, regime: str) -> float:
        """Calculate target risk-reward ratio."""
        # Base R:R based on signal strength
        base_rr = 1.5 + abs(signal_strength)  # 1.5-2.5 range

        # Adjust for confidence
        confidence_adj = 1 + (confidence - 0.5) * 0.5  # 0.75-1.25 range

        # Adjust for regime
        regime_adj = {
            'strong_bull': 1.3,
            'bull': 1.2,
            'neutral': 1.0,
            'bear': 0.9,
            'strong_bear': 0.8,
            'high_volatility': 0.7
        }.get(regime, 1.0)

        return float(base_rr * confidence_adj * regime_adj)

    def update_correlation_matrix(self, historical_data: Dict[str, pd.DataFrame]):
        """Update correlation matrix from historical data."""
        returns_dict = {}

        for symbol, data in historical_data.items():
            if 'Close' in data.columns and len(data) >= 60:
                returns_dict[symbol] = data['Close'].pct_change().dropna()

        if len(returns_dict) >= 2:
            returns_df = pd.DataFrame(returns_dict)
            self.correlation_matrix = returns_df.corr()
            logger.info(f"Updated correlation matrix for {len(returns_dict)} symbols")

    def check_entry_allowed(
        self,
        symbol: str,
        sector: str,
        position_value: float
    ) -> Tuple[bool, str]:
        """
        Check if a new entry is allowed based on portfolio constraints.

        Returns (allowed, reason)
        """
        # 1. Check drawdown limit
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        if current_drawdown >= self.max_drawdown_limit:
            return False, f"Max drawdown limit reached ({current_drawdown:.1%})"

        # 2. Check sector exposure
        sector_exposure = self._calculate_sector_exposure(sector)
        if sector_exposure + position_value / self.current_capital > self.max_sector_exposure:
            return False, f"Sector exposure limit reached ({sector}: {sector_exposure:.1%})"

        # 3. Check total position count
        max_positions = 20
        if len(self.positions) >= max_positions:
            return False, f"Max positions limit reached ({len(self.positions)})"

        # 4. Check if already in position
        if symbol in self.positions:
            return False, f"Already in position for {symbol}"

        return True, "Entry allowed"

    def _calculate_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to a sector."""
        sector_value = 0.0

        for symbol, position in self.positions.items():
            if self.sector_map.get(symbol) == sector:
                sector_value += position.position_value

        return sector_value / self.current_capital if self.current_capital > 0 else 0

    def update_position(
        self,
        symbol: str,
        current_price: float,
        high_since_entry: float = None,
        low_since_entry: float = None
    ) -> Dict:
        """
        Update position with current price and return exit signals.

        Returns dict with:
        - exit_signal: bool
        - exit_reason: str
        - new_stop_loss: float (if trailing)
        """
        if symbol not in self.positions:
            return {'exit_signal': False, 'exit_reason': '', 'new_stop_loss': None}

        position = self.positions[symbol]
        position.current_price = current_price
        position.pnl = (current_price - position.entry_price) * position.shares
        position.pnl_percent = (current_price / position.entry_price - 1) * 100
        position.position_value = current_price * position.shares
        position.days_held += 1

        # Update excursions
        if high_since_entry:
            position.max_favorable_excursion = max(
                position.max_favorable_excursion,
                (high_since_entry / position.entry_price - 1) * 100
            )
        if low_since_entry:
            position.max_adverse_excursion = min(
                position.max_adverse_excursion,
                (low_since_entry / position.entry_price - 1) * 100
            )

        # Check exit conditions
        exit_signal = False
        exit_reason = ""
        new_stop_loss = None

        # 1. Stop loss hit
        if current_price <= position.stop_loss:
            exit_signal = True
            exit_reason = "Stop loss triggered"

        # 2. Take profit hit
        elif current_price >= position.take_profit:
            exit_signal = True
            exit_reason = "Take profit triggered"

        # 3. Trailing stop
        elif position.pnl_percent > 0:
            # Move stop up as price increases
            trailing_stop_price = current_price * (1 - position.trailing_stop)
            if trailing_stop_price > position.stop_loss:
                new_stop_loss = trailing_stop_price
                position.stop_loss = trailing_stop_price

            # Check if trailing stop hit
            if current_price <= position.stop_loss:
                exit_signal = True
                exit_reason = "Trailing stop triggered"

        # 4. Time-based exit (max hold period)
        max_hold_days = 60
        if position.days_held >= max_hold_days:
            exit_signal = True
            exit_reason = f"Max hold period ({max_hold_days} days)"

        return {
            'exit_signal': exit_signal,
            'exit_reason': exit_reason,
            'new_stop_loss': new_stop_loss,
            'pnl': position.pnl,
            'pnl_percent': position.pnl_percent
        }

    def add_position(self, position: PositionRisk):
        """Add a new position to tracking."""
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> Optional[PositionRisk]:
        """Remove and return a position."""
        return self.positions.pop(symbol, None)

    def update_capital(self, new_capital: float):
        """Update current capital and peak."""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)

    def record_trade(self, pnl: float, win: bool):
        """Record a completed trade for statistics."""
        self.daily_pnl.append(pnl)

        # Update recent win rate (rolling 20 trades)
        if len(self.daily_pnl) > 0:
            recent_trades = self.daily_pnl[-20:]
            wins = sum(1 for p in recent_trades if p > 0)
            self.recent_win_rate = wins / len(recent_trades)

        # Adjust risk multiplier based on recent performance
        if self.recent_win_rate < 0.4:
            self.risk_multiplier = max(0.5, self.risk_multiplier - 0.1)
        elif self.recent_win_rate > 0.6:
            self.risk_multiplier = min(1.5, self.risk_multiplier + 0.05)

    def get_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics."""
        total_invested = sum(p.position_value for p in self.positions.values())
        cash = self.current_capital - total_invested

        # Daily VaR (95%)
        if len(self.daily_pnl) >= 20:
            daily_var = np.percentile(self.daily_pnl, 5)
        else:
            daily_var = -self.current_capital * 0.02  # Assume 2%

        # Current drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0

        # Sector concentration
        sector_exposure = {}
        for symbol, position in self.positions.items():
            sector = self.sector_map.get(symbol, "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position.position_value

        for sector in sector_exposure:
            sector_exposure[sector] /= self.current_capital if self.current_capital > 0 else 1

        # Correlation risk (average pairwise correlation of positions)
        correlation_risk = 0.0
        if self.correlation_matrix is not None and len(self.positions) >= 2:
            pos_symbols = list(self.positions.keys())
            corrs = []
            for i, s1 in enumerate(pos_symbols):
                for s2 in pos_symbols[i+1:]:
                    if s1 in self.correlation_matrix.columns and s2 in self.correlation_matrix.columns:
                        corrs.append(abs(self.correlation_matrix.loc[s1, s2]))
            if corrs:
                correlation_risk = np.mean(corrs)

        # Sharpe estimate (annualized)
        if len(self.daily_pnl) >= 20:
            daily_returns = np.array(self.daily_pnl) / self.initial_capital
            sharpe_estimate = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe_estimate = 0

        return PortfolioRisk(
            total_value=self.current_capital,
            cash=cash,
            invested=total_invested,
            daily_var=daily_var,
            max_drawdown=current_drawdown,
            current_drawdown=current_drawdown,
            sharpe_estimate=sharpe_estimate,
            correlation_risk=correlation_risk,
            sector_concentration=sector_exposure,
            beta=1.0  # Would need benchmark data to calculate
        )

    def set_sector_map(self, sector_map: Dict[str, str]):
        """Set the sector mapping for symbols."""
        self.sector_map = sector_map
