"""
monitoring/advanced_performance_tracker.py - Professional Performance Analytics
Features:
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, etc.)
- Benchmark comparison (vs SPY, custom benchmarks)
- Risk-adjusted returns
- Trade analytics with slippage tracking
- Rolling performance windows
- Attribution analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger(__name__)


class AdvancedPerformanceTracker:
    """
    Professional-grade performance tracking and analytics.

    Tracks:
    - All trades with full details
    - Equity curve with timestamps
    - Benchmark returns for comparison
    - Risk metrics (volatility, drawdown, VaR)
    - Return metrics (Sharpe, Sortino, Calmar, etc.)
    """

    def __init__(self, risk_free_rate: float = 0.02, benchmark_symbol: str = "SPY"):
        """
        Initialize performance tracker.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            benchmark_symbol: Benchmark symbol for comparison (default: SPY)
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_symbol = benchmark_symbol

        # Trade history
        self.trades: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        # Benchmark data
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)

        # Statistics
        self.starting_capital = 0.0
        self.total_pnl = 0.0
        self.total_commissions = 0.0
        self.total_slippage = 0.0

        # Trade statistics
        self.winning_trades = 0
        self.losing_trades = 0
        self.breakeven_trades = 0

        logger.info(f"📊 Advanced Performance Tracker initialized")
        logger.info(f"   Benchmark: {benchmark_symbol}")

    def set_starting_capital(self, capital: float):
        """Set starting capital."""
        self.starting_capital = float(capital)
        self.record_equity(capital)

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        expected_price: Optional[float] = None,
        pnl: Optional[float] = None
    ):
        """
        Record a trade with full details.

        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost (difference from expected price)
            expected_price: Expected/quoted price
            pnl: P&L for closing trades
        """
        timestamp = datetime.now()

        # Calculate actual slippage if expected price provided
        if expected_price and expected_price > 0:
            if side == 'BUY':
                slippage = max(0, price - expected_price) * quantity
            else:  # SELL
                slippage = max(0, expected_price - price) * quantity

        trade = {
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': float(price),
            'commission': float(commission),
            'slippage': float(slippage),
            'expected_price': float(expected_price) if expected_price else None,
            'pnl': float(pnl) if pnl is not None else 0.0,
            'value': float(quantity * price)
        }

        self.trades.append(trade)
        self.total_commissions += commission
        self.total_slippage += slippage

        # Update trade statistics
        if pnl is not None:
            if pnl > 0:
                self.winning_trades += 1
            elif pnl < 0:
                self.losing_trades += 1
            else:
                self.breakeven_trades += 1

        logger.debug(f"Trade recorded: {side} {quantity} {symbol} @ ${price:.2f}")

    def record_equity(self, value: float):
        """Record equity point."""
        try:
            value = float(value)
            timestamp = datetime.now()
            self.equity_curve.append((timestamp, value))
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid equity value: {value} ({type(value)}): {e}")

    def load_benchmark_data(self, benchmark_returns: pd.Series):
        """
        Load benchmark returns for comparison.

        Args:
            benchmark_returns: Series of benchmark returns (typically SPY)
        """
        self.benchmark_returns = benchmark_returns
        logger.info(f"✅ Loaded {len(benchmark_returns)} benchmark returns")

    def get_returns_series(self) -> pd.Series:
        """Get returns series from equity curve."""
        if len(self.equity_curve) < 2:
            return pd.Series(dtype=float)

        # Extract values and timestamps
        timestamps, values = zip(*self.equity_curve)
        values = [float(v) for v in values]

        # Create Series
        series = pd.Series(values, index=timestamps)

        # Calculate returns
        returns = series.pct_change().dropna()

        return returns

    def get_sharpe_ratio(self, annualization_factor: float = 252) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (mean_return - risk_free_rate) / std_dev

        Args:
            annualization_factor: 252 for daily, 52 for weekly, 12 for monthly

        Returns:
            Annualized Sharpe ratio
        """
        returns = self.get_returns_series()

        if len(returns) < 2:
            return 0.0

        try:
            excess_returns = returns - (self.risk_free_rate / annualization_factor)
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(annualization_factor)
            return float(sharpe)

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def get_sortino_ratio(self, annualization_factor: float = 252) -> float:
        """
        Calculate Sortino ratio.

        Like Sharpe but only penalizes downside volatility.
        Sortino = (mean_return - risk_free_rate) / downside_std

        Args:
            annualization_factor: 252 for daily data

        Returns:
            Annualized Sortino ratio
        """
        returns = self.get_returns_series()

        if len(returns) < 2:
            return 0.0

        try:
            excess_returns = returns - (self.risk_free_rate / annualization_factor)

            # Downside deviation (only negative returns)
            downside_returns = returns[returns < 0]

            if len(downside_returns) == 0:
                return float('inf')  # No downside

            downside_std = downside_returns.std()

            if downside_std == 0:
                return 0.0

            sortino = excess_returns.mean() / downside_std * np.sqrt(annualization_factor)
            return float(sortino)

        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def get_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.

        Calmar = annualized_return / max_drawdown

        Returns:
            Calmar ratio
        """
        if len(self.equity_curve) < 2:
            return 0.0

        try:
            # Calculate annualized return
            start_value = float(self.equity_curve[0][1])
            end_value = float(self.equity_curve[-1][1])
            days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days

            if days == 0 or start_value == 0:
                return 0.0

            total_return = (end_value / start_value - 1)
            annualized_return = (1 + total_return) ** (365.0 / days) - 1

            # Calculate max drawdown
            max_dd = self.get_max_drawdown()

            if max_dd == 0:
                return 0.0

            calmar = annualized_return / max_dd
            return float(calmar)

        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def get_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        try:
            values = [float(v) for _, v in self.equity_curve]
            peak = values[0]
            max_dd = 0.0

            for value in values:
                if value > peak:
                    peak = value
                dd = (value - peak) / peak
                if dd < max_dd:
                    max_dd = dd

            return abs(max_dd)

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def get_win_rate(self) -> float:
        """Calculate win rate from completed trades."""
        total_completed = self.winning_trades + self.losing_trades + self.breakeven_trades

        if total_completed == 0:
            return 0.0

        return self.winning_trades / total_completed

    def get_profit_factor(self) -> float:
        """
        Calculate profit factor.

        Profit Factor = gross_profits / gross_losses

        Returns:
            Profit factor (>1 is profitable)
        """
        gross_profits = 0.0
        gross_losses = 0.0

        for trade in self.trades:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                gross_profits += pnl
            elif pnl < 0:
                gross_losses += abs(pnl)

        if gross_losses == 0:
            return float('inf') if gross_profits > 0 else 0.0

        return gross_profits / gross_losses

    def get_average_trade(self) -> Tuple[float, float, float]:
        """
        Get average trade statistics.

        Returns:
            Tuple of (avg_win, avg_loss, avg_trade)
        """
        wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in self.trades if t.get('pnl', 0) < 0]
        all_pnl = [t.get('pnl', 0) for t in self.trades]

        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_trade = np.mean(all_pnl) if all_pnl else 0.0

        return float(avg_win), float(avg_loss), float(avg_trade)

    def get_alpha_beta(self) -> Tuple[float, float]:
        """
        Calculate alpha and beta vs benchmark.

        Alpha: Excess return over benchmark (risk-adjusted)
        Beta: Sensitivity to benchmark movements

        Returns:
            Tuple of (alpha, beta)
        """
        if len(self.benchmark_returns) < 2:
            return 0.0, 0.0

        returns = self.get_returns_series()

        if len(returns) < 2:
            return 0.0, 0.0

        try:
            # Align returns
            common_index = returns.index.intersection(self.benchmark_returns.index)

            if len(common_index) < 2:
                return 0.0, 0.0

            portfolio_returns = returns.loc[common_index]
            benchmark_returns = self.benchmark_returns.loc[common_index]

            # Calculate beta using linear regression
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)

            if benchmark_variance == 0:
                return 0.0, 0.0

            beta = covariance / benchmark_variance

            # Calculate alpha
            portfolio_mean = portfolio_returns.mean()
            benchmark_mean = benchmark_returns.mean()
            alpha = portfolio_mean - (self.risk_free_rate / 252 + beta * (benchmark_mean - self.risk_free_rate / 252))

            # Annualize alpha
            alpha_annualized = alpha * 252

            return float(alpha_annualized), float(beta)

        except Exception as e:
            logger.error(f"Error calculating alpha/beta: {e}")
            return 0.0, 0.0

    def get_information_ratio(self) -> float:
        """
        Calculate Information Ratio.

        IR = (portfolio_return - benchmark_return) / tracking_error

        Measures consistency of excess returns.
        """
        if len(self.benchmark_returns) < 2:
            return 0.0

        returns = self.get_returns_series()

        if len(returns) < 2:
            return 0.0

        try:
            # Align returns
            common_index = returns.index.intersection(self.benchmark_returns.index)

            if len(common_index) < 2:
                return 0.0

            portfolio_returns = returns.loc[common_index]
            benchmark_returns = self.benchmark_returns.loc[common_index]

            # Calculate excess returns
            excess_returns = portfolio_returns - benchmark_returns

            # Information ratio
            if excess_returns.std() == 0:
                return 0.0

            ir = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

            return float(ir)

        except Exception as e:
            logger.error(f"Error calculating Information Ratio: {e}")
            return 0.0

    def get_slippage_analysis(self) -> Dict:
        """
        Analyze execution slippage.

        Returns:
            Dict with slippage statistics
        """
        trades_with_slippage = [t for t in self.trades if t.get('slippage', 0) != 0]

        if not trades_with_slippage:
            return {
                'total_slippage': 0.0,
                'avg_slippage_per_trade': 0.0,
                'max_slippage': 0.0,
                'slippage_as_pct_pnl': 0.0
            }

        slippages = [t['slippage'] for t in trades_with_slippage]

        return {
            'total_slippage': self.total_slippage,
            'avg_slippage_per_trade': np.mean(slippages),
            'max_slippage': np.max(slippages),
            'slippage_as_pct_pnl': (self.total_slippage / self.total_pnl * 100) if self.total_pnl != 0 else 0.0
        }

    def get_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive performance report.

        Returns:
            Dict with all performance metrics
        """
        if len(self.equity_curve) == 0:
            return {'error': 'No data available'}

        start_value = float(self.equity_curve[0][1])
        end_value = float(self.equity_curve[-1][1])
        total_return = (end_value / start_value - 1) * 100 if start_value > 0 else 0.0

        avg_win, avg_loss, avg_trade = self.get_average_trade()
        alpha, beta = self.get_alpha_beta()
        slippage_stats = self.get_slippage_analysis()

        return {
            # Capital
            'starting_capital': self.starting_capital,
            'ending_capital': end_value,
            'total_return_pct': total_return,
            'total_pnl': end_value - start_value,

            # Trade Statistics
            'total_trades': len(self.trades),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': self.get_win_rate() * 100,
            'profit_factor': self.get_profit_factor(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,

            # Risk Metrics
            'sharpe_ratio': self.get_sharpe_ratio(),
            'sortino_ratio': self.get_sortino_ratio(),
            'calmar_ratio': self.get_calmar_ratio(),
            'max_drawdown_pct': self.get_max_drawdown() * 100,

            # Benchmark Comparison
            'alpha': alpha,
            'beta': beta,
            'information_ratio': self.get_information_ratio(),

            # Costs
            'total_commissions': self.total_commissions,
            'total_slippage': self.total_slippage,
            'slippage_analysis': slippage_stats,

            # Metadata
            'start_date': self.equity_curve[0][0].isoformat(),
            'end_date': self.equity_curve[-1][0].isoformat(),
            'days_traded': (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
        }

    def print_summary(self):
        """Print comprehensive performance summary."""
        report = self.get_comprehensive_report()

        if 'error' in report:
            logger.info(f"⚠️ {report['error']}")
            return

        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 COMPREHENSIVE PERFORMANCE REPORT")
        logger.info("=" * 80)

        logger.info(f"\n💰 CAPITAL:")
        logger.info(f"   Starting: ${report['starting_capital']:,.2f}")
        logger.info(f"   Ending: ${report['ending_capital']:,.2f}")
        logger.info(f"   Total P&L: ${report['total_pnl']:+,.2f} ({report['total_return_pct']:+.2f}%)")

        logger.info(f"\n📈 TRADES:")
        logger.info(f"   Total: {report['total_trades']}")
        logger.info(f"   Winners: {report['winning_trades']} ({report['win_rate_pct']:.1f}%)")
        logger.info(f"   Losers: {report['losing_trades']}")
        logger.info(f"   Profit Factor: {report['profit_factor']:.2f}")
        logger.info(f"   Avg Win: ${report['avg_win']:,.2f}")
        logger.info(f"   Avg Loss: ${report['avg_loss']:,.2f}")

        logger.info(f"\n📊 RISK METRICS:")
        logger.info(f"   Sharpe Ratio: {report['sharpe_ratio']:.2f}")
        logger.info(f"   Sortino Ratio: {report['sortino_ratio']:.2f}")
        logger.info(f"   Calmar Ratio: {report['calmar_ratio']:.2f}")
        logger.info(f"   Max Drawdown: {report['max_drawdown_pct']:.2f}%")

        if report['alpha'] != 0.0 or report['beta'] != 0.0:
            logger.info(f"\n🎯 VS BENCHMARK ({self.benchmark_symbol}):")
            logger.info(f"   Alpha: {report['alpha']*100:+.2f}%")
            logger.info(f"   Beta: {report['beta']:.2f}")
            logger.info(f"   Information Ratio: {report['information_ratio']:.2f}")

        logger.info(f"\n💸 COSTS:")
        logger.info(f"   Commissions: ${report['total_commissions']:,.2f}")
        logger.info(f"   Slippage: ${report['total_slippage']:,.2f}")

        logger.info("")
        logger.info("=" * 80)
