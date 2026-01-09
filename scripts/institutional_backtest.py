"""
scripts/institutional_backtest.py

Institutional-Grade Walk-Forward Backtester

Key Features:
- Walk-forward optimization (rolling training/test windows)
- Realistic transaction cost modeling
- Slippage based on volatility and volume
- Proper performance attribution
- Out-of-sample validation
- Statistical significance testing

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
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.institutional_signal_generator import InstitutionalSignalGenerator, SignalOutput
from risk.institutional_risk_manager import InstitutionalRiskManager, RiskConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    # Time windows
    train_window_days: int = 252      # 1 year training
    test_window_days: int = 63        # 3 months testing
    warmup_days: int = 60             # Warmup for indicators

    # Walk-forward
    n_walks: int = 4                  # Number of walk-forward periods
    retrain_frequency: str = "quarterly"  # quarterly, monthly, weekly

    # Position management
    initial_capital: float = 1_000_000
    max_positions: int = 15
    position_size_pct: float = 0.05   # 5% per position

    # Transaction costs
    commission_per_share: float = 0.005  # $0.005/share
    min_commission: float = 1.0          # $1 minimum
    slippage_bps: float = 5.0            # 5 basis points base
    market_impact_bps: float = 2.0       # Additional for size

    # Signal thresholds
    entry_threshold: float = 0.40
    exit_threshold: float = -0.25
    min_confidence: float = 0.35

    # Risk management
    stop_loss_pct: float = 0.05       # 5%
    take_profit_pct: float = 0.15     # 15%
    max_holding_days: int = 30


@dataclass
class Trade:
    """Single trade record."""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # "LONG" or "SHORT"

    # Costs
    entry_commission: float
    exit_commission: float
    entry_slippage: float
    exit_slippage: float

    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0

    # Signal info
    entry_signal: float = 0.0
    entry_confidence: float = 0.0
    exit_reason: str = ""

    @property
    def total_costs(self) -> float:
        return self.entry_commission + self.exit_commission + self.entry_slippage + self.exit_slippage

    @property
    def holding_days(self) -> int:
        if self.exit_date:
            return (self.exit_date - self.entry_date).days
        return 0


@dataclass
class WalkForwardResult:
    """Result of a single walk-forward period."""
    period_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Performance
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade statistics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Costs
    total_commission: float
    total_slippage: float

    # Model performance
    directional_accuracy: float


@dataclass
class BacktestResult:
    """Complete backtest result."""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime

    # Equity curve
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int

    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_holding_days: float

    # Costs
    total_commission: float
    total_slippage: float
    cost_drag: float  # Annualized cost as % of capital

    # Walk-forward results
    walk_forward_results: List[WalkForwardResult]

    # All trades
    trades: List[Trade]

    # Statistical tests
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05


class InstitutionalBacktester:
    """
    Walk-forward backtester with institutional-grade metrics.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.signal_generator = InstitutionalSignalGenerator()
        self.risk_manager = InstitutionalRiskManager()

        # State
        self.positions: Dict[str, Trade] = {}  # Open positions
        self.closed_trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.cash = self.config.initial_capital

        logger.info("InstitutionalBacktester initialized")
        logger.info(f"  Initial capital: ${self.config.initial_capital:,.0f}")
        logger.info(f"  Train window: {self.config.train_window_days} days")
        logger.info(f"  Test window: {self.config.test_window_days} days")

    def run(self, historical_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            historical_data: Dict of symbol -> OHLCV DataFrame with DatetimeIndex

        Returns:
            BacktestResult with all metrics
        """
        logger.info("=" * 60)
        logger.info("WALK-FORWARD BACKTEST")
        logger.info("=" * 60)

        # Get date range
        all_dates = set()
        for data in historical_data.values():
            all_dates.update(data.index.tolist())

        all_dates = sorted(all_dates)
        if len(all_dates) < self.config.train_window_days + self.config.test_window_days:
            raise ValueError("Insufficient historical data")

        start_date = all_dates[0]
        end_date = all_dates[-1]

        logger.info(f"Data range: {start_date} to {end_date}")
        logger.info(f"Total days: {len(all_dates)}")

        # Calculate walk-forward periods
        walk_periods = self._calculate_walk_periods(all_dates)
        logger.info(f"Walk-forward periods: {len(walk_periods)}")

        walk_results = []

        for period_id, (train_start, train_end, test_start, test_end) in enumerate(walk_periods):
            logger.info(f"\n--- Walk {period_id + 1}/{len(walk_periods)} ---")
            logger.info(f"Train: {train_start.date()} to {train_end.date()}")
            logger.info(f"Test:  {test_start.date()} to {test_end.date()}")

            # Get training data
            train_data = self._slice_data(historical_data, train_start, train_end)

            # Train models
            logger.info("Training models...")
            self.signal_generator.train(train_data)

            # Get test data
            test_data = self._slice_data(historical_data, test_start, test_end)

            # Run backtest on test period
            result = self._run_period(test_data, period_id)
            walk_results.append(result)

            logger.info(f"Period return: {result.total_return:.2%}")
            logger.info(f"Sharpe: {result.sharpe_ratio:.2f}")
            logger.info(f"Trades: {result.num_trades}")

        # Aggregate results
        return self._aggregate_results(walk_results, start_date, end_date)

    def _calculate_walk_periods(
        self,
        dates: List[datetime]
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Calculate walk-forward periods."""
        periods = []

        # Start after warmup + train window
        min_start = self.config.warmup_days + self.config.train_window_days

        for i in range(self.config.n_walks):
            # Test start index
            test_start_idx = min_start + i * self.config.test_window_days

            if test_start_idx + self.config.test_window_days > len(dates):
                break

            train_end_idx = test_start_idx - 1
            train_start_idx = train_end_idx - self.config.train_window_days

            test_end_idx = min(
                test_start_idx + self.config.test_window_days - 1,
                len(dates) - 1
            )

            periods.append((
                dates[train_start_idx],
                dates[train_end_idx],
                dates[test_start_idx],
                dates[test_end_idx]
            ))

        return periods

    def _slice_data(
        self,
        data: Dict[str, pd.DataFrame],
        start: datetime,
        end: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Slice data to date range."""
        sliced = {}
        for symbol, df in data.items():
            mask = (df.index >= start) & (df.index <= end)
            if mask.any():
                sliced[symbol] = df[mask].copy()
        return sliced

    def _run_period(
        self,
        data: Dict[str, pd.DataFrame],
        period_id: int
    ) -> WalkForwardResult:
        """Run backtest for a single period."""
        # Reset state
        self.positions = {}
        period_trades: List[Trade] = []
        period_equity = []

        # Get all dates in period
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if len(dates) == 0:
            return self._empty_result(period_id)

        # Initialize
        current_cash = self.cash
        starting_equity = current_cash

        for date in dates:
            # Get prices for this date
            prices = {}
            for symbol, df in data.items():
                if date in df.index:
                    prices[symbol] = float(df.loc[date, 'Close'])

            # Update position values
            position_value = 0.0
            for symbol, trade in list(self.positions.items()):
                if symbol in prices:
                    price = prices[symbol]

                    # Check exit conditions
                    should_exit, reason = self._check_exit(trade, price, date, data)

                    if should_exit:
                        closed = self._close_position(trade, price, date, reason)
                        period_trades.append(closed)
                        del self.positions[symbol]
                        current_cash += closed.net_pnl + trade.entry_price * trade.quantity
                    else:
                        if trade.side == "LONG":
                            position_value += trade.quantity * price
                        else:
                            position_value += trade.quantity * (2 * trade.entry_price - price)

            # Generate signals and potentially open positions
            if len(self.positions) < self.config.max_positions:
                for symbol, df in data.items():
                    if symbol in self.positions:
                        continue
                    if date not in df.index:
                        continue

                    # Get price history up to this date
                    hist = df.loc[:date, 'Close']
                    if len(hist) < self.config.warmup_days:
                        continue

                    # Generate signal
                    signal_output = self.signal_generator.generate_signal(symbol, hist)

                    # Check entry
                    if (abs(signal_output.signal) >= self.config.entry_threshold and
                        signal_output.confidence >= self.config.min_confidence):

                        price = prices[symbol]
                        trade = self._open_position(
                            symbol, price, date, signal_output
                        )

                        if trade:
                            self.positions[symbol] = trade
                            current_cash -= trade.entry_price * trade.quantity + trade.entry_commission + trade.entry_slippage

                            if len(self.positions) >= self.config.max_positions:
                                break

            # Record equity
            total_equity = current_cash + position_value
            period_equity.append((date, total_equity))

        # Close remaining positions at end
        for symbol, trade in list(self.positions.items()):
            if symbol in prices:
                closed = self._close_position(trade, prices[symbol], dates[-1], "Period end")
                period_trades.append(closed)

        self.positions = {}

        # Store trades
        self.closed_trades.extend(period_trades)

        # Calculate metrics
        return self._calculate_period_metrics(
            period_id,
            dates[0], dates[-1],
            period_equity,
            period_trades,
            starting_equity
        )

    def _open_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        signal: SignalOutput
    ) -> Optional[Trade]:
        """Open a new position."""
        # Calculate position size
        position_value = self.cash * self.config.position_size_pct
        quantity = int(position_value / price)

        if quantity < 1:
            return None

        side = "LONG" if signal.signal > 0 else "SHORT"

        # Calculate costs
        commission = max(
            quantity * self.config.commission_per_share,
            self.config.min_commission
        )

        slippage_pct = (self.config.slippage_bps + self.config.market_impact_bps) / 10000
        slippage = position_value * slippage_pct

        return Trade(
            symbol=symbol,
            entry_date=date,
            exit_date=None,
            entry_price=price,
            exit_price=None,
            quantity=quantity,
            side=side,
            entry_commission=commission,
            exit_commission=0.0,
            entry_slippage=slippage,
            exit_slippage=0.0,
            entry_signal=signal.signal,
            entry_confidence=signal.confidence
        )

    def _check_exit(
        self,
        trade: Trade,
        current_price: float,
        date: datetime,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[bool, str]:
        """Check exit conditions."""
        holding_days = (date - trade.entry_date).days

        # Calculate P&L
        if trade.side == "LONG":
            pnl_pct = (current_price / trade.entry_price - 1)
        else:
            pnl_pct = (trade.entry_price / current_price - 1)

        # Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return True, "Stop loss"

        # Take profit
        if pnl_pct >= self.config.take_profit_pct:
            return True, "Take profit"

        # Max holding
        if holding_days >= self.config.max_holding_days:
            return True, "Max holding"

        # Signal reversal
        if trade.symbol in data:
            df = data[trade.symbol]
            if date in df.index:
                hist = df.loc[:date, 'Close']
                if len(hist) >= self.config.warmup_days:
                    signal = self.signal_generator.generate_signal(trade.symbol, hist)

                    if trade.side == "LONG" and signal.signal < self.config.exit_threshold:
                        return True, "Signal reversal"
                    if trade.side == "SHORT" and signal.signal > -self.config.exit_threshold:
                        return True, "Signal reversal"

        return False, ""

    def _close_position(
        self,
        trade: Trade,
        price: float,
        date: datetime,
        reason: str
    ) -> Trade:
        """Close position and calculate P&L."""
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_reason = reason

        # Exit costs
        trade.exit_commission = max(
            trade.quantity * self.config.commission_per_share,
            self.config.min_commission
        )

        position_value = trade.quantity * price
        slippage_pct = (self.config.slippage_bps + self.config.market_impact_bps) / 10000
        trade.exit_slippage = position_value * slippage_pct

        # Calculate P&L
        if trade.side == "LONG":
            trade.gross_pnl = (price - trade.entry_price) * trade.quantity
        else:
            trade.gross_pnl = (trade.entry_price - price) * trade.quantity

        trade.net_pnl = trade.gross_pnl - trade.total_costs
        trade.return_pct = trade.net_pnl / (trade.entry_price * trade.quantity) if trade.entry_price > 0 else 0

        return trade

    def _calculate_period_metrics(
        self,
        period_id: int,
        test_start: datetime,
        test_end: datetime,
        equity: List[Tuple[datetime, float]],
        trades: List[Trade],
        starting_equity: float
    ) -> WalkForwardResult:
        """Calculate metrics for a period."""
        if not equity:
            return self._empty_result(period_id)

        # Equity curve
        equity_values = [e[1] for e in equity]
        final_equity = equity_values[-1]

        # Returns
        total_return = (final_equity / starting_equity - 1) if starting_equity > 0 else 0
        days = (test_end - test_start).days
        annualized_return = total_return * (252 / max(days, 1))

        # Calculate daily returns for Sharpe/Sortino
        daily_returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else []

        sharpe = 0.0
        sortino = 0.0

        if len(daily_returns) > 5:
            excess_returns = daily_returns - 0.02/252  # Risk-free
            if np.std(excess_returns) > 0:
                sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

            downside = excess_returns[excess_returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                sortino = np.mean(excess_returns) / np.std(downside) * np.sqrt(252)

        # Drawdown
        peak = equity_values[0]
        max_dd = 0.0
        for val in equity_values:
            peak = max(peak, val)
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        calmar = annualized_return / max_dd if max_dd > 0 else 0

        # Trade statistics
        num_trades = len(trades)
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]

        win_rate = len(winners) / num_trades if num_trades > 0 else 0

        total_wins = sum(t.net_pnl for t in winners)
        total_losses = abs(sum(t.net_pnl for t in losers))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.net_pnl for t in losers]) if losers else 0

        # Costs
        total_commission = sum(t.entry_commission + t.exit_commission for t in trades)
        total_slippage = sum(t.entry_slippage + t.exit_slippage for t in trades)

        # Directional accuracy
        correct = sum(1 for t in trades if (t.side == "LONG" and t.gross_pnl > 0) or (t.side == "SHORT" and t.gross_pnl > 0))
        dir_acc = correct / num_trades if num_trades > 0 else 0.5

        return WalkForwardResult(
            period_id=period_id,
            train_start=test_start - timedelta(days=self.config.train_window_days),
            train_end=test_start - timedelta(days=1),
            test_start=test_start,
            test_end=test_end,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_commission=total_commission,
            total_slippage=total_slippage,
            directional_accuracy=dir_acc
        )

    def _empty_result(self, period_id: int) -> WalkForwardResult:
        """Return empty result."""
        now = datetime.now()
        return WalkForwardResult(
            period_id=period_id,
            train_start=now,
            train_end=now,
            test_start=now,
            test_end=now,
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            calmar_ratio=0,
            num_trades=0,
            win_rate=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            total_commission=0,
            total_slippage=0,
            directional_accuracy=0.5
        )

    def _aggregate_results(
        self,
        walk_results: List[WalkForwardResult],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Aggregate walk-forward results."""
        # Build equity curve from all periods
        equity_data = []
        for date, value in self.equity_history:
            equity_data.append({'date': date, 'equity': value})

        if not equity_data:
            equity_data = [{'date': start_date, 'equity': self.config.initial_capital}]

        equity_df = pd.DataFrame(equity_data).set_index('date')

        # Calculate drawdown curve
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak']

        # Aggregate metrics
        all_trades = self.closed_trades
        num_trades = len(all_trades)

        # Overall returns
        starting = self.config.initial_capital
        ending = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else starting

        total_return = (ending / starting - 1) if starting > 0 else 0
        days = (end_date - start_date).days
        ann_return = total_return * (252 / max(days, 1))

        # Max drawdown
        max_dd = equity_df['drawdown'].max() if len(equity_df) > 0 else 0

        # Drawdown duration
        in_dd = equity_df['drawdown'] > 0
        dd_groups = (in_dd != in_dd.shift()).cumsum()
        dd_durations = in_dd.groupby(dd_groups).sum()
        max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0

        # Sharpe/Sortino from walk results
        sharpe = np.mean([r.sharpe_ratio for r in walk_results]) if walk_results else 0
        sortino = np.mean([r.sortino_ratio for r in walk_results]) if walk_results else 0
        calmar = ann_return / max_dd if max_dd > 0 else 0

        # Trade statistics
        if num_trades > 0:
            winners = [t for t in all_trades if t.net_pnl > 0]
            losers = [t for t in all_trades if t.net_pnl <= 0]

            win_rate = len(winners) / num_trades
            total_wins = sum(t.net_pnl for t in winners)
            total_losses = abs(sum(t.net_pnl for t in losers))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            avg_return = np.mean([t.return_pct for t in all_trades])
            avg_holding = np.mean([t.holding_days for t in all_trades])
        else:
            win_rate = 0
            profit_factor = 0
            avg_return = 0
            avg_holding = 0

        # Costs
        total_commission = sum(t.entry_commission + t.exit_commission for t in all_trades)
        total_slippage = sum(t.entry_slippage + t.exit_slippage for t in all_trades)
        cost_drag = (total_commission + total_slippage) / starting * (252 / max(days, 1))

        # Statistical significance (t-test on returns)
        returns = [t.return_pct for t in all_trades]
        if len(returns) > 10:
            t_stat = np.mean(returns) / (np.std(returns) / np.sqrt(len(returns))) if np.std(returns) > 0 else 0
            from scipy import stats
            try:
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(returns) - 1))
            except:
                p_value = 1.0
        else:
            t_stat = 0
            p_value = 1.0

        return BacktestResult(
            config=self.config,
            start_date=start_date,
            end_date=end_date,
            equity_curve=equity_df[['equity']],
            drawdown_curve=equity_df[['drawdown']],
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_return,
            avg_holding_days=avg_holding,
            total_commission=total_commission,
            total_slippage=total_slippage,
            cost_drag=cost_drag,
            walk_forward_results=walk_results,
            trades=all_trades,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=p_value < 0.05
        )


def print_backtest_report(result: BacktestResult):
    """Print formatted backtest report."""
    print("\n" + "=" * 70)
    print("INSTITUTIONAL BACKTEST REPORT")
    print("=" * 70)

    print(f"\nPeriod: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Initial Capital: ${result.config.initial_capital:,.0f}")

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Total Return:      {result.total_return:+.2%}")
    print(f"Annualized Return: {result.annualized_return:+.2%}")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:     {result.sortino_ratio:.2f}")
    print(f"Calmar Ratio:      {result.calmar_ratio:.2f}")
    print(f"Max Drawdown:      {result.max_drawdown:.2%}")
    print(f"Max DD Duration:   {result.max_drawdown_duration} days")

    print("\n--- TRADE STATISTICS ---")
    print(f"Total Trades:      {result.total_trades}")
    print(f"Win Rate:          {result.win_rate:.1%}")
    print(f"Profit Factor:     {result.profit_factor:.2f}")
    print(f"Avg Trade Return:  {result.avg_trade_return:.2%}")
    print(f"Avg Holding Days:  {result.avg_holding_days:.1f}")

    print("\n--- TRANSACTION COSTS ---")
    print(f"Total Commission:  ${result.total_commission:,.2f}")
    print(f"Total Slippage:    ${result.total_slippage:,.2f}")
    print(f"Annual Cost Drag:  {result.cost_drag:.2%}")

    print("\n--- STATISTICAL SIGNIFICANCE ---")
    print(f"T-Statistic:       {result.t_statistic:.2f}")
    print(f"P-Value:           {result.p_value:.4f}")
    print(f"Significant (95%): {'YES' if result.is_significant else 'NO'}")

    print("\n--- WALK-FORWARD BREAKDOWN ---")
    for wf in result.walk_forward_results:
        print(f"  Period {wf.period_id + 1}: Return={wf.total_return:+.2%}, "
              f"Sharpe={wf.sharpe_ratio:.2f}, Trades={wf.num_trades}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    from data.market_data import MarketDataFetcher

    print("Loading data...")
    fetcher = MarketDataFetcher()

    # Load sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    data = {}

    for symbol in symbols:
        df = fetcher.fetch_historical_data(symbol, days=500)
        if not df.empty:
            data[symbol] = df
            print(f"  {symbol}: {len(df)} days")

    if data:
        config = BacktestConfig(
            train_window_days=200,
            test_window_days=50,
            n_walks=3,
            initial_capital=100_000
        )

        backtester = InstitutionalBacktester(config)
        result = backtester.run(data)
        print_backtest_report(result)
    else:
        print("No data loaded")
