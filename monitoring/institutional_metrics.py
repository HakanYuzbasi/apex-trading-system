"""
monitoring/institutional_metrics.py

Institutional-Grade Performance Metrics

Key Features:
- Professional risk-adjusted metrics (Sharpe, Sortino, Calmar)
- Drawdown analysis (depth, duration, recovery)
- Trade-level statistics
- Rolling performance windows
- Risk attribution

Author: Institutional Quant Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DrawdownAnalysis:
    """Detailed drawdown analysis."""
    max_drawdown: float
    current_drawdown: float
    max_drawdown_start: Optional[datetime]
    max_drawdown_end: Optional[datetime]
    max_drawdown_duration_days: int
    avg_drawdown: float
    drawdown_count: int  # Number of >5% drawdowns
    time_in_drawdown_pct: float
    recovery_factor: float  # Total return / max drawdown


@dataclass
class RiskMetrics:
    """Risk metrics."""
    volatility_annual: float
    downside_volatility: float
    var_95: float  # 1-day Value at Risk
    var_99: float
    expected_shortfall_95: float
    beta: float  # vs benchmark
    tracking_error: float
    information_ratio: float
    skewness: float
    kurtosis: float


@dataclass
class TradeMetrics:
    """Trade-level metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_return: float
    avg_winning_trade_return: float
    avg_losing_trade_return: float
    avg_holding_period_days: float
    win_loss_ratio: float  # avg_win / abs(avg_loss)
    expectancy: float  # Expected return per trade


@dataclass
class PerformanceReport:
    """Complete performance report."""
    # Period info
    start_date: datetime
    end_date: datetime
    trading_days: int

    # Returns
    total_return: float
    annualized_return: float
    cagr: float  # Compound annual growth rate
    monthly_returns: pd.Series

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Drawdown
    drawdown: DrawdownAnalysis

    # Risk
    risk: RiskMetrics

    # Trades
    trades: TradeMetrics

    # Rolling metrics
    rolling_sharpe_1y: Optional[pd.Series] = None
    rolling_volatility_20d: Optional[pd.Series] = None


class InstitutionalMetrics:
    """
    Calculate institutional-grade performance metrics.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trades: List[Dict] = []
        self.benchmark_returns: Optional[pd.Series] = None

    def record_equity(self, timestamp: datetime, value: float):
        """Record equity point."""
        self.equity_curve.append((timestamp, float(value)))

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        entry_date: datetime,
        exit_date: datetime,
        pnl: float,
        commission: float = 0.0
    ):
        """Record completed trade."""
        self.trades.append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'gross_pnl': pnl + commission,
            'net_pnl': pnl,
            'commission': commission,
            'holding_days': (exit_date - entry_date).days,
            'return_pct': pnl / (entry_price * quantity) if entry_price * quantity > 0 else 0
        })

    def set_benchmark(self, returns: pd.Series):
        """Set benchmark returns for comparison."""
        self.benchmark_returns = returns

    def generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        if len(self.equity_curve) < 2:
            return self._empty_report()

        # Build equity DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        equity_df.set_index('date', inplace=True)
        equity_df = equity_df.sort_index()

        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)

        # Period info
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        trading_days = len(equity_df)

        # Total return
        start_value = equity_df['equity'].iloc[0]
        end_value = equity_df['equity'].iloc[-1]
        total_return = (end_value / start_value - 1) if start_value > 0 else 0

        # Annualized return
        years = (end_date - start_date).days / 365.25
        ann_return = total_return * (252 / trading_days) if trading_days > 0 else 0
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Monthly returns
        monthly = equity_df['equity'].resample('M').last().pct_change().fillna(0)

        # Risk-adjusted metrics
        sharpe = self._calculate_sharpe(equity_df['returns'])
        sortino = self._calculate_sortino(equity_df['returns'])

        # Drawdown analysis
        dd_analysis = self._calculate_drawdown(equity_df['equity'])
        calmar = ann_return / dd_analysis.max_drawdown if dd_analysis.max_drawdown > 0 else 0

        # Omega ratio
        omega = self._calculate_omega(equity_df['returns'])

        # Risk metrics
        risk = self._calculate_risk_metrics(equity_df['returns'])

        # Trade metrics
        trade_metrics = self._calculate_trade_metrics()

        # Rolling metrics
        if len(equity_df) >= 252:
            rolling_sharpe = equity_df['returns'].rolling(252).apply(
                lambda x: self._calculate_sharpe(x), raw=False
            )
        else:
            rolling_sharpe = None

        rolling_vol = equity_df['returns'].rolling(20).std() * np.sqrt(252)

        return PerformanceReport(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            total_return=total_return,
            annualized_return=ann_return,
            cagr=cagr,
            monthly_returns=monthly,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            drawdown=dd_analysis,
            risk=risk,
            trades=trade_metrics,
            rolling_sharpe_1y=rolling_sharpe,
            rolling_volatility_20d=rolling_vol
        )

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 20:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252
        std = excess_returns.std()

        if std == 0 or np.isnan(std):
            return 0.0

        return float(excess_returns.mean() / std * np.sqrt(252))

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 20:
            return 0.0

        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        downside_std = downside_returns.std()

        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        return float(excess_returns.mean() / downside_std * np.sqrt(252))

    def _calculate_omega(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        if len(returns) < 20:
            return 1.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if losses.sum() == 0:
            return float('inf') if gains.sum() > 0 else 1.0

        return float(gains.sum() / losses.sum())

    def _calculate_drawdown(self, equity: pd.Series) -> DrawdownAnalysis:
        """Calculate detailed drawdown analysis."""
        peak = equity.cummax()
        drawdown = (equity - peak) / peak

        max_dd = float(drawdown.min())
        current_dd = float(drawdown.iloc[-1])
        avg_dd = float(drawdown.mean())

        # Find max drawdown period
        max_dd_end_idx = drawdown.idxmin()
        max_dd_start_idx = equity.loc[:max_dd_end_idx].idxmax()

        if pd.notna(max_dd_start_idx) and pd.notna(max_dd_end_idx):
            duration = (max_dd_end_idx - max_dd_start_idx).days
        else:
            duration = 0

        # Count significant drawdowns (>5%)
        dd_count = 0
        in_dd = False
        for dd in drawdown:
            if dd < -0.05 and not in_dd:
                dd_count += 1
                in_dd = True
            elif dd > -0.01:
                in_dd = False

        # Time in drawdown
        time_in_dd = (drawdown < -0.01).mean()

        # Recovery factor
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if equity.iloc[0] > 0 else 0
        recovery = total_return / abs(max_dd) if max_dd != 0 else 0

        return DrawdownAnalysis(
            max_drawdown=abs(max_dd),
            current_drawdown=abs(current_dd),
            max_drawdown_start=max_dd_start_idx if pd.notna(max_dd_start_idx) else None,
            max_drawdown_end=max_dd_end_idx if pd.notna(max_dd_end_idx) else None,
            max_drawdown_duration_days=duration,
            avg_drawdown=abs(avg_dd),
            drawdown_count=dd_count,
            time_in_drawdown_pct=float(time_in_dd),
            recovery_factor=float(recovery)
        )

    def _calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate risk metrics."""
        vol = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0

        # Downside volatility
        downside = returns[returns < 0]
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0

        # VaR (parametric)
        var_95 = float(np.percentile(returns, 5)) if len(returns) > 0 else 0
        var_99 = float(np.percentile(returns, 1)) if len(returns) > 0 else 0

        # Expected Shortfall (CVaR)
        es_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95

        # Higher moments
        skew = float(returns.skew()) if len(returns) > 2 else 0
        kurt = float(returns.kurtosis()) if len(returns) > 3 else 0

        # Benchmark metrics (if available)
        beta = 0.0
        tracking_error = 0.0
        info_ratio = 0.0

        if self.benchmark_returns is not None and len(self.benchmark_returns) > 0:
            aligned = returns.align(self.benchmark_returns, join='inner')
            if len(aligned[0]) > 20:
                cov = np.cov(aligned[0], aligned[1])
                var_benchmark = cov[1, 1]
                if var_benchmark > 0:
                    beta = float(cov[0, 1] / var_benchmark)

                excess = aligned[0] - aligned[1]
                tracking_error = float(excess.std() * np.sqrt(252))
                if tracking_error > 0:
                    info_ratio = float(excess.mean() * 252 / tracking_error)

        return RiskMetrics(
            volatility_annual=vol,
            downside_volatility=downside_vol,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=info_ratio,
            skewness=skew,
            kurtosis=kurt
        )

    def _calculate_trade_metrics(self) -> TradeMetrics:
        """Calculate trade-level metrics."""
        if not self.trades:
            return TradeMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                avg_trade_return=0,
                avg_winning_trade_return=0,
                avg_losing_trade_return=0,
                avg_holding_period_days=0,
                win_loss_ratio=0,
                expectancy=0
            )

        total = len(self.trades)
        winners = [t for t in self.trades if t['net_pnl'] > 0]
        losers = [t for t in self.trades if t['net_pnl'] <= 0]

        win_count = len(winners)
        loss_count = len(losers)
        win_rate = win_count / total if total > 0 else 0

        # P&L stats
        win_pnls = [t['net_pnl'] for t in winners]
        loss_pnls = [t['net_pnl'] for t in losers]

        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0
        largest_win = max(win_pnls) if win_pnls else 0
        largest_loss = min(loss_pnls) if loss_pnls else 0

        total_wins = sum(win_pnls)
        total_losses = abs(sum(loss_pnls))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Returns
        all_returns = [t['return_pct'] for t in self.trades]
        win_returns = [t['return_pct'] for t in winners]
        loss_returns = [t['return_pct'] for t in losers]

        avg_return = np.mean(all_returns) if all_returns else 0
        avg_win_return = np.mean(win_returns) if win_returns else 0
        avg_loss_return = np.mean(loss_returns) if loss_returns else 0

        # Holding period
        holding_days = [t['holding_days'] for t in self.trades]
        avg_holding = np.mean(holding_days) if holding_days else 0

        # Win/loss ratio
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return TradeMetrics(
            total_trades=total,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_return=avg_return,
            avg_winning_trade_return=avg_win_return,
            avg_losing_trade_return=avg_loss_return,
            avg_holding_period_days=avg_holding,
            win_loss_ratio=wl_ratio,
            expectancy=expectancy
        )

    def _empty_report(self) -> PerformanceReport:
        """Return empty report."""
        now = datetime.now()
        return PerformanceReport(
            start_date=now,
            end_date=now,
            trading_days=0,
            total_return=0,
            annualized_return=0,
            cagr=0,
            monthly_returns=pd.Series(),
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            omega_ratio=1,
            drawdown=DrawdownAnalysis(0, 0, None, None, 0, 0, 0, 0, 0),
            risk=RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            trades=TradeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )


def print_performance_report(report: PerformanceReport):
    """Print formatted performance report."""
    print("\n" + "=" * 70)
    print("PERFORMANCE REPORT")
    print("=" * 70)

    print(f"\nPeriod: {report.start_date.date()} to {report.end_date.date()}")
    print(f"Trading Days: {report.trading_days}")

    print("\n--- RETURNS ---")
    print(f"Total Return:      {report.total_return:+.2%}")
    print(f"Annualized Return: {report.annualized_return:+.2%}")
    print(f"CAGR:              {report.cagr:+.2%}")

    print("\n--- RISK-ADJUSTED ---")
    print(f"Sharpe Ratio:  {report.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {report.sortino_ratio:.2f}")
    print(f"Calmar Ratio:  {report.calmar_ratio:.2f}")
    print(f"Omega Ratio:   {report.omega_ratio:.2f}")

    print("\n--- DRAWDOWN ---")
    print(f"Max Drawdown:     {report.drawdown.max_drawdown:.2%}")
    print(f"Max DD Duration:  {report.drawdown.max_drawdown_duration_days} days")
    print(f"Current DD:       {report.drawdown.current_drawdown:.2%}")
    print(f"DD Count (>5%):   {report.drawdown.drawdown_count}")
    print(f"Time in DD:       {report.drawdown.time_in_drawdown_pct:.1%}")
    print(f"Recovery Factor:  {report.drawdown.recovery_factor:.2f}")

    print("\n--- RISK ---")
    print(f"Volatility:        {report.risk.volatility_annual:.2%}")
    print(f"Downside Vol:      {report.risk.downside_volatility:.2%}")
    print(f"VaR (95%, 1-day):  {report.risk.var_95:.2%}")
    print(f"Expected Shortfall:{report.risk.expected_shortfall_95:.2%}")
    print(f"Skewness:          {report.risk.skewness:.2f}")
    print(f"Kurtosis:          {report.risk.kurtosis:.2f}")

    print("\n--- TRADES ---")
    print(f"Total Trades:    {report.trades.total_trades}")
    print(f"Win Rate:        {report.trades.win_rate:.1%}")
    print(f"Profit Factor:   {report.trades.profit_factor:.2f}")
    print(f"Avg Win:         ${report.trades.avg_win:,.2f}")
    print(f"Avg Loss:        ${report.trades.avg_loss:,.2f}")
    print(f"W/L Ratio:       {report.trades.win_loss_ratio:.2f}")
    print(f"Expectancy:      ${report.trades.expectancy:,.2f}")
    print(f"Avg Holding:     {report.trades.avg_holding_period_days:.1f} days")

    print("\n" + "=" * 70)


# Convenience function
def calculate_metrics(
    equity_curve: List[Tuple[datetime, float]],
    trades: Optional[List[Dict]] = None,
    risk_free_rate: float = 0.02
) -> PerformanceReport:
    """Calculate all metrics from equity curve and trades."""
    metrics = InstitutionalMetrics(risk_free_rate=risk_free_rate)

    for timestamp, value in equity_curve:
        metrics.record_equity(timestamp, value)

    if trades:
        for trade in trades:
            metrics.record_trade(**trade)

    return metrics.generate_report()
