"""
monitoring/performance_tracker.py
FIXED: Proper equity curve tracking with float conversion
"""

import logging
import math
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import collections
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track trading performance metrics."""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        history_filename: str = "performance_history.json",
    ):
        self.trades: collections.deque = collections.deque(maxlen=5000)
        self.equity_curve: collections.deque = collections.deque(maxlen=100000)  # ✅ Memory capped
        self.benchmark_curve: collections.deque = collections.deque(maxlen=100000) # 🎯 Phase 2
        self.starting_capital: float = 0.0
        self.data_dir = Path(data_dir) if data_dir is not None else Path("data")
        self.history_file = self.data_dir / str(history_filename)
        self._load_state()

    
    async def record_trade(self, symbol: str, side: str, quantity: int, price: float, commission: float = 0.0):
        """Record a trade with commission."""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': float(price),  # ✅ Force float
            'commission': float(commission),  # ✅ Force float
            'pnl': 0.0  # Calculated on exit
        }
        
        self.trades.append(trade)
        await self._save_state_async()  # ✅ Non-blocking save
        logger.debug(f"Trade recorded: {side} {quantity} {symbol} @ ${price:.2f}")
    
    async def record_benchmark(self, price: float, timestamp: str):
        """🎯 Phase 2: Stores the SPY benchmark price safely."""
        try:
            self.benchmark_curve.append((timestamp, float(price)))
            # No disk write here to save I/O; record_equity handles persistence.
        except Exception as e:
            logger.error(f"Error recording benchmark: {e}")
    
    async def record_equity(self, value: float):
        """✅ FIXED: Record equity point with proper float conversion and sanity guard."""
        try:
            value = float(value)  # ✅ Force conversion
            
            # 🛡️ Sanity Guard: Prevent recording massive transient drops
            if self.equity_curve:
                last_value = self.equity_curve[-1][1]
                if last_value > 0:
                    drop_pct = (last_value - value) / last_value
                    if drop_pct > 0.5:
                        logger.warning(
                            f"🛑 record_equity: Skipped outlier value ${value:,.2f} "
                            f"(drop of {drop_pct:.1%} from ${last_value:,.2f})"
                        )
                        return

            timestamp = datetime.now().isoformat()
            self.equity_curve.append((timestamp, value))
            await self._save_state_async()  # ✅ Non-blocking save
            logger.debug(f"Equity recorded: ${value:,.2f}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid equity value: {value} ({type(value)}): {e}")
    
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol for matching: strip CRYPTO:/FX: prefix, upper-case."""
        s = str(symbol).upper()
        for prefix in ("CRYPTO:", "FX:"):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        return s

    def _daily_close_values(self) -> List[float]:
        """Resample the sub-minute equity curve to one value per calendar day (last observation).

        Equity is recorded at ~1-second frequency.  Using raw observations inflates the
        number of periods-per-year far above 252, making √252 annualization produce
        numerically meaningless results.  Reducing to daily closes ensures every
        Sharpe/Sortino/Calmar calculation uses the same frequency assumption as the
        kill-switch (which has the same fix applied independently).
        """
        day_map: dict = {}
        for ts_str, val in self.equity_curve:
            try:
                day = str(ts_str)[:10]   # "YYYY-MM-DD" prefix of ISO timestamp
                day_map[day] = float(val)  # last write per day = daily close
            except Exception:
                continue
        return [day_map[d] for d in sorted(day_map)]

    def _hourly_close_values(self) -> List[float]:
        """Resample equity curve to hourly closes. Used when < 2 daily periods available."""
        hour_map: dict = {}
        for ts_str, val in self.equity_curve:
            try:
                key = str(ts_str)[:13]  # "YYYY-MM-DDTHH" prefix
                hour_map[key] = float(val)
            except Exception:
                continue
        return [hour_map[h] for h in sorted(hour_map)]

    def get_trade_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """Sharpe computed from closed trade returns (not equity curve ticks).

        Used as the intraday fallback (< 2 calendar days of equity data).
        Annualises assuming 252 trading days / avg_holding_days trades per year.
        Returns 0.0 when fewer than 3 closed trades are available.
        """
        try:
            positions: Dict = {}
            trade_returns: List[float] = []
            holding_days: List[float] = []

            for trade in self.trades:
                symbol = self._normalize_symbol(trade['symbol'])
                side = str(trade.get('side', '')).upper()
                qty = float(trade['quantity'])
                price = float(trade['price'])
                commission = float(trade.get('commission', 0))
                ts = trade.get('timestamp')

                if side == 'BUY':
                    positions.setdefault(symbol, []).append({
                        'qty': qty, 'price': price,
                        'commission': commission, 'timestamp': ts,
                    })
                elif side == 'SELL' and positions.get(symbol):
                    entry = positions[symbol].pop(0)
                    gross_pnl = (price - entry['price']) * qty
                    net_pnl = gross_pnl - entry['commission'] - commission
                    cost = entry['price'] * qty
                    if cost > 1e-9:
                        trade_returns.append(net_pnl / cost)
                    # Holding period in days
                    try:
                        if entry['timestamp'] and ts:
                            t0 = pd.Timestamp(entry['timestamp'])
                            t1 = pd.Timestamp(ts)
                            days = max((t1 - t0).total_seconds() / 86400.0, 1 / 1440.0)
                            holding_days.append(days)
                    except Exception:
                        pass

            if len(trade_returns) < 3:
                return 0.0

            arr = np.array(trade_returns, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) < 3:
                return 0.0

            # Annualization: derive trades-per-year from avg holding time
            if holding_days:
                avg_days = max(float(np.mean(holding_days)), 1 / 1440.0)
                trades_per_year = min(252.0 / avg_days, 2520.0)  # cap at 10 trades/day
            else:
                trades_per_year = 252.0  # assume ~1 trade/day if no timing info

            excess = arr - (risk_free_rate / trades_per_year)
            vol = float(np.std(excess))
            if vol < 1e-9:
                return 0.0
            return float(np.clip(
                np.mean(excess) / vol * math.sqrt(trades_per_year), -10.0, 10.0
            ))
        except Exception as e:
            logger.error("Error calculating trade Sharpe: %s", e)
            return 0.0

    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualised Sharpe ratio from equity samples.

        Uses daily closes when 2+ days of data are available.
        Falls back to trade-level Sharpe for intraday-only sessions (avoids the
        √(252×24) artefact that makes day-1 Sharpe look like -10 or +99).
        """
        daily = self._daily_close_values()
        if len(daily) >= 2:
            try:
                values = np.array(daily, dtype=float)
                returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
                returns = returns[np.isfinite(returns)]
                if len(returns) == 0 or np.max(np.abs(returns)) < 1e-9:
                    return 0.0
                excess = returns - (risk_free_rate / 252.0)
                vol = float(np.std(excess))
                if vol < 1e-9:
                    return 0.0
                return float(np.clip(np.mean(excess) / vol * math.sqrt(252.0), -10.0, 10.0))
            except Exception as e:
                logger.error("Error calculating Sharpe ratio: %s", e)
                return 0.0
        # Intraday-only: prefer trade-level Sharpe (meaningful) over hourly equity noise
        return self.get_trade_sharpe(risk_free_rate=risk_free_rate)
    
    def get_win_rate(self) -> float:
        """Calculate win rate from completed trades.

        Normalizes symbol keys (strips CRYPTO:/FX: prefix) so that an entry
        recorded as 'ETH/USD' matches an exit recorded as 'CRYPTO:ETH/USD'.
        """
        if len(self.trades) < 2:
            return 0.0

        try:
            # Match buys and sells using normalized symbol keys
            positions: Dict = {}
            completed_trades = []

            for trade in self.trades:
                symbol = self._normalize_symbol(trade['symbol'])
                side = str(trade.get('side', '')).upper()
                qty = trade['quantity']
                price = trade['price']
                commission = trade.get('commission', 0)

                if side == 'BUY':
                    positions.setdefault(symbol, []).append({
                        'qty': qty,
                        'price': price,
                        'commission': commission
                    })

                elif side == 'SELL':
                    if positions.get(symbol):
                        entry = positions[symbol].pop(0)
                        pnl = (price - entry['price']) * qty - entry['commission'] - commission
                        completed_trades.append({'pnl': pnl})

            if len(completed_trades) == 0:
                return 0.0

            winners = sum(1 for t in completed_trades if t['pnl'] > 0)
            return winners / len(completed_trades)

        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
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

    def get_completed_trade_count(self) -> int:
        """Count completed BUY->SELL round-trips (symbol-normalized)."""
        if len(self.trades) < 2:
            return 0
        try:
            positions: Dict = {}
            completed = 0
            for trade in self.trades:
                symbol = self._normalize_symbol(trade['symbol'])
                side = str(trade['side']).upper()
                qty = trade['quantity']
                if side == 'BUY':
                    positions.setdefault(symbol, []).append(qty)
                elif side == 'SELL' and positions.get(symbol):
                    positions[symbol].pop(0)
                    completed += 1
            return completed
        except Exception:
            return 0

    def get_calmar_ratio(self) -> float:
        """Calmar ratio: annualised return / max drawdown, using actual calendar days."""
        daily = self._daily_close_values()
        if len(daily) < 5:
            return 0.0
        try:
            total_return = (daily[-1] / max(daily[0], 1e-9)) - 1
            # n_days is the actual number of calendar days in the sample — not observation count
            n_days = max(len(daily), 1)
            annual_return = total_return * (252 / n_days)
            max_dd = self.get_max_drawdown()
            if max_dd == 0:
                return 0.0
            return float(annual_return / max_dd)
        except Exception as e:
            logger.error("Error calculating Calmar ratio: %s", e)
            return 0.0

    def _get_daily_returns(self, lookback: int = 252) -> np.ndarray:
        """
        Daily returns derived from the daily-close resampled equity curve.

        Using raw sub-minute equity observations here would produce ~86400× too many
        data points per day, making VaR/CVaR percentile cuts meaningless.
        """
        try:
            daily = self._daily_close_values()
            daily = daily[-(lookback + 1):]  # apply lookback window
            if len(daily) < 2:
                return np.array([])
            values = np.array(daily, dtype=float)
            denom = np.where(values[:-1] != 0, values[:-1], 1.0)
            returns = np.diff(values) / denom
            return returns[np.isfinite(returns)]
        except Exception as e:
            logger.error("Error computing daily returns for VaR: %s", e)
            return np.array([])

    def get_var(self, confidence: float = 0.95, lookback: int = 252) -> float:
        """
        Historical Simulation Value-at-Risk (1-day, percentage of equity).

        Returns the loss (as a negative fraction of equity, e.g. -0.023 = -2.3%)
        that is NOT exceeded with probability `confidence`.  Requires at least 20
        return observations; returns 0.0 when insufficient history exists.

        Example: get_var(0.95) → -0.018 means there is a 95% chance that the
        single-day loss will not exceed 1.8% of equity.
        """
        try:
            returns = self._get_daily_returns(lookback)
            if len(returns) < 20:
                return 0.0
            # (1 - confidence) percentile of the return distribution
            var = float(np.percentile(returns, (1.0 - confidence) * 100.0))
            return var if math.isfinite(var) else 0.0
        except Exception as e:
            logger.error("Error calculating VaR: %s", e)
            return 0.0

    def get_cvar(self, confidence: float = 0.95, lookback: int = 252) -> float:
        """
        Conditional VaR — Expected Shortfall (1-day, percentage of equity).

        Returns the *mean* return in the tail beyond VaR: the average of the worst
        (1 - confidence) fraction of daily returns.  Always ≤ VaR.  Requires at
        least 20 return observations; returns 0.0 when insufficient history exists.

        Example: get_cvar(0.95) → -0.031 means that on the 5% worst days, the
        average loss is 3.1% of equity.
        """
        try:
            returns = self._get_daily_returns(lookback)
            if len(returns) < 20:
                return 0.0
            var = self.get_var(confidence, lookback)
            tail = returns[returns <= var]
            if len(tail) == 0:
                return var
            cvar = float(np.mean(tail))
            return cvar if math.isfinite(cvar) else 0.0
        except Exception as e:
            logger.error("Error calculating CVaR: %s", e)
            return 0.0

    def get_alpha_retention(self) -> float:
        """🎯 Phase 2: Calculates (Strategy Return - Benchmark Return) / Abs(Benchmark Return)"""
        if len(self.equity_curve) < 2 or len(self.benchmark_curve) < 2:
            return 0.0
            
        strat_start, strat_end = self.equity_curve[0][1], self.equity_curve[-1][1]
        bench_start, bench_end = self.benchmark_curve[0][1], self.benchmark_curve[-1][1]
        
        if strat_start <= 0 or bench_start <= 0:
            return 0.0
            
        strat_ret = (strat_end - strat_start) / strat_start
        bench_ret = (bench_end - bench_start) / bench_start
        
        # Avoid division by zero if benchmark is perfectly flat
        if abs(bench_ret) < 1e-6:
            return 0.0
            
        return float((strat_ret - bench_ret) / abs(bench_ret))

    def get_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualised Sortino ratio computed from daily-close equity samples."""
        daily = self._daily_close_values()
        if len(daily) < 5:
            return 0.0
        try:
            values = np.array(daily, dtype=float)
            returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
            returns = returns[np.isfinite(returns)]
            if len(returns) == 0:
                return 0.0
            mean_excess = float(np.mean(returns)) - risk_free_rate / 252
            downside = returns[returns < 0]
            if len(downside) == 0:
                return 0.0
            downside_dev = float(np.std(downside))
            if downside_dev <= 1e-8:
                return 0.0
            return float(mean_excess / downside_dev * np.sqrt(252))
        except Exception as e:
            logger.error("Error calculating Sortino ratio: %s", e)
            return 0.0

    def get_profit_factor(self) -> float:
        """🎯 Phase 2: Gross Profit / Gross Loss"""
        if not self.trades:
            return 0.0
            
        gross_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        
        if gross_loss <= 1e-8:
            return float(gross_profit) if gross_profit > 0 else 0.0
            
        return float(gross_profit / gross_loss)

    def get_avg_trade_pnl(self) -> Tuple[float, float]:
        """Calculate average winning and losing trade P&L."""
        if len(self.trades) < 2:
            return (0.0, 0.0)

        try:
            winners = []
            losers = []
            positions = {}

            for trade in self.trades:
                symbol = trade['symbol']
                side = trade['side']
                qty = trade['quantity']
                price = trade['price']
                commission = trade.get('commission', 0)

                if side == 'BUY':
                    if symbol not in positions:
                        positions[symbol] = []
                    positions[symbol].append({'qty': qty, 'price': price, 'commission': commission})

                elif side == 'SELL':
                    if symbol in positions and len(positions[symbol]) > 0:
                        entry = positions[symbol].pop(0)
                        pnl = (price - entry['price']) * qty - entry['commission'] - commission

                        if pnl > 0:
                            winners.append(pnl)
                        else:
                            losers.append(pnl)

            avg_win = np.mean(winners) if winners else 0.0
            avg_loss = np.mean(losers) if losers else 0.0

            return (float(avg_win), float(avg_loss))

        except Exception as e:
            logger.error(f"Error calculating avg trade P&L: {e}")
            return (0.0, 0.0)
    
    def print_summary(self):
        """Print performance summary with full quant metrics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Total Trades: {len(self.trades)}")

        if len(self.trades) > 0:
            logger.info(f"Win Rate: {self.get_win_rate()*100:.1f}%")
            logger.info(f"Profit Factor: {self.get_profit_factor():.2f}")
            avg_win, avg_loss = self.get_avg_trade_pnl()
            logger.info(f"Avg Win: ${avg_win:+,.2f} | Avg Loss: ${avg_loss:+,.2f}")

        if len(self.equity_curve) > 1:
            start_value = float(self.equity_curve[0][1])
            end_value = float(self.equity_curve[-1][1])
            total_return = (end_value / start_value - 1) * 100

            logger.info(f"Starting Capital: ${start_value:,.2f}")
            logger.info(f"Ending Capital: ${end_value:,.2f}")
            logger.info(f"Total Return: {total_return:+.2f}%")
            logger.info("─" * 40)
            logger.info("📈 RISK-ADJUSTED METRICS:")
            logger.info(f"   Sharpe Ratio: {self.get_sharpe_ratio():.2f}")
            logger.info(f"   Sortino Ratio: {self.get_sortino_ratio():.2f}")
            logger.info(f"   Calmar Ratio: {self.get_calmar_ratio():.2f}")
            logger.info(f"   Max Drawdown: {self.get_max_drawdown()*100:.2f}%")

        logger.info("=" * 80)

    async def _save_state_async(self):
        """Asynchronously save state to disk using a thread to avoid blocking."""
        await asyncio.to_thread(self._save_state)

    def _save_state(self):
        """Save performance history to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'trades': list(self.trades),
                'equity_curve': list(self.equity_curve),
                'benchmark_curve': list(self.benchmark_curve),
                'starting_capital': self.starting_capital,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.history_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance state: {e}")

    def _clean_equity_curve(self, curve: list) -> list:
        """
        Remove transient outlier points that corrupt drawdown calculation.
        A point is an outlier if its value is < 10% OR > 1000% of the median
        of all values in the curve. This preserves legitimate multi-year trends
        while stripping one-off bad broker readings.
        """
        if len(curve) < 3:
            return curve
        try:
            values = [float(v) for _, v in curve]
            median_val = float(np.median(values))
            if median_val <= 0:
                return curve
            cleaned = [
                point for point, val in zip(curve, values)
                if 0.10 * median_val <= val <= 10.0 * median_val
            ]
            removed = len(curve) - len(cleaned)
            if removed > 0:
                logger.warning(
                    "🩹 Stripped %d outlier equity points from history "
                    "(median=$%.2f). Drawdown calculation is now clean.",
                    removed, median_val,
                )
            return cleaned
        except Exception as e:
            logger.error("Failed to clean equity curve: %s", e)
            return curve

    def _load_state(self):
        """Load performance history from disk."""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, 'r') as f:
                state = json.load(f)

            self.trades = collections.deque(state.get('trades', []), maxlen=5000)
            raw_curve = state.get('equity_curve', [])
            self.equity_curve = collections.deque(self._clean_equity_curve(raw_curve), maxlen=100000)
            self.benchmark_curve = collections.deque(state.get('benchmark_curve', []), maxlen=100000)
            self.starting_capital = state.get('starting_capital', 0.0)

            logger.info(f"📊 Restored {len(self.trades)} trades and {len(self.equity_curve)} equity points")
            # Persist the cleaned curve so outliers don't reappear on next load
            if len(self.equity_curve) != len(raw_curve):
                self._save_state()
        except Exception as e:
            logger.error(f"Failed to load performance state: {e}")

    async def reset_history(self, *, starting_capital: float, reason: str = "manual_reset"):
        """Reset persisted performance history and seed with a single baseline equity point."""
        try:
            capital = float(starting_capital)
        except Exception:
            logger.warning("Cannot reset performance history with invalid capital: %s", starting_capital)
            return

        if capital <= 0:
            logger.warning("Cannot reset performance history with non-positive capital: %s", capital)
            return

        self.trades.clear()
        self.equity_curve.clear()
        self.equity_curve.append((datetime.now().isoformat(), capital))
        self.benchmark_curve.clear()
        self.starting_capital = capital
        await self._save_state_async()
        logger.warning(
            "🩹 Performance history reset (%s). Seeded baseline equity: $%.2f",
            reason,
            capital,
        )

    async def rebase_baseline(
        self,
        *,
        starting_capital: float,
        reason: str = "runtime_rebase",
        reset_trades: bool = False,
        clear_benchmark: bool = False,
    ) -> None:
        """Append a new equity baseline without destroying accumulated trade context."""
        try:
            capital = float(starting_capital)
        except Exception:
            logger.warning("Cannot rebase performance baseline with invalid capital: %s", starting_capital)
            return

        if capital <= 0:
            logger.warning("Cannot rebase performance baseline with non-positive capital: %s", capital)
            return

        if reset_trades:
            self.trades.clear()
        if clear_benchmark:
            self.benchmark_curve.clear()

        now_iso = datetime.now().isoformat()
        last_value = None
        if self.equity_curve:
            try:
                last_value = float(self.equity_curve[-1][1])
            except Exception:
                last_value = None
        if last_value is None or abs(last_value - capital) > 1e-9:
            self.equity_curve.append((now_iso, capital))

        self.starting_capital = capital
        await self._save_state_async()
        logger.warning(
            "🩹 Performance baseline rebased (%s). Preserved trades=%d equity_points=%d baseline=$%.2f",
            reason,
            len(self.trades),
            len(self.equity_curve),
            capital,
        )
