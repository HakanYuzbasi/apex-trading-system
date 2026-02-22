"""
backtesting/backtest_engine.py - Event-Driven Backtesting Engine (Live-Trade Ready)

Simulates:
- Market data events (bar-by-bar)
- Order execution (latency, slippage, commission)
- Portfolio tracking
- Signal generation replay

Supports:
- Transaction costs
- Dynamic slippage (market impact model)
- Partial fills (volume-based)
- Monte Carlo Simulation
- Advanced Risk Metrics
- Order types: Market, Limit, Stop, Stop-Limit
- Built-in stop management (SL/TP/trailing/max hold)
- Risk guards (position limits, daily loss, drawdown, circuit breaker, cooldown)
- Dynamic position sizing (Kelly, vol-adjusted, regime-aware)
- Open-price fills for t+1 execution realism
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import inspect
import uuid

from core.symbols import AssetClass, parse_symbol, is_market_open
from config import ApexConfig

# Import dynamic slippage model
try:
    from backtesting.market_impact import MarketImpactModel, MarketConditions
    MARKET_IMPACT_AVAILABLE = True
except ImportError:
    MARKET_IMPACT_AVAILABLE = False

try:
    from scipy.stats import norm as _scipy_norm
    _norm_cdf = _scipy_norm.cdf
    _norm_ppf = _scipy_norm.ppf
except ImportError:
    import math

    def _norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_ppf(p):
        if p <= 0:
            return float('-inf')
        if p >= 1:
            return float('inf')
        if p < 0.5:
            return -_norm_ppf(1 - p)
        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

EULER_MASCHERONI = 0.5772156649

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(Enum):
    DAY = "DAY"
    GTC = "GTC"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class _DataView:
    """Read-only time-sliced data view to prevent lookahead."""
    def __init__(self, data: Dict[str, pd.DataFrame], current_time: datetime):
        self._data = data
        self._current_time = current_time

    def __getitem__(self, key: str) -> pd.DataFrame:
        df = self._data[key]
        return df.loc[: self._current_time]

    def get(self, key: str, default=None):
        if key not in self._data:
            return default
        return self.__getitem__(key)

    def items(self):
        for k, v in self._data.items():
            yield k, v.loc[: self._current_time]

    def keys(self):
        return self._data.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._data


@dataclass
class Order:
    """Represents a pending order with type, price levels, and expiry."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    expiry_bar: Optional[int] = None  # Bar index at which order expires
    is_exit: bool = False  # True if this order closes a position (bypasses risk gates)


@dataclass
class Trade:
    """Record of a simulated trade."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    pnl: float = 0.0
    entry_id: Optional[str] = None

    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class Position:
    """Simulated position."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    max_price: float  # For trailing stop logic
    entry_bar: int = 0  # Bar index when position was opened

    def update_price(self, price: float):
        self.current_price = price
        if self.quantity > 0:
            self.max_price = max(self.max_price, price)
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
        else:
            self.max_price = min(self.max_price, price)
            self.unrealized_pnl = (self.avg_entry_price - price) * abs(self.quantity)


@dataclass
class StopLevel:
    """Per-position stop/target/trailing configuration."""
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    trailing_activation_pct: float = 0.025
    trailing_distance_pct: float = 0.02
    max_hold_bars: int = 14
    trailing_active: bool = False
    peak_since_activation: Optional[float] = None


# ---------------------------------------------------------------------------
# StopManager
# ---------------------------------------------------------------------------

class StopManager:
    """Manages per-position stop-loss, take-profit, trailing stop, and max-hold exits.

    Mirrors the behavior of risk/dynamic_exit_manager.py from the live system.
    """

    # Regime multiplier table (matches DynamicExitManager)
    REGIME_MULTIPLIERS = {
        'strong_bull':     {'stop': 1.2, 'target': 1.5, 'hold': 1.5},
        'bull':            {'stop': 1.1, 'target': 1.3, 'hold': 1.2},
        'neutral':         {'stop': 0.9, 'target': 0.8, 'hold': 0.7},
        'bear':            {'stop': 0.8, 'target': 1.2, 'hold': 0.8},
        'strong_bear':     {'stop': 0.7, 'target': 1.4, 'hold': 0.6},
        'high_volatility': {'stop': 0.6, 'target': 0.7, 'hold': 0.5},
    }

    def __init__(self):
        self.stop_levels: Dict[str, StopLevel] = {}

    def register(self, symbol: str, stop_level: StopLevel):
        self.stop_levels[symbol] = stop_level

    def remove(self, symbol: str):
        self.stop_levels.pop(symbol, None)

    def check_exits(
        self,
        positions: Dict[str, Position],
        current_bar: int,
        data: Dict[str, pd.DataFrame],
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Return list of exit orders to queue for positions that hit stop conditions."""
        exits = []
        for symbol, pos in list(positions.items()):
            sl = self.stop_levels.get(symbol)
            if sl is None:
                continue

            # P&L percentage
            if pos.avg_entry_price <= 0:
                continue
            if pos.quantity > 0:
                pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price
            else:
                pnl_pct = (pos.avg_entry_price - pos.current_price) / pos.avg_entry_price

            exit_reason = None

            # 1. Stop loss
            if pnl_pct <= -sl.stop_loss_pct:
                exit_reason = "stop_loss"

            # 2. Take profit
            elif pnl_pct >= sl.take_profit_pct:
                exit_reason = "take_profit"

            # 3. Trailing stop
            elif pnl_pct >= sl.trailing_activation_pct:
                if not sl.trailing_active:
                    sl.trailing_active = True
                    sl.peak_since_activation = pos.current_price
                else:
                    if pos.quantity > 0:
                        sl.peak_since_activation = max(sl.peak_since_activation, pos.current_price)
                        trail_price = sl.peak_since_activation * (1 - sl.trailing_distance_pct)
                        if pos.current_price <= trail_price:
                            exit_reason = "trailing_stop"
                    else:
                        sl.peak_since_activation = min(sl.peak_since_activation, pos.current_price)
                        trail_price = sl.peak_since_activation * (1 + sl.trailing_distance_pct)
                        if pos.current_price >= trail_price:
                            exit_reason = "trailing_stop"
            elif sl.trailing_active:
                # Price dropped below activation after being active — check trail from peak
                if pos.quantity > 0:
                    sl.peak_since_activation = max(sl.peak_since_activation or pos.current_price, pos.current_price)
                    trail_price = sl.peak_since_activation * (1 - sl.trailing_distance_pct)
                    if pos.current_price <= trail_price:
                        exit_reason = "trailing_stop"
                else:
                    sl.peak_since_activation = min(sl.peak_since_activation or pos.current_price, pos.current_price)
                    trail_price = sl.peak_since_activation * (1 + sl.trailing_distance_pct)
                    if pos.current_price >= trail_price:
                        exit_reason = "trailing_stop"

            # 4. Max hold
            if exit_reason is None:
                bars_held = current_bar - pos.entry_bar
                if bars_held >= sl.max_hold_bars:
                    exit_reason = "max_hold"

            if exit_reason:
                side = "SELL" if pos.quantity > 0 else "BUY"
                exits.append({
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs(pos.quantity),
                    "reason": exit_reason,
                })
                logger.info(
                    "event=stop_exit symbol=%s reason=%s pnl_pct=%.4f bars_held=%d",
                    symbol, exit_reason, pnl_pct, current_bar - pos.entry_bar,
                )

        return exits

    def apply_regime(self, symbol: str, regime: str):
        """Scale stop levels by regime multipliers."""
        sl = self.stop_levels.get(symbol)
        if sl is None or regime not in self.REGIME_MULTIPLIERS:
            return
        m = self.REGIME_MULTIPLIERS[regime]
        sl.stop_loss_pct *= m['stop']
        sl.take_profit_pct *= m['target']
        sl.max_hold_bars = int(sl.max_hold_bars * m['hold'])


# ---------------------------------------------------------------------------
# RiskGuard
# ---------------------------------------------------------------------------

class RiskGuard:
    """Enforces position limits, daily loss, drawdown, circuit breaker, and cooldown.

    Mirrors behavior from risk/risk_session.py and config.py.
    """

    def __init__(
        self,
        max_positions: int = 40,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.10,
        enable_circuit_breaker: bool = True,
        circuit_breaker_consecutive_losses: int = 5,
        circuit_breaker_cooldown_bars: int = 1,
        trade_cooldown_bars: int = 1,
        max_order_notional: float = 250_000.0,
        max_order_shares: int = 10_000,
        max_price_deviation_bps: float = 250.0,
    ):
        self.max_positions = max_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.enable_circuit_breaker = enable_circuit_breaker
        self.cb_max_losses = circuit_breaker_consecutive_losses
        self.cb_cooldown_bars = circuit_breaker_cooldown_bars
        self.trade_cooldown_bars = trade_cooldown_bars
        self.max_order_notional = max_order_notional
        self.max_order_shares = max_order_shares
        self.max_price_deviation_bps = max_price_deviation_bps

        # State
        self.day_start_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.consecutive_losses: int = 0
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_trip_bar: int = -9999
        self.last_trade_bar: Dict[str, int] = {}
        self._current_day: Optional[str] = None

    def reset(self, initial_equity: float):
        self.day_start_equity = initial_equity
        self.peak_equity = initial_equity
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_trip_bar = -9999
        self.last_trade_bar = {}
        self._current_day = None

    def on_new_bar(self, equity: float, timestamp: datetime, bar_idx: int):
        """Called at the start of each bar to update daily/peak tracking."""
        day_str = timestamp.strftime('%Y-%m-%d') if hasattr(timestamp, 'strftime') else str(timestamp)[:10]
        if self._current_day != day_str:
            self._current_day = day_str
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        # Auto-reset circuit breaker after cooldown
        if self.circuit_breaker_active and (bar_idx - self.circuit_breaker_trip_bar) >= self.cb_cooldown_bars:
            self.circuit_breaker_active = False
            self.consecutive_losses = 0
            logger.info("event=circuit_breaker_reset bar=%d", bar_idx)

    def record_trade_pnl(self, pnl: float, bar_idx: int):
        """Update consecutive loss counter after a closing trade."""
        if pnl < 0:
            self.consecutive_losses += 1
            if self.enable_circuit_breaker and self.consecutive_losses >= self.cb_max_losses:
                self.circuit_breaker_active = True
                self.circuit_breaker_trip_bar = bar_idx
                logger.warning(
                    "event=circuit_breaker_tripped consecutive_losses=%d bar=%d",
                    self.consecutive_losses, bar_idx,
                )
        else:
            self.consecutive_losses = 0

    @property
    def initialized(self) -> bool:
        """True if reset() or on_new_bar() has been called (via engine.run())."""
        return self._current_day is not None

    def can_enter(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        equity: float,
        n_positions: int,
        bar_idx: int,
        is_exit: bool = False,
        recent_close: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Check whether a new order is allowed.  Returns (allowed, reason)."""
        # Exits always allowed (must be able to close positions)
        if is_exit:
            return True, ""

        # Skip all risk checks if guard was never initialized (backward compat
        # for direct _execute_order_now calls outside of run()).
        if not self.initialized:
            return True, ""

        # Circuit breaker
        if self.circuit_breaker_active:
            return False, "circuit_breaker"

        # Position limit
        if n_positions >= self.max_positions:
            return False, "max_positions"

        # Daily loss
        if self.day_start_equity > 0:
            daily_return = (equity - self.day_start_equity) / self.day_start_equity
            if daily_return <= -self.max_daily_loss_pct:
                return False, "daily_loss_limit"

        # Drawdown
        if self.peak_equity > 0:
            drawdown = (equity - self.peak_equity) / self.peak_equity
            if drawdown <= -self.max_drawdown_pct:
                return False, "drawdown_limit"

        # Per-symbol cooldown
        last_bar = self.last_trade_bar.get(symbol, -9999)
        if (bar_idx - last_bar) < self.trade_cooldown_bars:
            return False, "cooldown"

        # Pre-trade notional check
        notional = abs(quantity) * price
        if notional > self.max_order_notional:
            return False, "max_order_notional"

        # Pre-trade share limit
        if abs(quantity) > self.max_order_shares:
            return False, "max_order_shares"

        # Price deviation check
        if recent_close is not None and recent_close > 0:
            dev_bps = abs(price / recent_close - 1.0) * 10_000
            if dev_bps > self.max_price_deviation_bps:
                return False, "price_deviation"

        return True, ""


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtesting engine with Monte Carlo capability.

    Supports dynamic slippage modeling based on:
    - Order size relative to volume
    - Market volatility
    - Bid-ask spread estimation
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_bps: float = 5.0,
        use_dynamic_slippage: bool = True,
        fx_commission_bps: Optional[float] = None,
        crypto_commission_bps: Optional[float] = None,
        fx_min_commission: float = 0.0,
        crypto_min_commission: float = 0.0,
        fx_spread_bps: Optional[float] = None,
        crypto_spread_bps: Optional[float] = None,
        # Risk controls
        max_positions: int = 40,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.10,
        enable_circuit_breaker: bool = True,
        circuit_breaker_consecutive_losses: int = 5,
        circuit_breaker_cooldown_bars: int = 1,
        trade_cooldown_bars: int = 1,
        # Execution realism
        use_open_price_fill: bool = True,
        max_participation_rate: float = 0.10,
        max_order_notional: float = 250_000.0,
        max_order_shares: int = 10_000,
        # Stop management
        enable_stop_management: bool = True,
        default_stop_loss_pct: float = 0.03,
        default_take_profit_pct: float = 0.06,
        default_trailing_stop_pct: float = 0.02,
        default_trailing_activation_pct: float = 0.025,
        default_max_hold_bars: int = 14,
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.base_slippage_bps = slippage_bps
        self.slippage_pct = slippage_bps / 10000.0
        self.use_dynamic_slippage = use_dynamic_slippage and MARKET_IMPACT_AVAILABLE
        self.fx_commission_bps = fx_commission_bps if fx_commission_bps is not None else ApexConfig.FX_COMMISSION_BPS
        self.crypto_commission_bps = crypto_commission_bps if crypto_commission_bps is not None else ApexConfig.CRYPTO_COMMISSION_BPS
        self.fx_min_commission = fx_min_commission
        self.crypto_min_commission = crypto_min_commission
        self.fx_spread_bps = fx_spread_bps if fx_spread_bps is not None else ApexConfig.FX_SPREAD_BPS
        self.crypto_spread_bps = crypto_spread_bps if crypto_spread_bps is not None else ApexConfig.CRYPTO_SPREAD_BPS

        # Execution realism
        self.use_open_price_fill = use_open_price_fill
        self.max_participation_rate = max_participation_rate

        # Stop management config
        self.enable_stop_management = enable_stop_management
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.default_trailing_stop_pct = default_trailing_stop_pct
        self.default_trailing_activation_pct = default_trailing_activation_pct
        self.default_max_hold_bars = default_max_hold_bars

        self.data: Dict[str, pd.DataFrame] = {}
        self.current_time: datetime = datetime.min

        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.history: List[Dict] = []
        self.pending_orders: List[Order] = []

        self.strategy: Optional[Callable] = None
        self._bar_idx: int = 0

        # Sub-systems
        self.stop_manager = StopManager()
        self.risk_guard = RiskGuard(
            max_positions=max_positions,
            max_daily_loss_pct=max_daily_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
            enable_circuit_breaker=enable_circuit_breaker,
            circuit_breaker_consecutive_losses=circuit_breaker_consecutive_losses,
            circuit_breaker_cooldown_bars=circuit_breaker_cooldown_bars,
            trade_cooldown_bars=trade_cooldown_bars,
            max_order_notional=max_order_notional,
            max_order_shares=max_order_shares,
        )

        # Initialize market impact model for dynamic slippage
        if self.use_dynamic_slippage:
            self.market_impact_model = MarketImpactModel(
                base_spread_bps=slippage_bps,
                impact_multiplier=1.0,
                random_slippage_std=2.0
            )
            logger.info("BacktestEngine initialized with dynamic slippage model")
        else:
            self.market_impact_model = None
            logger.info("BacktestEngine initialized with fixed slippage")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data: Dict[str, pd.DataFrame]):
        """Load historical data."""
        self.data = data
        # Ensure data is sorted
        for symbol in self.data:
            self.data[symbol] = self.data[symbol].sort_index()

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, strategy_func: Callable, start_date: datetime, end_date: datetime):
        """
        Run backtest.

        Args:
            strategy_func: Function(engine, symbol, data) -> None
            start_date: Start datetime
            end_date: End datetime
        """
        self.strategy = strategy_func
        try:
            self._strategy_arity = len(inspect.signature(strategy_func).parameters)
        except Exception:
            self._strategy_arity = 2

        # Collect all timestamps
        timestamps = set()
        for df in self.data.values():
            timestamps.update(df.index)

        sorted_timestamps = sorted([t for t in timestamps if start_date <= t <= end_date])

        logger.info(f"Running backtest from {start_date} to {end_date} (steps: {len(sorted_timestamps)})")

        # Initialize risk guard
        self.risk_guard.reset(self.initial_capital)
        self._bar_idx = 0

        for timestamp in sorted_timestamps:
            self.current_time = timestamp
            self._process_step(timestamp)
            self._bar_idx += 1

        return self.get_results()

    def _process_step(self, timestamp: datetime):
        """Process a single time step."""
        # 0. Update risk guard for new bar
        equity = self.total_equity()
        self.risk_guard.on_new_bar(equity, timestamp, self._bar_idx)

        # 1. Process pending orders (t-1 -> t)
        if self.pending_orders:
            self._process_pending_orders(timestamp)

        # 2. Update prices & equity
        for symbol, pos in list(self.positions.items()):
            if symbol in self.data and timestamp in self.data[symbol].index:
                price = self.data[symbol].loc[timestamp]['Close']
                pos.update_price(price)

        # 3. Check stop management exits (after price updates, before strategy)
        if self.enable_stop_management:
            stop_exits = self.stop_manager.check_exits(
                self.positions, self._bar_idx, self.data, timestamp,
            )
            for ex in stop_exits:
                self._execute_order_now(
                    ex["symbol"], ex["side"], ex["quantity"], is_exit=True,
                )
                self.stop_manager.remove(ex["symbol"])

        # 4. Run strategy (signals at t, executes at t+1)
        if self.strategy:
            data_view = _DataView(self.data, timestamp)
            original_data = self.data
            self.data = data_view
            try:
                if self._strategy_arity >= 3:
                    self.strategy(self, timestamp, data_view)
                else:
                    self.strategy(self, timestamp)
            finally:
                self.data = original_data

        # 5. Record history
        eq = self.total_equity()
        peak = self.risk_guard.peak_equity if self.risk_guard.peak_equity > 0 else eq
        dd = (eq - peak) / peak if peak > 0 else 0.0
        self.history.append({
            'timestamp': timestamp,
            'equity': eq,
            'cash': self.cash,
            'positions': len(self.positions),
            'drawdown': dd,
        })

    def _process_pending_orders(self, timestamp: datetime):
        """Process all pending orders, handling different order types."""
        remaining = []
        for order in self.pending_orders:
            # Check expiry
            if order.expiry_bar is not None and self._bar_idx > order.expiry_bar:
                logger.info("event=order_expired order_id=%s symbol=%s type=%s",
                            order.order_id, order.symbol, order.order_type.value)
                continue

            filled = False
            fill_price = None

            if order.order_type == OrderType.MARKET:
                # Market orders always fill
                filled = True

            elif order.order_type == OrderType.LIMIT:
                bar = self._get_bar(order.symbol, timestamp)
                if bar is not None:
                    if order.side == 'BUY' and bar.get('Low', bar['Close']) <= order.limit_price:
                        fill_price = order.limit_price
                        filled = True
                    elif order.side == 'SELL' and bar.get('High', bar['Close']) >= order.limit_price:
                        fill_price = order.limit_price
                        filled = True

            elif order.order_type == OrderType.STOP:
                bar = self._get_bar(order.symbol, timestamp)
                if bar is not None:
                    if order.side == 'BUY' and bar.get('High', bar['Close']) >= order.stop_price:
                        fill_price = order.stop_price
                        filled = True
                    elif order.side == 'SELL' and bar.get('Low', bar['Close']) <= order.stop_price:
                        fill_price = order.stop_price
                        filled = True

            elif order.order_type == OrderType.STOP_LIMIT:
                bar = self._get_bar(order.symbol, timestamp)
                if bar is not None:
                    triggered = False
                    if order.side == 'BUY' and bar.get('High', bar['Close']) >= order.stop_price:
                        triggered = True
                    elif order.side == 'SELL' and bar.get('Low', bar['Close']) <= order.stop_price:
                        triggered = True

                    if triggered:
                        # Now check limit condition
                        if order.side == 'BUY' and bar.get('Low', bar['Close']) <= order.limit_price:
                            fill_price = order.limit_price
                            filled = True
                        elif order.side == 'SELL' and bar.get('High', bar['Close']) >= order.limit_price:
                            fill_price = order.limit_price
                            filled = True
                        # If stop triggered but limit not met, convert to limit for next bars
                        if not filled:
                            order.order_type = OrderType.LIMIT
                            remaining.append(order)
                            continue

            if filled:
                self._execute_order_now(
                    order.symbol, order.side, order.quantity, fill_price,
                    is_exit=order.is_exit,
                )
            else:
                # Keep non-filled GTC orders; drop expired DAY orders
                if order.time_in_force == TimeInForce.GTC:
                    remaining.append(order)
                elif order.time_in_force == TimeInForce.DAY:
                    # DAY orders expire at end of submitted bar (already past)
                    logger.info("event=day_order_unfilled order_id=%s symbol=%s",
                                order.order_id, order.symbol)

        self.pending_orders = remaining

    def _get_bar(self, symbol: str, timestamp: datetime) -> Optional[Dict]:
        """Get OHLCV bar for symbol at timestamp as a dict."""
        if symbol in self.data and timestamp in self.data[symbol].index:
            row = self.data[symbol].loc[timestamp]
            return row.to_dict() if hasattr(row, 'to_dict') else {'Close': row}
        return None

    # ------------------------------------------------------------------
    # Order placement (public API)
    # ------------------------------------------------------------------

    def execute_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None):
        """Queue market order for next bar execution (t+1). Backward-compatible."""
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            limit_price=price,
            submitted_at=self.current_time,
            time_in_force=TimeInForce.DAY,
            is_exit=self._is_closing_order(symbol, side),
        )
        self.pending_orders.append(order)
        logger.info("event=order_queued symbol=%s side=%s qty=%s type=MARKET submitted_at=%s",
                     symbol, side, quantity, self.current_time)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        time_in_force: TimeInForce = TimeInForce.GTC,
        gtc_bars: int = 5,
    ) -> str:
        """Place a limit order. Returns order_id."""
        order_id = str(uuid.uuid4())[:8]
        expiry = self._bar_idx + gtc_bars if time_in_force == TimeInForce.GTC else self._bar_idx + 1
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            submitted_at=self.current_time,
            time_in_force=time_in_force,
            expiry_bar=expiry,
            is_exit=self._is_closing_order(symbol, side),
        )
        self.pending_orders.append(order)
        logger.info("event=limit_order symbol=%s side=%s qty=%s limit=%.2f id=%s",
                     symbol, side, quantity, limit_price, order_id)
        return order_id

    def place_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        gtc_bars: int = 5,
    ) -> str:
        """Place a stop or stop-limit order. Returns order_id."""
        order_id = str(uuid.uuid4())[:8]
        order_type = OrderType.STOP_LIMIT if limit_price is not None else OrderType.STOP
        expiry = self._bar_idx + gtc_bars if time_in_force == TimeInForce.GTC else self._bar_idx + 1
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            submitted_at=self.current_time,
            time_in_force=time_in_force,
            expiry_bar=expiry,
            is_exit=self._is_closing_order(symbol, side),
        )
        self.pending_orders.append(order)
        logger.info("event=stop_order symbol=%s side=%s qty=%s stop=%.2f type=%s id=%s",
                     symbol, side, quantity, stop_price, order_type.value, order_id)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID. Returns True if found and cancelled."""
        for i, order in enumerate(self.pending_orders):
            if order.order_id == order_id:
                self.pending_orders.pop(i)
                logger.info("event=order_cancelled order_id=%s symbol=%s", order_id, order.symbol)
                return True
        return False

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all pending orders, optionally filtered by symbol."""
        if symbol is None:
            count = len(self.pending_orders)
            self.pending_orders = []
        else:
            before = len(self.pending_orders)
            self.pending_orders = [o for o in self.pending_orders if o.symbol != symbol]
            count = before - len(self.pending_orders)
        logger.info("event=orders_cancelled count=%d symbol=%s", count, symbol or "ALL")

    def _is_closing_order(self, symbol: str, side: str) -> bool:
        """Determine if this order would close/reduce an existing position."""
        if symbol not in self.positions:
            return False
        pos = self.positions[symbol]
        return (pos.quantity > 0 and side == 'SELL') or (pos.quantity < 0 and side == 'BUY')

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def total_equity(self) -> float:
        """Calculate total equity."""
        pos_value = sum(p.quantity * p.current_price for p in self.positions.values())
        return self.cash + pos_value

    def _execute_order_now(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        is_exit: bool = False,
    ):
        """Execute simulation order immediately with realistic slippage."""
        # Auto-detect exit orders for backward compatibility
        if not is_exit:
            is_exit = self._is_closing_order(symbol, side)

        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            logger.warning("event=order_rejected symbol=%s reason=invalid_symbol", symbol)
            return
        logger.info(
            "event=symbol_normalization input=%s normalized=%s broker=%s",
            symbol,
            parsed.normalized,
            parsed.normalized,
        )

        if quantity <= 0:
            logger.warning("event=order_rejected symbol=%s reason=non_positive_quantity quantity=%s", symbol, quantity)
            return

        # Market hours gating (24/7 crypto, 24/5 FX, equity hours)
        if not is_market_open(parsed, self.current_time, assume_daily=True):
            logger.warning("event=order_rejected symbol=%s reason=market_closed", parsed.normalized)
            return

        if parsed.asset_class == AssetClass.EQUITY and isinstance(quantity, float) and not quantity.is_integer():
            logger.warning("event=order_rejected symbol=%s reason=fractional_equity quantity=%s", parsed.normalized, quantity)
            return

        # Get price if not provided
        if price is None:
            if symbol in self.data and self.current_time in self.data[symbol].index:
                row = self.data[symbol].loc[self.current_time]
                if self.use_open_price_fill and 'Open' in self.data[symbol].columns:
                    raw_price = row['Open']
                else:
                    raw_price = row['Close']
            else:
                logger.warning("event=order_rejected symbol=%s reason=no_price", parsed.normalized)
                return
        else:
            raw_price = price

        # Risk guard check
        recent_close = None
        if symbol in self.data and self.current_time in self.data[symbol].index:
            recent_close = self.data[symbol].loc[self.current_time]['Close']

        allowed, reason = self.risk_guard.can_enter(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=raw_price,
            equity=self.total_equity(),
            n_positions=len(self.positions),
            bar_idx=self._bar_idx,
            is_exit=is_exit,
            recent_close=recent_close,
        )
        if not allowed:
            logger.warning("event=order_rejected symbol=%s reason=risk_%s", parsed.normalized, reason)
            return

        # Partial fill simulation (volume-based)
        if self.max_participation_rate > 0 and not is_exit:
            fill_qty = self._apply_partial_fill(symbol, quantity)
            if fill_qty <= 0:
                logger.warning("event=order_rejected symbol=%s reason=no_volume", parsed.normalized)
                return
            if fill_qty < quantity:
                # Re-queue remainder
                remainder = quantity - fill_qty
                remainder_order = Order(
                    order_id=str(uuid.uuid4())[:8],
                    symbol=symbol,
                    side=side,
                    quantity=remainder,
                    order_type=OrderType.MARKET,
                    submitted_at=self.current_time,
                    time_in_force=TimeInForce.GTC,
                    expiry_bar=self._bar_idx + 5,
                    is_exit=is_exit,
                )
                self.pending_orders.append(remainder_order)
                logger.info("event=partial_fill symbol=%s filled=%s remainder=%s",
                            symbol, fill_qty, remainder)
            quantity = fill_qty

        # Calculate slippage
        slippage_pct = self._get_slippage_pct(parsed.asset_class)

        if self.use_dynamic_slippage and self.market_impact_model:
            fill_price = self._calculate_dynamic_fill_price(
                symbol, raw_price, quantity, side
            )
        else:
            if side == 'BUY':
                fill_price = raw_price * (1 + slippage_pct)
            else:
                fill_price = raw_price * (1 - slippage_pct)

        # Calculate commission
        notional = abs(quantity) * fill_price
        if parsed.asset_class == AssetClass.EQUITY:
            commission = max(self.min_commission, abs(quantity) * self.commission_per_share)
        elif parsed.asset_class == AssetClass.FOREX:
            commission = max(self.fx_min_commission, notional * (self.fx_commission_bps / 10000.0))
        else:
            commission = max(self.crypto_min_commission, notional * (self.crypto_commission_bps / 10000.0))

        logger.info(
            "event=fee_model asset=%s symbol=%s notional=%.2f commission=%.4f slippage_bps=%.2f",
            parsed.asset_class.value,
            parsed.normalized,
            notional,
            commission,
            slippage_pct * 10000,
        )

        cost = quantity * fill_price

        # Capture P&L for closing trades
        pnl = 0.0
        is_new_position = symbol not in self.positions

        # Update accounting
        if side == 'BUY':
            self.cash -= (cost + commission)

            if symbol in self.positions:
                pos = self.positions[symbol]

                # Increasing long
                if pos.quantity > 0:
                    total_cost = pos.quantity * pos.avg_entry_price + cost
                    pos.avg_entry_price = total_cost / (pos.quantity + quantity)

                # Closing short
                elif pos.quantity < 0:
                    covered = min(quantity, abs(pos.quantity))
                    trade_pnl = (pos.avg_entry_price - fill_price) * covered - (commission * (covered/quantity))
                    pos.realized_pnl += trade_pnl
                    pnl = trade_pnl

                    if quantity > abs(pos.quantity):
                        pos.avg_entry_price = fill_price
                        pos.max_price = fill_price
                        pos.entry_bar = self._bar_idx

                pos.quantity += quantity
                if pos.quantity == 0:
                    self.stop_manager.remove(symbol)
                    del self.positions[symbol]
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    max_price=fill_price,
                    entry_bar=self._bar_idx,
                )

        elif side == 'SELL':
            self.cash += (cost - commission)

            if symbol in self.positions:
                pos = self.positions[symbol]

                # Closing long
                if pos.quantity > 0:
                    sold = min(quantity, pos.quantity)
                    trade_pnl = (fill_price - pos.avg_entry_price) * sold - (commission * (sold/quantity))
                    pos.realized_pnl += trade_pnl
                    pnl = trade_pnl

                    if quantity > pos.quantity:
                        pos.avg_entry_price = fill_price
                        pos.max_price = fill_price
                        pos.entry_bar = self._bar_idx

                # Increasing short
                elif pos.quantity < 0:
                    total_val = abs(pos.quantity) * pos.avg_entry_price + cost
                    pos.avg_entry_price = total_val / (abs(pos.quantity) + quantity)

                pos.quantity -= quantity
                if pos.quantity == 0:
                    self.stop_manager.remove(symbol)
                    del self.positions[symbol]
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=-quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    unrealized_pnl=0,
                    realized_pnl=0,
                    max_price=fill_price,
                    entry_bar=self._bar_idx,
                )

        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            timestamp=self.current_time,
            pnl=pnl
        )
        self.trades.append(trade)

        # Record trade in risk guard
        self.risk_guard.last_trade_bar[symbol] = self._bar_idx
        if pnl != 0:
            self.risk_guard.record_trade_pnl(pnl, self._bar_idx)

        # Register default stops for new positions
        if is_new_position and self.enable_stop_management and symbol in self.positions:
            self.stop_manager.register(symbol, StopLevel(
                stop_loss_pct=self.default_stop_loss_pct,
                take_profit_pct=self.default_take_profit_pct,
                trailing_activation_pct=self.default_trailing_activation_pct,
                trailing_distance_pct=self.default_trailing_stop_pct,
                max_hold_bars=self.default_max_hold_bars,
            ))

    def _apply_partial_fill(self, symbol: str, quantity: float) -> float:
        """Reduce quantity based on volume participation limit. Returns fillable qty."""
        if symbol not in self.data or self.current_time not in self.data[symbol].index:
            return quantity
        row = self.data[symbol].loc[self.current_time]
        volume = row.get('Volume', row.get('volume', None))
        if volume is None or volume <= 0:
            return quantity
        max_fill = volume * self.max_participation_rate
        if max_fill <= 0:
            return 0.0
        return min(quantity, max_fill)

    # ------------------------------------------------------------------
    # Dynamic position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        entry_price: float,
        signal_strength: float,
        confidence: float = 0.5,
        prices: Optional[pd.Series] = None,
        regime: str = 'neutral',
        max_portfolio_risk: float = 0.02,
        max_position_pct: float = 0.05,
    ) -> Dict[str, Any]:
        """Calculate position size matching GodLevelRiskManager logic.

        Returns dict with shares, stop_loss, take_profit, trailing_stop_pct, etc.
        """
        equity = self.total_equity()

        # ATR calculation (14 period)
        atr = 0.0
        atr_pct = 0.02  # default 2%
        if prices is not None and len(prices) >= 15:
            tr_values = []
            for i in range(1, len(prices)):
                high_low = abs(prices.iloc[i] - prices.iloc[i - 1])  # simplified TR
                tr_values.append(high_low)
            if tr_values:
                atr = float(np.mean(tr_values[-14:]))
                atr_pct = atr / entry_price if entry_price > 0 else 0.02
                atr_pct = max(atr_pct, 0.001)

        # ATR multiplier by regime
        atr_mult_table = {
            'strong_bull': 2.5, 'bull': 2.0, 'neutral': 1.5,
            'bear': 2.0, 'strong_bear': 2.5, 'high_volatility': 3.0,
        }
        base_mult = atr_mult_table.get(regime, 2.0)
        signal_adj = 0.5 + abs(signal_strength) * 0.5
        atr_multiplier = base_mult * signal_adj

        stop_distance_pct = min(atr_pct * atr_multiplier, 0.05)

        # Stop/target prices
        if signal_strength > 0:
            stop_loss = entry_price * (1 - stop_distance_pct)
            take_profit = entry_price * (1 + stop_distance_pct * 2.5)
        else:
            stop_loss = entry_price * (1 + stop_distance_pct)
            take_profit = entry_price * (1 - stop_distance_pct * 2.5)

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.02

        # Kelly fraction (half-Kelly)
        win_prob = min(max(0.5 + (confidence * abs(signal_strength)) * 0.3, 0.4), 0.75)
        win_loss_ratio = 1.0 + abs(signal_strength) * 0.5
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly = max(min(kelly * 0.5, 0.25), 0.02)  # half-Kelly, capped

        # Volatility adjustment
        target_vol = 0.02
        vol_ratio = target_vol / atr_pct
        vol_adj = max(min(vol_ratio, 2.0), 0.25)

        # Drawdown multiplier
        peak = self.risk_guard.peak_equity if self.risk_guard.peak_equity > 0 else equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd <= 0.05:
            dd_mult = 1.0
        elif dd <= 0.10:
            dd_mult = 0.50
        elif dd <= 0.15:
            dd_mult = 0.25
        else:
            dd_mult = 0.10

        # Regime multiplier
        regime_mult_table = {
            'strong_bull': 1.2, 'bull': 1.1, 'neutral': 0.9,
            'bear': 0.8, 'strong_bear': 0.7, 'high_volatility': 0.6,
        }
        regime_mult = regime_mult_table.get(regime, 1.0)

        # Combined sizing
        risk_budget = equity * max_portfolio_risk
        adjusted_budget = risk_budget * kelly * vol_adj * dd_mult * regime_mult

        shares = int(adjusted_budget / risk_per_share) if risk_per_share > 0 else 0

        # Cap by max position value
        max_pos_value = equity * max_position_pct
        max_shares = int(max_pos_value / entry_price) if entry_price > 0 else 0
        shares = min(shares, max_shares)
        shares = max(shares, 0)

        # Trailing stop
        trailing_stop_pct = stop_distance_pct * (0.5 + 0.5 * (1 - confidence))

        position_value = shares * entry_price
        total_risk = shares * risk_per_share

        return {
            'shares': shares,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop_pct': trailing_stop_pct,
            'risk_per_share': risk_per_share,
            'total_risk': total_risk,
            'risk_reward_ratio': (abs(take_profit - entry_price) / risk_per_share) if risk_per_share > 0 else 0,
            'atr': atr,
            'atr_pct': atr_pct,
            'kelly_fraction': kelly,
            'position_pct': position_value / equity if equity > 0 else 0,
            'stop_loss_pct': stop_distance_pct,
            'drawdown_multiplier': dd_mult,
            'regime_multiplier': regime_mult,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _annualization_factor(self) -> int:
        classes = set()
        for symbol in self.data.keys():
            try:
                classes.add(parse_symbol(symbol).asset_class)
            except ValueError:
                continue
        if classes == {AssetClass.CRYPTO}:
            return 365
        if classes == {AssetClass.FOREX}:
            return 260
        return 252

    def _get_slippage_pct(self, asset_class: AssetClass) -> float:
        if asset_class == AssetClass.FOREX:
            return self.fx_spread_bps / 10000.0
        if asset_class == AssetClass.CRYPTO:
            return self.crypto_spread_bps / 10000.0
        return self.slippage_pct

    def _calculate_dynamic_fill_price(
        self,
        symbol: str,
        raw_price: float,
        quantity: float,
        side: str
    ) -> float:
        """
        Calculate fill price using dynamic market impact model.

        Considers:
        - Order size relative to average volume
        - Recent volatility
        - Estimated bid-ask spread
        """
        if not self.market_impact_model:
            mult = 1 + self.slippage_pct if side == 'BUY' else 1 - self.slippage_pct
            return raw_price * mult

        if symbol not in self.data:
            mult = 1 + self.slippage_pct if side == 'BUY' else 1 - self.slippage_pct
            return raw_price * mult

        df = self.data[symbol]

        hist = df[df.index <= self.current_time].tail(20)

        if len(hist) < 5:
            mult = 1 + self.slippage_pct if side == 'BUY' else 1 - self.slippage_pct
            return raw_price * mult

        close = hist['Close'] if 'Close' in hist.columns else hist.get('close', pd.Series([raw_price]))
        volume = hist['Volume'] if 'Volume' in hist.columns else hist.get('volume', pd.Series([1000000]))
        high = hist['High'] if 'High' in hist.columns else hist.get('high', close)
        low = hist['Low'] if 'Low' in hist.columns else hist.get('low', close)

        avg_volume = volume.mean() if len(volume) > 0 else 1000000

        returns = close.pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0.02

        if len(close) > 0 and close.iloc[-1] > 0:
            avg_range = ((high - low) / close).mean()
            spread_bps = max(avg_range * 10000 / 4, self.base_slippage_bps)
        else:
            spread_bps = self.base_slippage_bps

        conditions = MarketConditions(
            avg_daily_volume=avg_volume,
            avg_daily_turnover=avg_volume * raw_price,
            volatility=volatility,
            bid_ask_spread_bps=spread_bps,
            current_volume_ratio=1.0,
            time_of_day=self.current_time.time() if hasattr(self.current_time, 'time') else None
        )

        costs = self.market_impact_model.calculate_execution_costs(
            order_size_shares=quantity,
            price=raw_price,
            side=side,
            conditions=conditions
        )

        return costs.effective_price

    # ------------------------------------------------------------------
    # Results & metrics
    # ------------------------------------------------------------------

    def get_results(self) -> Dict:
        """Calculate advanced backtest metrics."""
        if not self.history:
            return {}

        df = pd.DataFrame(self.history).set_index('timestamp')
        df['returns'] = df['equity'].pct_change().fillna(0)

        # Basic Metrics
        total_return = (df['equity'].iloc[-1] / self.initial_capital) - 1
        ann_factor = self._annualization_factor()
        volatility = df['returns'].std() * np.sqrt(ann_factor)
        sharpe = (df['returns'].mean() / df['returns'].std()) * np.sqrt(ann_factor) if df['returns'].std() > 0 else 0

        # Drawdown Stats
        peaks = df['equity'].cummax()
        drawdown = (df['equity'] / peaks) - 1
        max_drawdown = drawdown.min()
        max_dd_duration = (drawdown < 0).astype(int).groupby(drawdown.eq(0).cumsum()).cumsum().max()

        # Advanced Metrics
        sortino = 0
        downside_returns = df['returns'][df['returns'] < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (df['returns'].mean() / downside_returns.std()) * np.sqrt(ann_factor)

        cagr = ((1 + total_return) ** (ann_factor / len(df))) - 1 if len(df) > 1 else total_return
        calmar = 0
        if max_drawdown < 0:
            calmar = cagr / abs(max_drawdown)

        # Trade Analysis
        win_rate = 0
        profit_factor = 0
        avg_trade = 0

        closed_trades = [t for t in self.trades if t.pnl != 0]
        if closed_trades:
            winners = [t for t in closed_trades if t.pnl > 0]
            losers = [t for t in closed_trades if t.pnl <= 0]

            win_rate = len(winners) / len(closed_trades)
            gross_profit = sum(t.pnl for t in winners)
            gross_loss = abs(sum(t.pnl for t in losers))

            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_trade = sum(t.pnl for t in closed_trades) / len(closed_trades)

        # Probabilistic Sharpe Ratio (PSR)
        psr = 0.0
        n_obs = len(df)
        returns_arr = df['returns'].values
        if n_obs >= 10 and sharpe > 0:
            skew = float(pd.Series(returns_arr).skew())
            kurt = float(pd.Series(returns_arr).kurtosis()) + 3
            sr_per = sharpe / np.sqrt(ann_factor)
            se_denom = 1 - skew * sr_per + ((kurt - 1) / 4) * sr_per ** 2
            se_sr = np.sqrt(max(se_denom, 1e-10) / (n_obs - 1))
            psr = float(_norm_cdf(sr_per / se_sr)) if se_sr > 0 else 0.0

        # Risk guard stats
        risk_stats = {
            'circuit_breaker_trips': 1 if self.risk_guard.circuit_breaker_active else 0,
            'peak_equity': self.risk_guard.peak_equity,
            'final_drawdown': (df['equity'].iloc[-1] - self.risk_guard.peak_equity) / self.risk_guard.peak_equity if self.risk_guard.peak_equity > 0 else 0,
        }

        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'probabilistic_sharpe': psr,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'max_dd_duration': int(max_dd_duration),
            'volatility': volatility,
            'final_equity': df['equity'].iloc[-1],
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'risk_stats': risk_stats,
            'history': df
        }

        return metrics

    def run_monte_carlo(self, n_sims: int = 1000) -> Dict[str, float]:
        """
        Run Monte Carlo simulation on the equity curve returns.

        Args:
            n_sims: Number of simulations

        Returns:
            Dict of risk metrics (VaR, CVaR at various confidence levels)
        """
        if not self.history:
            return {}

        df = pd.DataFrame(self.history)
        returns = df['equity'].pct_change().dropna().values

        if len(returns) < 10:
            return {}

        final_values = []
        days = len(returns)
        start_equity = df['equity'].iloc[-1]

        # Bootstrap resampling
        for _ in range(n_sims):
            sim_returns = np.random.choice(returns, size=days, replace=True)
            cum_returns = np.cumprod(1 + sim_returns)
            final_values.append(start_equity * cum_returns[-1])

        final_values = np.array(final_values)

        sorted_outcomes = np.sort(final_values)

        return {
            'mc_min_equity': sorted_outcomes[0],
            'mc_median_equity': np.median(sorted_outcomes),
            'mc_95_pct_equity': sorted_outcomes[int(n_sims * 0.05)],
            'mc_99_pct_equity': sorted_outcomes[int(n_sims * 0.01)],
            'mc_max_equity': sorted_outcomes[-1]
        }
