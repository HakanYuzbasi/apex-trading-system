"""
core/contracts.py

Data Contracts and Interfaces

Defines the canonical data structures and interfaces used throughout the system.
This provides a single source of truth for data formats and ensures consistency.

Author: Institutional Quant Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Protocol
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(Enum):
    """Position direction."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class SignalType(Enum):
    """Signal type."""
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    HIGH_VOL = "HIGH_VOL"


# ═══════════════════════════════════════════════════════════════════════════
# DATA CONTRACTS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Symbol:
    """Symbol identifier with metadata."""
    ticker: str
    exchange: str = "SMART"
    currency: str = "USD"
    sec_type: str = "STK"
    sector: str = "Unknown"

    def __str__(self) -> str:
        return self.ticker


@dataclass
class OHLCV:
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'Open': self.open,
            'High': self.high,
            'Low': self.low,
            'Close': self.close,
            'Volume': self.volume
        }


@dataclass
class Signal:
    """Trading signal output."""
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    strength: float  # [-1, 1]
    confidence: float  # [0, 1]

    # Component signals
    components: Dict[str, float] = field(default_factory=dict)

    # Metadata
    model_version: str = ""
    data_quality: float = 1.0

    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on."""
        return abs(self.strength) >= 0.40 and self.confidence >= 0.35


@dataclass
class Position:
    """Current position state."""
    symbol: str
    quantity: int
    side: PositionSide
    entry_price: float
    entry_time: datetime
    current_price: float

    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Risk
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return (self.current_price / self.entry_price - 1) * 100
        else:
            return (self.entry_price / self.current_price - 1) * 100


@dataclass
class Order:
    """Order details."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: str  # MARKET, LIMIT, STOP

    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    # Costs
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Trade:
    """Completed trade (round trip)."""
    trade_id: str
    symbol: str
    side: PositionSide
    quantity: int

    # Entry
    entry_price: float
    entry_time: datetime
    entry_order_id: str

    # Exit
    exit_price: float
    exit_time: datetime
    exit_order_id: str
    exit_reason: str

    # P&L
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.net_pnl / (self.entry_price * self.quantity) * 100

    @property
    def holding_days(self) -> int:
        return (self.exit_time - self.entry_time).days


@dataclass
class PortfolioState:
    """Current portfolio state."""
    timestamp: datetime
    cash: float
    equity: float

    positions: Dict[str, Position] = field(default_factory=dict)
    pending_orders: Dict[str, Order] = field(default_factory=dict)

    # Exposure
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0

    # Risk
    daily_pnl: float = 0.0
    drawdown: float = 0.0

    @property
    def num_positions(self) -> int:
        return len([p for p in self.positions.values() if p.quantity != 0])


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.30
    max_positions: int = 15
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    target_volatility: float = 0.12


# ═══════════════════════════════════════════════════════════════════════════
# INTERFACES (PROTOCOLS)
# ═══════════════════════════════════════════════════════════════════════════

class IDataProvider(Protocol):
    """Interface for market data providers."""

    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        ...

    def get_historical_data(
        self,
        symbol: str,
        days: int = 252
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        ...

    def get_quote(self, symbol: str) -> Tuple[float, float]:
        """Get bid/ask quote."""
        ...


class ISignalGenerator(Protocol):
    """Interface for signal generators."""

    def generate_signal(
        self,
        symbol: str,
        prices: pd.Series
    ) -> Signal:
        """Generate trading signal."""
        ...

    def train(
        self,
        historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Train underlying models."""
        ...


class IRiskManager(Protocol):
    """Interface for risk management."""

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        ...

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal: Signal,
        portfolio: PortfolioState
    ) -> int:
        """Calculate position size."""
        ...

    def check_exit_conditions(
        self,
        position: Position,
        signal: Signal
    ) -> Tuple[bool, str]:
        """Check if position should be exited."""
        ...


class IOrderExecutor(Protocol):
    """Interface for order execution."""

    def submit_order(self, order: Order) -> str:
        """Submit order, return order ID."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        ...

    def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        ...

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        ...


class IPerformanceTracker(Protocol):
    """Interface for performance tracking."""

    def record_equity(self, timestamp: datetime, value: float):
        """Record equity point."""
        ...

    def record_trade(self, trade: Trade):
        """Record completed trade."""
        ...

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def validate_signal(signal: Signal) -> Tuple[bool, List[str]]:
    """Validate signal data."""
    errors = []

    if not signal.symbol:
        errors.append("Symbol is required")

    if not -1 <= signal.strength <= 1:
        errors.append(f"Signal strength must be in [-1, 1], got {signal.strength}")

    if not 0 <= signal.confidence <= 1:
        errors.append(f"Confidence must be in [0, 1], got {signal.confidence}")

    return len(errors) == 0, errors


def validate_order(order: Order, portfolio: PortfolioState) -> Tuple[bool, List[str]]:
    """Validate order against portfolio state."""
    errors = []

    if order.quantity <= 0:
        errors.append("Quantity must be positive")

    if order.order_type == "LIMIT" and order.limit_price is None:
        errors.append("Limit price required for LIMIT orders")

    if order.order_type == "STOP" and order.stop_price is None:
        errors.append("Stop price required for STOP orders")

    # Check buying power for buys
    if order.side == OrderSide.BUY:
        estimated_cost = order.quantity * (order.limit_price or 0)
        if estimated_cost > portfolio.cash:
            errors.append(f"Insufficient cash: need {estimated_cost}, have {portfolio.cash}")

    return len(errors) == 0, errors


def validate_position(position: Position) -> Tuple[bool, List[str]]:
    """Validate position data."""
    errors = []

    if position.quantity == 0:
        errors.append("Position quantity is zero")

    if position.entry_price <= 0:
        errors.append("Entry price must be positive")

    if position.current_price <= 0:
        errors.append("Current price must be positive")

    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_signal(
    symbol: str,
    strength: float,
    confidence: float,
    components: Optional[Dict[str, float]] = None
) -> Signal:
    """Create a signal with proper type inference."""
    if strength > 0.40:
        signal_type = SignalType.ENTRY_LONG
    elif strength < -0.40:
        signal_type = SignalType.ENTRY_SHORT
    elif abs(strength) < 0.20:
        signal_type = SignalType.HOLD
    else:
        signal_type = SignalType.HOLD

    return Signal(
        symbol=symbol,
        timestamp=datetime.now(),
        signal_type=signal_type,
        strength=strength,
        confidence=confidence,
        components=components or {}
    )


def create_market_order(
    symbol: str,
    side: OrderSide,
    quantity: int
) -> Order:
    """Create a market order."""
    import uuid
    return Order(
        order_id=str(uuid.uuid4())[:8],
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type="MARKET"
    )


def create_limit_order(
    symbol: str,
    side: OrderSide,
    quantity: int,
    limit_price: float
) -> Order:
    """Create a limit order."""
    import uuid
    return Order(
        order_id=str(uuid.uuid4())[:8],
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type="LIMIT",
        limit_price=limit_price
    )
