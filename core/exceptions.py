"""
core/exceptions.py - Trading System Exceptions

Comprehensive exception hierarchy for all trading scenarios.
Provides specific error types for better error handling and recovery.

Features:
- Hierarchical exception structure
- Rich error context
- Error codes for programmatic handling
- Serializable for logging and monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


@dataclass
class ApexBaseException(Exception):
    """Base exception for standardized APEX error propagation."""

    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class ApexTradingError(ApexBaseException):
    """Execution and order placement error."""


@dataclass
class ApexRiskError(ApexBaseException):
    """Risk management violation or policy failure."""


@dataclass
class ApexBrokerError(ApexBaseException):
    """Broker connectivity, account, or routing error."""


@dataclass
class ApexDataError(ApexBaseException):
    """Market data and pipeline error."""


@dataclass
class ApexAuthError(ApexBaseException):
    """Authentication and authorization error."""


class ErrorCode(Enum):
    """Standard error codes for trading operations."""
    # Connection errors (1xx)
    CONNECTION_FAILED = "E101"
    CONNECTION_TIMEOUT = "E102"
    CONNECTION_LOST = "E103"
    AUTHENTICATION_FAILED = "E104"

    # Order errors (2xx)
    ORDER_REJECTED = "E201"
    ORDER_TIMEOUT = "E202"
    ORDER_CANCELLED = "E203"
    ORDER_PARTIAL_FILL = "E204"
    INSUFFICIENT_FUNDS = "E205"
    POSITION_LIMIT_EXCEEDED = "E206"
    INVALID_ORDER = "E207"
    DUPLICATE_ORDER = "E208"

    # Data errors (3xx)
    DATA_UNAVAILABLE = "E301"
    DATA_STALE = "E302"
    DATA_INVALID = "E303"
    SYMBOL_NOT_FOUND = "E304"

    # Risk errors (4xx)
    RISK_LIMIT_EXCEEDED = "E401"
    CIRCUIT_BREAKER_OPEN = "E402"
    MAX_DRAWDOWN_EXCEEDED = "E403"
    SECTOR_LIMIT_EXCEEDED = "E404"
    DAILY_LOSS_LIMIT = "E405"

    # System errors (5xx)
    SYSTEM_OVERLOAD = "E501"
    RATE_LIMIT_EXCEEDED = "E502"
    CONFIGURATION_ERROR = "E503"
    INTERNAL_ERROR = "E504"

    # Validation errors (6xx)
    VALIDATION_FAILED = "E601"
    INVALID_SYMBOL = "E602"
    INVALID_PRICE = "E603"
    INVALID_QUANTITY = "E604"


@dataclass
class ErrorContext:
    """Rich context for errors."""
    error_code: ErrorCode
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_code': self.error_code.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'order_id': self.order_id,
            **self.additional_data
        }


class TradingError(Exception):
    """Base exception for all trading errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext(error_code=error_code)
        self.cause = cause

    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code.value,
            'context': self.context.to_dict() if self.context else None
        }


# ============================================================================
# Connection Errors
# ============================================================================

class ConnectionError(TradingError):
    """Base class for connection-related errors."""
    pass


class BrokerConnectionError(ConnectionError):
    """Failed to connect to broker."""

    def __init__(
        self,
        message: str = "Failed to connect to broker",
        host: str = "",
        port: int = 0,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorCode.CONNECTION_FAILED,
            ErrorContext(
                error_code=ErrorCode.CONNECTION_FAILED,
                additional_data={'host': host, 'port': port}
            ),
            cause
        )
        self.host = host
        self.port = port


class ConnectionTimeoutError(ConnectionError):
    """Connection attempt timed out."""

    def __init__(
        self,
        message: str = "Connection timed out",
        timeout_seconds: float = 0,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorCode.CONNECTION_TIMEOUT,
            ErrorContext(
                error_code=ErrorCode.CONNECTION_TIMEOUT,
                additional_data={'timeout_seconds': timeout_seconds}
            ),
            cause
        )
        self.timeout_seconds = timeout_seconds


class ConnectionLostError(ConnectionError):
    """Connection to broker was lost."""

    def __init__(
        self,
        message: str = "Connection lost",
        last_connected: Optional[datetime] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorCode.CONNECTION_LOST,
            ErrorContext(
                error_code=ErrorCode.CONNECTION_LOST,
                additional_data={'last_connected': last_connected.isoformat() if last_connected else None}
            ),
            cause
        )
        self.last_connected = last_connected


class AuthenticationError(ConnectionError):
    """Authentication with broker failed."""

    def __init__(
        self,
        message: str = "Authentication failed",
        reason: str = "",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorCode.AUTHENTICATION_FAILED,
            ErrorContext(
                error_code=ErrorCode.AUTHENTICATION_FAILED,
                additional_data={'reason': reason}
            ),
            cause
        )
        self.reason = reason


# ============================================================================
# Order Errors
# ============================================================================

class OrderError(TradingError):
    """Base class for order-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        order_id: str = "",
        symbol: str = "",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            error_code,
            ErrorContext(
                error_code=error_code,
                symbol=symbol,
                order_id=order_id
            ),
            cause
        )
        self.order_id = order_id
        self.symbol = symbol


class OrderRejectedError(OrderError):
    """Order was rejected by broker."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        reason: str,
        rejection_code: str = "",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Order {order_id} rejected: {reason}",
            ErrorCode.ORDER_REJECTED,
            order_id,
            symbol,
            cause
        )
        self.reason = reason
        self.rejection_code = rejection_code
        self.context.additional_data['reason'] = reason
        self.context.additional_data['rejection_code'] = rejection_code


class OrderTimeoutError(OrderError):
    """Order execution timed out."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        timeout_seconds: float,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Order {order_id} timed out after {timeout_seconds}s",
            ErrorCode.ORDER_TIMEOUT,
            order_id,
            symbol,
            cause
        )
        self.timeout_seconds = timeout_seconds
        self.context.additional_data['timeout_seconds'] = timeout_seconds


class PartialFillError(OrderError):
    """Order was only partially filled."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        requested_quantity: int,
        filled_quantity: int,
        fill_price: float,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Order {order_id} partially filled: {filled_quantity}/{requested_quantity}",
            ErrorCode.ORDER_PARTIAL_FILL,
            order_id,
            symbol,
            cause
        )
        self.requested_quantity = requested_quantity
        self.filled_quantity = filled_quantity
        self.remaining_quantity = requested_quantity - filled_quantity
        self.fill_price = fill_price
        self.context.additional_data.update({
            'requested_quantity': requested_quantity,
            'filled_quantity': filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'fill_price': fill_price
        })


class InsufficientFundsError(OrderError):
    """Insufficient funds to execute order."""

    def __init__(
        self,
        symbol: str,
        required_amount: float,
        available_amount: float,
        order_id: str = "",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Insufficient funds: need ${required_amount:,.2f}, have ${available_amount:,.2f}",
            ErrorCode.INSUFFICIENT_FUNDS,
            order_id,
            symbol,
            cause
        )
        self.required_amount = required_amount
        self.available_amount = available_amount
        self.shortfall = required_amount - available_amount
        self.context.additional_data.update({
            'required_amount': required_amount,
            'available_amount': available_amount,
            'shortfall': self.shortfall
        })


class PositionLimitExceededError(OrderError):
    """Position limit would be exceeded."""

    def __init__(
        self,
        symbol: str,
        current_quantity: int,
        requested_quantity: int,
        max_quantity: int,
        order_id: str = "",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Position limit exceeded for {symbol}: {current_quantity + requested_quantity} > {max_quantity}",
            ErrorCode.POSITION_LIMIT_EXCEEDED,
            order_id,
            symbol,
            cause
        )
        self.current_quantity = current_quantity
        self.requested_quantity = requested_quantity
        self.max_quantity = max_quantity
        self.context.additional_data.update({
            'current_quantity': current_quantity,
            'requested_quantity': requested_quantity,
            'max_quantity': max_quantity
        })


class DuplicateOrderError(OrderError):
    """Duplicate order detected."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        existing_order_id: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Duplicate order: {order_id} conflicts with {existing_order_id}",
            ErrorCode.DUPLICATE_ORDER,
            order_id,
            symbol,
            cause
        )
        self.existing_order_id = existing_order_id


# ============================================================================
# Data Errors
# ============================================================================

class DataError(TradingError):
    """Base class for data-related errors."""
    pass


class DataUnavailableError(DataError):
    """Required data is not available."""

    def __init__(
        self,
        message: str = "Data unavailable",
        symbol: str = "",
        data_type: str = "",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorCode.DATA_UNAVAILABLE,
            ErrorContext(
                error_code=ErrorCode.DATA_UNAVAILABLE,
                symbol=symbol,
                additional_data={'data_type': data_type}
            ),
            cause
        )
        self.symbol = symbol
        self.data_type = data_type


class StaleDataError(DataError):
    """Data is too old to be reliable."""

    def __init__(
        self,
        symbol: str,
        data_age_seconds: float,
        max_age_seconds: float,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Stale data for {symbol}: {data_age_seconds:.1f}s old (max: {max_age_seconds:.1f}s)",
            ErrorCode.DATA_STALE,
            ErrorContext(
                error_code=ErrorCode.DATA_STALE,
                symbol=symbol,
                additional_data={
                    'data_age_seconds': data_age_seconds,
                    'max_age_seconds': max_age_seconds
                }
            ),
            cause
        )
        self.symbol = symbol
        self.data_age_seconds = data_age_seconds
        self.max_age_seconds = max_age_seconds


class SymbolNotFoundError(DataError):
    """Symbol not found or invalid."""

    def __init__(
        self,
        symbol: str,
        reason: str = "Symbol not found",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Symbol not found: {symbol} - {reason}",
            ErrorCode.SYMBOL_NOT_FOUND,
            ErrorContext(
                error_code=ErrorCode.SYMBOL_NOT_FOUND,
                symbol=symbol,
                additional_data={'reason': reason}
            ),
            cause
        )
        self.symbol = symbol
        self.reason = reason


# ============================================================================
# Risk Errors
# ============================================================================

class RiskError(TradingError):
    """Base class for risk-related errors."""
    pass


class RiskLimitExceededError(RiskError):
    """A risk limit has been exceeded."""

    def __init__(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        symbol: str = "",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Risk limit exceeded: {limit_type} = {current_value:.2f} (limit: {limit_value:.2f})",
            ErrorCode.RISK_LIMIT_EXCEEDED,
            ErrorContext(
                error_code=ErrorCode.RISK_LIMIT_EXCEEDED,
                symbol=symbol,
                additional_data={
                    'limit_type': limit_type,
                    'current_value': current_value,
                    'limit_value': limit_value
                }
            ),
            cause
        )
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class CircuitBreakerOpenError(RiskError):
    """Circuit breaker is open, trading halted."""

    def __init__(
        self,
        reason: str,
        triggered_at: Optional[datetime] = None,
        cooldown_seconds: int = 0,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Circuit breaker open: {reason}",
            ErrorCode.CIRCUIT_BREAKER_OPEN,
            ErrorContext(
                error_code=ErrorCode.CIRCUIT_BREAKER_OPEN,
                additional_data={
                    'reason': reason,
                    'triggered_at': triggered_at.isoformat() if triggered_at else None,
                    'cooldown_seconds': cooldown_seconds
                }
            ),
            cause
        )
        self.reason = reason
        self.triggered_at = triggered_at or datetime.now()
        self.cooldown_seconds = cooldown_seconds


class MaxDrawdownExceededError(RiskError):
    """Maximum drawdown limit exceeded."""

    def __init__(
        self,
        current_drawdown: float,
        max_drawdown: float,
        peak_value: float,
        current_value: float,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Max drawdown exceeded: {current_drawdown:.2%} (limit: {max_drawdown:.2%})",
            ErrorCode.MAX_DRAWDOWN_EXCEEDED,
            ErrorContext(
                error_code=ErrorCode.MAX_DRAWDOWN_EXCEEDED,
                additional_data={
                    'current_drawdown': current_drawdown,
                    'max_drawdown': max_drawdown,
                    'peak_value': peak_value,
                    'current_value': current_value
                }
            ),
            cause
        )
        self.current_drawdown = current_drawdown
        self.max_drawdown = max_drawdown
        self.peak_value = peak_value
        self.current_value = current_value


class DailyLossLimitError(RiskError):
    """Daily loss limit exceeded."""

    def __init__(
        self,
        daily_loss: float,
        daily_limit: float,
        starting_capital: float,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Daily loss limit exceeded: ${abs(daily_loss):,.2f} (limit: ${daily_limit:,.2f})",
            ErrorCode.DAILY_LOSS_LIMIT,
            ErrorContext(
                error_code=ErrorCode.DAILY_LOSS_LIMIT,
                additional_data={
                    'daily_loss': daily_loss,
                    'daily_limit': daily_limit,
                    'starting_capital': starting_capital
                }
            ),
            cause
        )
        self.daily_loss = daily_loss
        self.daily_limit = daily_limit


class SectorLimitExceededError(RiskError):
    """Sector exposure limit exceeded."""

    def __init__(
        self,
        sector: str,
        current_exposure: float,
        max_exposure: float,
        symbol: str = "",
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Sector limit exceeded: {sector} = {current_exposure:.1%} (limit: {max_exposure:.1%})",
            ErrorCode.SECTOR_LIMIT_EXCEEDED,
            ErrorContext(
                error_code=ErrorCode.SECTOR_LIMIT_EXCEEDED,
                symbol=symbol,
                additional_data={
                    'sector': sector,
                    'current_exposure': current_exposure,
                    'max_exposure': max_exposure
                }
            ),
            cause
        )
        self.sector = sector
        self.current_exposure = current_exposure
        self.max_exposure = max_exposure


# ============================================================================
# System Errors
# ============================================================================

class SystemError(TradingError):
    """Base class for system-related errors."""
    pass


class RateLimitExceededError(SystemError):
    """Rate limit exceeded."""

    def __init__(
        self,
        operation: str,
        current_rate: float,
        max_rate: float,
        retry_after_seconds: int = 0,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Rate limit exceeded for {operation}: {current_rate:.1f}/s (limit: {max_rate:.1f}/s)",
            ErrorCode.RATE_LIMIT_EXCEEDED,
            ErrorContext(
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                additional_data={
                    'operation': operation,
                    'current_rate': current_rate,
                    'max_rate': max_rate,
                    'retry_after_seconds': retry_after_seconds
                }
            ),
            cause
        )
        self.operation = operation
        self.retry_after_seconds = retry_after_seconds


class ConfigurationError(SystemError):
    """Configuration is invalid."""

    def __init__(
        self,
        message: str,
        config_key: str = "",
        expected_value: Any = None,
        actual_value: Any = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorCode.CONFIGURATION_ERROR,
            ErrorContext(
                error_code=ErrorCode.CONFIGURATION_ERROR,
                additional_data={
                    'config_key': config_key,
                    'expected_value': str(expected_value),
                    'actual_value': str(actual_value)
                }
            ),
            cause
        )
        self.config_key = config_key


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(TradingError):
    """Base class for validation errors."""

    def __init__(
        self,
        message: str,
        field: str = "",
        value: Any = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message,
            ErrorCode.VALIDATION_FAILED,
            ErrorContext(
                error_code=ErrorCode.VALIDATION_FAILED,
                additional_data={'field': field, 'value': str(value)}
            ),
            cause
        )
        self.field = field
        self.value = value


class InvalidSymbolError(ValidationError):
    """Invalid symbol format."""

    def __init__(self, symbol: str, reason: str = "Invalid format"):
        super().__init__(
            f"Invalid symbol '{symbol}': {reason}",
            field="symbol",
            value=symbol
        )
        self.error_code = ErrorCode.INVALID_SYMBOL


class InvalidPriceError(ValidationError):
    """Invalid price value."""

    def __init__(self, price: float, reason: str = "Invalid value"):
        super().__init__(
            f"Invalid price {price}: {reason}",
            field="price",
            value=price
        )
        self.error_code = ErrorCode.INVALID_PRICE


class InvalidQuantityError(ValidationError):
    """Invalid quantity value."""

    def __init__(self, quantity: int, reason: str = "Invalid value"):
        super().__init__(
            f"Invalid quantity {quantity}: {reason}",
            field="quantity",
            value=quantity
        )
        self.error_code = ErrorCode.INVALID_QUANTITY
