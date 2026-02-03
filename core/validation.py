"""
core/validation.py - Input Validation Module

Provides comprehensive validation for all trading inputs including
symbols, prices, quantities, orders, and configuration values.

Features:
- Symbol validation with format and exchange checks
- Price validation with bounds and sanity checks
- Quantity validation for orders
- Order validation combining all checks
- Validation result with detailed error messages
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Set, Any, Dict
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"       # Block the operation
    WARNING = "warning"   # Allow but log warning
    INFO = "info"         # Informational only


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    code: str = ""
    value: Any = None

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_value: Any = None

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_error(self, field: str, message: str, code: str = "", value: Any = None):
        self.issues.append(ValidationIssue(field, message, ValidationSeverity.ERROR, code, value))
        self.is_valid = False

    def add_warning(self, field: str, message: str, code: str = "", value: Any = None):
        self.issues.append(ValidationIssue(field, message, ValidationSeverity.WARNING, code, value))

    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        self.issues.extend(other.issues)
        if not other.is_valid:
            self.is_valid = False

    def __bool__(self) -> bool:
        return self.is_valid


# Known valid exchanges and their symbol patterns
EXCHANGE_PATTERNS = {
    'NYSE': r'^[A-Z]{1,4}$',
    'NASDAQ': r'^[A-Z]{1,5}$',
    'AMEX': r'^[A-Z]{1,4}$',
    'ARCA': r'^[A-Z]{1,5}$',
}

# Known ETFs and special symbols
KNOWN_ETFS = {
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'GLD', 'SLV',
    'USO', 'TLT', 'HYG', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU',
    'XLP', 'XLY', 'XLB', 'XLRE', 'VNQ', 'EEM', 'EFA', 'FXI', 'ARKK'
}

# Invalid/delisted symbols (should be updated regularly)
INVALID_SYMBOLS: Set[str] = set()

# Price boundaries
MIN_PRICE = 0.0001  # Penny stocks can be very low
MAX_PRICE = 1_000_000  # Safety limit for extreme values
MAX_DAILY_CHANGE_PCT = 50.0  # 50% daily change is extreme

# Quantity boundaries
MIN_QUANTITY = 1
MAX_QUANTITY = 1_000_000  # Safety limit


class SymbolValidator:
    """Validates trading symbols."""

    # Standard US equity pattern
    SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}$')

    # Extended pattern for options, futures
    EXTENDED_PATTERN = re.compile(r'^[A-Z]{1,5}[0-9]{0,6}[A-Z]?$')

    @classmethod
    def validate(
        cls,
        symbol: str,
        allow_extended: bool = False,
        check_known: bool = True
    ) -> ValidationResult:
        """
        Validate a trading symbol.

        Args:
            symbol: The symbol to validate
            allow_extended: Allow options/futures symbols
            check_known: Check against known invalid symbols

        Returns:
            ValidationResult with is_valid and any issues
        """
        result = ValidationResult(is_valid=True, validated_value=symbol)

        # Basic type check
        if not isinstance(symbol, str):
            result.add_error("symbol", f"Symbol must be string, got {type(symbol).__name__}")
            return result

        # Normalize
        symbol = symbol.strip().upper()
        result.validated_value = symbol

        # Empty check
        if not symbol:
            result.add_error("symbol", "Symbol cannot be empty")
            return result

        # Length check
        if len(symbol) > 10:
            result.add_error("symbol", f"Symbol too long: {len(symbol)} chars (max 10)")
            return result

        # Pattern check
        pattern = cls.EXTENDED_PATTERN if allow_extended else cls.SYMBOL_PATTERN
        if not pattern.match(symbol):
            result.add_error(
                "symbol",
                f"Invalid symbol format: {symbol}",
                code="INVALID_FORMAT"
            )
            return result

        # Check against known invalid symbols
        if check_known and symbol in INVALID_SYMBOLS:
            result.add_error(
                "symbol",
                f"Symbol is invalid or delisted: {symbol}",
                code="DELISTED"
            )

        # Warnings for unusual patterns
        if len(symbol) == 1:
            result.add_warning("symbol", f"Single-letter symbol: {symbol}")

        if symbol.startswith('X') and len(symbol) == 3:
            result.add_warning("symbol", f"Possible sector ETF: {symbol}")

        return result


class PriceValidator:
    """Validates price values."""

    @classmethod
    def validate(
        cls,
        price: float,
        reference_price: Optional[float] = None,
        min_price: float = MIN_PRICE,
        max_price: float = MAX_PRICE,
        max_change_pct: float = MAX_DAILY_CHANGE_PCT
    ) -> ValidationResult:
        """
        Validate a price value.

        Args:
            price: The price to validate
            reference_price: Reference price for change validation
            min_price: Minimum allowed price
            max_price: Maximum allowed price
            max_change_pct: Maximum allowed change from reference

        Returns:
            ValidationResult with is_valid and any issues
        """
        result = ValidationResult(is_valid=True, validated_value=price)

        # Type check
        if not isinstance(price, (int, float)):
            result.add_error("price", f"Price must be numeric, got {type(price).__name__}")
            return result

        # Convert to float
        price = float(price)
        result.validated_value = price

        # NaN check
        if price != price:  # NaN check
            result.add_error("price", "Price is NaN", code="NAN_VALUE")
            return result

        # Infinity check
        if abs(price) == float('inf'):
            result.add_error("price", "Price is infinite", code="INF_VALUE")
            return result

        # Negative check
        if price < 0:
            result.add_error("price", f"Price cannot be negative: {price}", code="NEGATIVE")
            return result

        # Zero check
        if price == 0:
            result.add_error("price", "Price cannot be zero", code="ZERO_VALUE")
            return result

        # Bounds check
        if price < min_price:
            result.add_error(
                "price",
                f"Price {price} below minimum {min_price}",
                code="BELOW_MIN"
            )

        if price > max_price:
            result.add_error(
                "price",
                f"Price {price} above maximum {max_price}",
                code="ABOVE_MAX"
            )

        # Change validation against reference
        if reference_price is not None and reference_price > 0:
            change_pct = abs((price - reference_price) / reference_price * 100)
            if change_pct > max_change_pct:
                result.add_warning(
                    "price",
                    f"Large price change: {change_pct:.1f}% from reference",
                    code="LARGE_CHANGE",
                    value=change_pct
                )

        # Decimal precision warning
        if price > 1 and len(str(price).split('.')[-1]) > 4:
            result.add_warning(
                "price",
                f"Excessive decimal precision: {price}",
                code="PRECISION"
            )

        return result


class QuantityValidator:
    """Validates order quantities."""

    @classmethod
    def validate(
        cls,
        quantity: int,
        min_quantity: int = MIN_QUANTITY,
        max_quantity: int = MAX_QUANTITY,
        lot_size: int = 1
    ) -> ValidationResult:
        """
        Validate an order quantity.

        Args:
            quantity: The quantity to validate
            min_quantity: Minimum allowed quantity
            max_quantity: Maximum allowed quantity
            lot_size: Required lot size (for round lots)

        Returns:
            ValidationResult with is_valid and any issues
        """
        result = ValidationResult(is_valid=True, validated_value=quantity)

        # Type check
        if not isinstance(quantity, (int, float)):
            result.add_error("quantity", f"Quantity must be numeric, got {type(quantity).__name__}")
            return result

        # Convert to int
        quantity = int(quantity)
        result.validated_value = quantity

        # Positive check
        if quantity <= 0:
            result.add_error("quantity", f"Quantity must be positive: {quantity}", code="NON_POSITIVE")
            return result

        # Bounds check
        if quantity < min_quantity:
            result.add_error(
                "quantity",
                f"Quantity {quantity} below minimum {min_quantity}",
                code="BELOW_MIN"
            )

        if quantity > max_quantity:
            result.add_error(
                "quantity",
                f"Quantity {quantity} above maximum {max_quantity}",
                code="ABOVE_MAX"
            )

        # Lot size check
        if lot_size > 1 and quantity % lot_size != 0:
            result.add_warning(
                "quantity",
                f"Quantity {quantity} not a multiple of lot size {lot_size}",
                code="ODD_LOT"
            )

        # Large order warning
        if quantity > 10000:
            result.add_warning(
                "quantity",
                f"Large order quantity: {quantity}",
                code="LARGE_ORDER"
            )

        return result


class OrderValidator:
    """Validates complete orders."""

    VALID_SIDES = {'BUY', 'SELL', 'LONG', 'SHORT'}
    VALID_ORDER_TYPES = {'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT', 'MOC', 'LOC'}
    VALID_TIME_IN_FORCE = {'DAY', 'GTC', 'IOC', 'FOK', 'GTD'}

    @classmethod
    def validate(
        cls,
        symbol: str,
        side: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "MARKET",
        time_in_force: str = "DAY",
        reference_price: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate a complete order.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Limit price (if applicable)
            order_type: Type of order
            time_in_force: Time in force
            reference_price: Current market price for validation

        Returns:
            ValidationResult with is_valid and any issues
        """
        result = ValidationResult(is_valid=True)

        # Validate symbol
        symbol_result = SymbolValidator.validate(symbol)
        result.merge(symbol_result)

        # Validate side
        side_upper = side.upper() if isinstance(side, str) else ""
        if side_upper not in cls.VALID_SIDES:
            result.add_error(
                "side",
                f"Invalid order side: {side}. Must be one of {cls.VALID_SIDES}",
                code="INVALID_SIDE"
            )

        # Validate order type
        order_type_upper = order_type.upper() if isinstance(order_type, str) else ""
        if order_type_upper not in cls.VALID_ORDER_TYPES:
            result.add_error(
                "order_type",
                f"Invalid order type: {order_type}. Must be one of {cls.VALID_ORDER_TYPES}",
                code="INVALID_ORDER_TYPE"
            )

        # Validate time in force
        tif_upper = time_in_force.upper() if isinstance(time_in_force, str) else ""
        if tif_upper not in cls.VALID_TIME_IN_FORCE:
            result.add_error(
                "time_in_force",
                f"Invalid time in force: {time_in_force}. Must be one of {cls.VALID_TIME_IN_FORCE}",
                code="INVALID_TIF"
            )

        # Validate quantity
        quantity_result = QuantityValidator.validate(quantity)
        result.merge(quantity_result)

        # Validate price for limit orders
        if order_type_upper in {'LIMIT', 'STOP_LIMIT'}:
            if price is None:
                result.add_error(
                    "price",
                    f"Price required for {order_type} orders",
                    code="MISSING_PRICE"
                )
            else:
                price_result = PriceValidator.validate(price, reference_price)
                result.merge(price_result)

        # Business logic validations
        if reference_price is not None and price is not None:
            # Check limit price reasonableness
            if side_upper == 'BUY' and price > reference_price * 1.1:
                result.add_warning(
                    "price",
                    f"Buy limit price {price} is 10%+ above market {reference_price}",
                    code="HIGH_BUY_LIMIT"
                )
            elif side_upper == 'SELL' and price < reference_price * 0.9:
                result.add_warning(
                    "price",
                    f"Sell limit price {price} is 10%+ below market {reference_price}",
                    code="LOW_SELL_LIMIT"
                )

        # Store validated order
        if result.is_valid:
            result.validated_value = {
                'symbol': symbol_result.validated_value,
                'side': side_upper,
                'quantity': quantity_result.validated_value if quantity_result.is_valid else quantity,
                'price': price,
                'order_type': order_type_upper,
                'time_in_force': tif_upper
            }

        return result


class ConfigValidator:
    """Validates configuration values."""

    @classmethod
    def validate_port(cls, port: int, valid_ports: List[int] = None) -> ValidationResult:
        """Validate a network port."""
        result = ValidationResult(is_valid=True, validated_value=port)

        if not isinstance(port, int):
            result.add_error("port", f"Port must be integer, got {type(port).__name__}")
            return result

        if port < 1 or port > 65535:
            result.add_error("port", f"Port {port} out of valid range (1-65535)")

        if valid_ports and port not in valid_ports:
            result.add_error("port", f"Port {port} not in allowed ports: {valid_ports}")

        return result

    @classmethod
    def validate_percentage(cls, value: float, name: str = "value") -> ValidationResult:
        """Validate a percentage value (0-100 or 0-1)."""
        result = ValidationResult(is_valid=True, validated_value=value)

        if not isinstance(value, (int, float)):
            result.add_error(name, f"Must be numeric, got {type(value).__name__}")
            return result

        # Assume 0-1 range if value is <= 1
        if 0 <= value <= 1:
            result.validated_value = value
        elif 0 <= value <= 100:
            result.validated_value = value / 100.0
            result.add_warning(name, f"Converted {value}% to decimal {result.validated_value}")
        else:
            result.add_error(name, f"Invalid percentage: {value}")

        return result

    @classmethod
    def validate_positive_int(cls, value: int, name: str = "value", max_val: int = None) -> ValidationResult:
        """Validate a positive integer."""
        result = ValidationResult(is_valid=True, validated_value=value)

        if not isinstance(value, int):
            result.add_error(name, f"Must be integer, got {type(value).__name__}")
            return result

        if value <= 0:
            result.add_error(name, f"Must be positive: {value}")

        if max_val and value > max_val:
            result.add_error(name, f"{value} exceeds maximum {max_val}")

        return result


# Convenience functions
def validate_symbol(symbol: str) -> ValidationResult:
    """Validate a trading symbol."""
    return SymbolValidator.validate(symbol)


def validate_price(price: float, reference: float = None) -> ValidationResult:
    """Validate a price value."""
    return PriceValidator.validate(price, reference)


def validate_quantity(quantity: int) -> ValidationResult:
    """Validate an order quantity."""
    return QuantityValidator.validate(quantity)


def validate_order(
    symbol: str,
    side: str,
    quantity: int,
    price: float = None,
    order_type: str = "MARKET"
) -> ValidationResult:
    """Validate a complete order."""
    return OrderValidator.validate(symbol, side, quantity, price, order_type)


def is_valid_symbol(symbol: str) -> bool:
    """Quick check if symbol is valid."""
    return SymbolValidator.validate(symbol).is_valid


def is_valid_price(price: float) -> bool:
    """Quick check if price is valid."""
    return PriceValidator.validate(price).is_valid


def is_valid_quantity(quantity: int) -> bool:
    """Quick check if quantity is valid."""
    return QuantityValidator.validate(quantity).is_valid
