"""
core - Core data structures and interfaces

This module contains the canonical data contracts and interfaces
used throughout the APEX trading system.
"""

from .contracts import (
    # Enums
    OrderSide,
    PositionSide,
    OrderStatus,
    SignalType,
    MarketRegime,

    # Data contracts
    Symbol,
    OHLCV,
    Signal,
    Position,
    Order,
    Trade,
    PortfolioState,
    RiskLimits,

    # Interfaces
    IDataProvider,
    ISignalGenerator,
    IRiskManager,
    IOrderExecutor,
    IPerformanceTracker,

    # Validation
    validate_signal,
    validate_order,
    validate_position,

    # Factory functions
    create_signal,
    create_market_order,
    create_limit_order,
)

__all__ = [
    # Enums
    'OrderSide',
    'PositionSide',
    'OrderStatus',
    'SignalType',
    'MarketRegime',

    # Data contracts
    'Symbol',
    'OHLCV',
    'Signal',
    'Position',
    'Order',
    'Trade',
    'PortfolioState',
    'RiskLimits',

    # Interfaces
    'IDataProvider',
    'ISignalGenerator',
    'IRiskManager',
    'IOrderExecutor',
    'IPerformanceTracker',

    # Validation
    'validate_signal',
    'validate_order',
    'validate_position',

    # Factory functions
    'create_signal',
    'create_market_order',
    'create_limit_order',
]
