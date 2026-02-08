"""
core - Core data structures, interfaces, and utilities

This module contains the canonical data contracts, interfaces, and
production utilities used throughout the APEX trading system.

Modules:
- contracts: Data contracts and interfaces
- timeout: Timeout utilities for async operations
- bulkhead: Failure isolation pattern
- feature_flags: Runtime feature toggles
- logging_config: Structured logging
- health_checker: System health monitoring
- tracing: Distributed tracing with OpenTelemetry
- hot_reload: Dynamic configuration updates
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

# Timeout utilities
from .timeout import (
    TimeoutConfig,
    TimeoutError,
    with_timeout,
    run_with_timeout,
    run_with_timeout_and_retry,
    TimeoutContext,
    configure_timeouts,
)

# Bulkhead pattern
from .bulkhead import (
    BulkheadState,
    BulkheadConfig,
    BulkheadMetrics,
    Bulkhead,
    BulkheadOpenError,
    BulkheadTimeoutError,
    BulkheadFullError,
    BulkheadRegistry,
    get_bulkhead_registry,
    create_trading_bulkheads,
)

# Feature flags
from .feature_flags import (
    FeatureFlags,
    FeatureFlagManager,
    get_flag_manager,
    get_feature_flags,
    is_enabled,
    feature_flag,
)

# Structured logging
from .logging_config import (
    setup_logging,
    LogContext,
    TradingLogger,
    get_trading_logger,
)

# Health checking
from .health_checker import (
    HealthStatus,
    HealthCheck,
    HealthChecker,
    get_health_checker,
    create_health_endpoints,
)

# Distributed tracing
from .tracing import (
    TracingConfig,
    setup_tracing,
    get_tracer,
    trace_operation,
    span,
    async_span,
    TradingTracer,
    get_trading_tracer,
    shutdown_tracing,
)

# Hot reload configuration
from .hot_reload import (
    HotReloadConfig,
    HotReloadManager,
    get_hot_reload_manager,
    init_hot_reload,
    get_config,
    on_config_change,
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

    # Timeout utilities
    'TimeoutConfig',
    'TimeoutError',
    'with_timeout',
    'run_with_timeout',
    'run_with_timeout_and_retry',
    'TimeoutContext',
    'configure_timeouts',

    # Bulkhead pattern
    'BulkheadState',
    'BulkheadConfig',
    'BulkheadMetrics',
    'Bulkhead',
    'BulkheadOpenError',
    'BulkheadTimeoutError',
    'BulkheadFullError',
    'BulkheadRegistry',
    'get_bulkhead_registry',
    'create_trading_bulkheads',

    # Feature flags
    'FeatureFlags',
    'FeatureFlagManager',
    'get_flag_manager',
    'get_feature_flags',
    'is_enabled',
    'feature_flag',

    # Structured logging
    'setup_logging',
    'LogContext',
    'TradingLogger',
    'get_trading_logger',

    # Health checking
    'HealthStatus',
    'HealthCheck',
    'HealthChecker',
    'get_health_checker',
    'create_health_endpoints',

    # Distributed tracing
    'TracingConfig',
    'setup_tracing',
    'get_tracer',
    'trace_operation',
    'span',
    'async_span',
    'TradingTracer',
    'get_trading_tracer',
    'shutdown_tracing',

    # Hot reload configuration
    'HotReloadConfig',
    'HotReloadManager',
    'get_hot_reload_manager',
    'init_hot_reload',
    'get_config',
    'on_config_change',
]
