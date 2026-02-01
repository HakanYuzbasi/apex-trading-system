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
    TimeoutCategory,
    TimeoutConfig,
    TimeoutError,
    with_timeout,
    with_timeout_retry,
    timeout_decorator,
    retry_with_timeout,
    TimeoutContext,
    broker_timeout,
    market_data_timeout,
    cache_timeout,
)

# Bulkhead pattern
from .bulkhead import (
    BulkheadState,
    BulkheadConfig,
    BulkheadMetrics,
    Bulkhead,
    BulkheadOpenError,
    BulkheadRejectedError,
    BulkheadRegistry,
    get_bulkhead_registry,
    bulkhead_protected,
    create_trading_bulkheads,
)

# Feature flags
from .feature_flags import (
    FeatureFlags,
    FeatureFlagManager,
    get_flag_manager,
    is_enabled,
    set_flag,
    flag_override,
    feature_gated,
)

# Structured logging
from .logging_config import (
    LogConfig,
    setup_logging,
    get_logger,
    LogContext,
    set_correlation_id,
    set_trading_cycle,
    set_symbol,
    PerformanceLogger,
)

# Health checking
from .health_checker import (
    HealthStatus,
    HealthCheckResult,
    HealthCheck,
    HealthChecker,
    get_health_checker,
    create_health_endpoints,
    MemoryUsageCheck,
    DiskSpaceCheck,
    HeartbeatCheck,
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
    'TimeoutCategory',
    'TimeoutConfig',
    'TimeoutError',
    'with_timeout',
    'with_timeout_retry',
    'timeout_decorator',
    'retry_with_timeout',
    'TimeoutContext',
    'broker_timeout',
    'market_data_timeout',
    'cache_timeout',

    # Bulkhead pattern
    'BulkheadState',
    'BulkheadConfig',
    'BulkheadMetrics',
    'Bulkhead',
    'BulkheadOpenError',
    'BulkheadRejectedError',
    'BulkheadRegistry',
    'get_bulkhead_registry',
    'bulkhead_protected',
    'create_trading_bulkheads',

    # Feature flags
    'FeatureFlags',
    'FeatureFlagManager',
    'get_flag_manager',
    'is_enabled',
    'set_flag',
    'flag_override',
    'feature_gated',

    # Structured logging
    'LogConfig',
    'setup_logging',
    'get_logger',
    'LogContext',
    'set_correlation_id',
    'set_trading_cycle',
    'set_symbol',
    'PerformanceLogger',

    # Health checking
    'HealthStatus',
    'HealthCheckResult',
    'HealthCheck',
    'HealthChecker',
    'get_health_checker',
    'create_health_endpoints',
    'MemoryUsageCheck',
    'DiskSpaceCheck',
    'HeartbeatCheck',

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
