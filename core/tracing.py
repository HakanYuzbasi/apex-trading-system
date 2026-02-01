"""
core/tracing.py - Distributed Tracing with OpenTelemetry

Provides distributed tracing for observability across trading operations.
Integrates with OpenTelemetry for compatibility with various backends
(Jaeger, Zipkin, OTLP collectors, etc.)

Features:
- Automatic span creation for trading operations
- Context propagation across async boundaries
- Custom attributes for trading-specific metadata
- Performance metrics extraction
"""

import asyncio
import functools
import logging
import os
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for OpenTelemetry availability
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.info("OpenTelemetry not available. Install with: pip install opentelemetry-sdk")

# Try to import OTLP exporter
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""
    service_name: str = "apex-trading"
    service_version: str = "2.0.0"
    environment: str = "production"
    otlp_endpoint: Optional[str] = None  # e.g., "localhost:4317"
    console_export: bool = False
    sample_rate: float = 1.0  # 1.0 = trace everything


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception):
        pass

    def add_event(self, name: str, attributes: Optional[Dict] = None):
        pass

    def end(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs):
        return NoOpSpan()


# Global tracer instance
_tracer = None
_initialized = False


def setup_tracing(config: Optional[TracingConfig] = None) -> Any:
    """
    Initialize OpenTelemetry tracing.

    Args:
        config: Tracing configuration

    Returns:
        Tracer instance (or NoOpTracer if unavailable)
    """
    global _tracer, _initialized

    if _initialized:
        return _tracer

    cfg = config or TracingConfig(
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        environment=os.getenv("APEX_ENVIRONMENT", "production")
    )

    if not OTEL_AVAILABLE:
        logger.info("OpenTelemetry not available, using no-op tracer")
        _tracer = NoOpTracer()
        _initialized = True
        return _tracer

    try:
        # Create resource with service info
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: cfg.service_name,
            ResourceAttributes.SERVICE_VERSION: cfg.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: cfg.environment,
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter if configured
        if cfg.otlp_endpoint and OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(endpoint=cfg.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP tracing enabled: {cfg.otlp_endpoint}")

        # Add console exporter if configured (useful for debugging)
        if cfg.console_export:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("Console tracing enabled")

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer
        _tracer = trace.get_tracer(cfg.service_name, cfg.service_version)
        _initialized = True

        logger.info(f"Distributed tracing initialized for {cfg.service_name}")
        return _tracer

    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        _tracer = NoOpTracer()
        _initialized = True
        return _tracer


def get_tracer():
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        setup_tracing()
    return _tracer


def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator to trace a function execution.

    Args:
        operation_name: Name for the span
        attributes: Additional attributes to add to span

    Example:
        @trace_operation("generate_signal", {"component": "signal_generator"})
        async def generate_signal(symbol: str):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(operation_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(operation_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for creating a span.

    Example:
        with span("process_order", {"symbol": "AAPL"}):
            order = await place_order(...)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for key, value in attributes.items():
                s.set_attribute(key, value)
        yield s


@asynccontextmanager
async def async_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Async context manager for creating a span.

    Example:
        async with async_span("fetch_market_data", {"symbol": symbol}):
            data = await fetch_data(symbol)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for key, value in attributes.items():
                s.set_attribute(key, value)
        yield s


class TradingTracer:
    """
    Specialized tracer for trading operations.

    Provides pre-configured spans for common trading operations.
    """

    def __init__(self):
        self.tracer = get_tracer()

    def trace_trading_cycle(self, cycle_id: int):
        """Start a span for a complete trading cycle."""
        return self.tracer.start_as_current_span(
            "trading_cycle",
            attributes={
                "cycle_id": cycle_id,
                "start_time": datetime.now().isoformat()
            }
        )

    def trace_signal_generation(self, symbol: str):
        """Start a span for signal generation."""
        return self.tracer.start_as_current_span(
            "signal_generation",
            attributes={"symbol": symbol}
        )

    def trace_order_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str
    ):
        """Start a span for order execution."""
        return self.tracer.start_as_current_span(
            "order_execution",
            attributes={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type
            }
        )

    def trace_market_data_fetch(self, symbols: list):
        """Start a span for market data fetching."""
        return self.tracer.start_as_current_span(
            "market_data_fetch",
            attributes={
                "symbol_count": len(symbols),
                "symbols": ",".join(symbols[:10])  # Limit to first 10
            }
        )

    def trace_risk_check(self, check_type: str):
        """Start a span for risk checking."""
        return self.tracer.start_as_current_span(
            "risk_check",
            attributes={"check_type": check_type}
        )

    def trace_portfolio_update(self):
        """Start a span for portfolio update."""
        return self.tracer.start_as_current_span("portfolio_update")

    def add_order_result(
        self,
        span,
        success: bool,
        fill_price: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Add order result attributes to a span."""
        span.set_attribute("order.success", success)
        if fill_price is not None:
            span.set_attribute("order.fill_price", fill_price)
        if slippage_bps is not None:
            span.set_attribute("order.slippage_bps", slippage_bps)
        if error:
            span.set_attribute("order.error", error)

    def add_signal_result(
        self,
        span,
        signal_strength: float,
        confidence: float,
        direction: str,
        regime: str
    ):
        """Add signal result attributes to a span."""
        span.set_attribute("signal.strength", signal_strength)
        span.set_attribute("signal.confidence", confidence)
        span.set_attribute("signal.direction", direction)
        span.set_attribute("signal.regime", regime)


# Global trading tracer instance
_trading_tracer: Optional[TradingTracer] = None


def get_trading_tracer() -> TradingTracer:
    """Get global trading tracer instance."""
    global _trading_tracer
    if _trading_tracer is None:
        setup_tracing()
        _trading_tracer = TradingTracer()
    return _trading_tracer


# Middleware for FastAPI (if using web endpoints)
def create_tracing_middleware():
    """
    Create FastAPI middleware for tracing HTTP requests.

    Returns:
        Middleware class for FastAPI
    """
    if not OTEL_AVAILABLE:
        return None

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        return FastAPIInstrumentor
    except ImportError:
        logger.info("FastAPI instrumentor not available")
        return None


def shutdown_tracing():
    """Shutdown tracing and flush pending spans."""
    global _initialized
    if OTEL_AVAILABLE and _initialized:
        try:
            provider = trace.get_tracer_provider()
            if hasattr(provider, 'shutdown'):
                provider.shutdown()
            logger.info("Tracing shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down tracing: {e}")
    _initialized = False
