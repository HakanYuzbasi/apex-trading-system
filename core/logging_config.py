"""
core/logging_config.py - Structured Logging Configuration

Provides JSON-structured logging for better observability, aggregation, and alerting.
Supports both human-readable console output and machine-parseable file output.
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass, asdict

# Check for optional dependencies
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


@dataclass
class LogContext:
    """Context information to include in all log entries."""
    service: str = "apex-trading"
    environment: str = "production"
    version: str = "2.0.0"
    hostname: str = ""

    def __post_init__(self):
        if not self.hostname:
            import socket
            self.hostname = socket.gethostname()


class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON.

    Output format:
    {
        "timestamp": "2026-02-01T10:30:00.000Z",
        "level": "INFO",
        "logger": "apex.trading",
        "message": "Order filled",
        "context": {...},
        "extra": {...}
    }
    """

    def __init__(self, context: Optional[LogContext] = None):
        super().__init__()
        self.context = context or LogContext()

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": asdict(self.context),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields (excluding standard LogRecord attributes)
        standard_attrs = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'taskName', 'message'
        }

        extra = {}
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                try:
                    json.dumps(value)  # Check if serializable
                    extra[key] = value
                except (TypeError, ValueError):
                    extra[key] = str(value)

        if extra:
            log_entry["extra"] = extra

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }

        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Human-readable colored console formatter.

    Uses ANSI color codes for terminal output.
    """

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Format: timestamp - level - logger - message
        formatted = f"{timestamp} - {color}{record.levelname:8}{self.RESET} - {record.name} - {record.getMessage()}"

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class TradingLogger:
    """
    Specialized logger for trading operations.

    Provides structured logging methods for common trading events.
    """

    def __init__(self, name: str = "apex.trading"):
        self.logger = logging.getLogger(name)

    def order_placed(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        price: Optional[float] = None,
        order_id: Optional[str] = None
    ):
        """Log order placement."""
        self.logger.info(
            "Order placed",
            extra={
                "event": "order_placed",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "order_id": order_id
            }
        )

    def order_filled(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        expected_price: float,
        slippage_bps: float,
        commission: float,
        order_id: Optional[str] = None
    ):
        """Log order fill."""
        self.logger.info(
            "Order filled",
            extra={
                "event": "order_filled",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "fill_price": fill_price,
                "expected_price": expected_price,
                "slippage_bps": slippage_bps,
                "commission": commission,
                "order_id": order_id
            }
        )

    def signal_generated(
        self,
        symbol: str,
        signal_strength: float,
        confidence: float,
        direction: str,
        regime: str
    ):
        """Log signal generation."""
        self.logger.info(
            "Signal generated",
            extra={
                "event": "signal_generated",
                "symbol": symbol,
                "signal_strength": signal_strength,
                "confidence": confidence,
                "direction": direction,
                "regime": regime
            }
        )

    def risk_alert(
        self,
        alert_type: str,
        current_value: float,
        threshold: float,
        action: str
    ):
        """Log risk alert."""
        self.logger.warning(
            f"Risk alert: {alert_type}",
            extra={
                "event": "risk_alert",
                "alert_type": alert_type,
                "current_value": current_value,
                "threshold": threshold,
                "action": action
            }
        )

    def circuit_breaker_triggered(
        self,
        reason: str,
        value: float,
        threshold: float
    ):
        """Log circuit breaker activation."""
        self.logger.error(
            f"Circuit breaker triggered: {reason}",
            extra={
                "event": "circuit_breaker_triggered",
                "reason": reason,
                "value": value,
                "threshold": threshold
            }
        )

    def position_update(
        self,
        symbol: str,
        old_quantity: int,
        new_quantity: int,
        reason: str
    ):
        """Log position change."""
        self.logger.info(
            f"Position updated: {symbol}",
            extra={
                "event": "position_update",
                "symbol": symbol,
                "old_quantity": old_quantity,
                "new_quantity": new_quantity,
                "reason": reason
            }
        )

    def performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = ""
    ):
        """Log performance metric."""
        self.logger.info(
            f"Performance: {metric_name}={value}{unit}",
            extra={
                "event": "performance_metric",
                "metric_name": metric_name,
                "value": value,
                "unit": unit
            }
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
    console_output: bool = True,
    context: Optional[LogContext] = None
) -> logging.Logger:
    """
    Set up structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        json_format: Use JSON format for file output
        console_output: Enable console output
        context: Log context information

    Returns:
        Root logger instance
    """
    # Create log context
    ctx = context or LogContext(
        environment=os.getenv("APEX_ENVIRONMENT", "production")
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (human-readable)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredConsoleFormatter())
        root_logger.addHandler(console_handler)

    # File handler (JSON format)
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )

        if json_format:
            file_handler.setFormatter(JsonFormatter(ctx))
        else:
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )

        root_logger.addHandler(file_handler)

    return root_logger


def setup_structlog():
    """
    Set up structlog for enhanced structured logging.

    Only available if structlog is installed.
    """
    if not STRUCTLOG_AVAILABLE:
        logging.warning("structlog not available, using standard logging")
        return

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_trading_logger(name: str = "apex.trading") -> TradingLogger:
    """Get a specialized trading logger instance."""
    return TradingLogger(name)


# Module-level initialization
_initialized = False


def init_logging():
    """Initialize logging with default settings."""
    global _initialized
    if not _initialized:
        setup_logging(
            level=os.getenv("APEX_LOG_LEVEL", "INFO"),
            log_file=os.getenv("APEX_LOG_FILE", "logs/apex.log"),
            json_format=True,
            console_output=True
        )
        _initialized = True
