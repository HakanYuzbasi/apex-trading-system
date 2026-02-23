"""Structured logging utility for production-grade logging with JSON formatting."""

import logging
import json
import sys
from datetime import datetime
from typing import Optional
from contextvars import ContextVar
import traceback

# Context variables for correlation tracking
request_id_ctx: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add correlation IDs if available
        request_id = request_id_ctx.get()
        if request_id:
            log_data['request_id'] = request_id

        user_id = user_id_ctx.get()
        if user_id:
            log_data['user_id'] = user_id

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data)


class StructuredLogger:
    """Wrapper for structured logging with context management."""

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers = []
        
        # Add structured formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False

    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with extra fields."""
        extra_fields = {k: v for k, v in kwargs.items() if k != 'exc_info'}
        exc_info = kwargs.get('exc_info', False)
        
        # Create a log record with extra fields
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            '(unknown file)',
            0,
            message,
            (),
            exc_info
        )
        
        if extra_fields:
            record.extra_fields = extra_fields
        
        self.logger.handle(record)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self._log(logging.ERROR, message, **kwargs)


def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """Get or create a structured logger.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, level)


def set_request_id(request_id: str):
    """Set request ID in context for correlation.
    
    Args:
        request_id: Unique request identifier
    """
    request_id_ctx.set(request_id)


def set_user_id(user_id: str):
    """Set user ID in context for correlation.
    
    Args:
        user_id: User identifier
    """
    user_id_ctx.set(user_id)


def clear_context():
    """Clear all context variables."""
    request_id_ctx.set(None)
    user_id_ctx.set(None)


class LogContext:
    """Context manager for setting correlation IDs."""

    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        """Initialize log context.
        
        Args:
            request_id: Request ID to set
            user_id: User ID to set
        """
        self.request_id = request_id
        self.user_id = user_id
        self.previous_request_id = None
        self.previous_user_id = None

    def __enter__(self):
        """Enter context and set IDs."""
        self.previous_request_id = request_id_ctx.get()
        self.previous_user_id = user_id_ctx.get()
        
        if self.request_id:
            set_request_id(self.request_id)
        if self.user_id:
            set_user_id(self.user_id)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous IDs."""
        request_id_ctx.set(self.previous_request_id)
        user_id_ctx.set(self.previous_user_id)


# Example usage:
# logger = get_logger(__name__)
# logger.info("Processing trade", trade_id="12345", symbol="AAPL", quantity=100)
# 
# with LogContext(request_id="req-123", user_id="user-456"):
#     logger.info("User action", action="place_order")
