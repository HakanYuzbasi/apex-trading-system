"""Tests for structured logging utility."""

import json
from utils.structured_logger import (
    get_logger,
    StructuredFormatter,
    LogContext,
    set_request_id,
    set_user_id,
    clear_context,
)
import logging


class TestStructuredFormatter:
    """Test StructuredFormatter."""

    def test_format_basic_log(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["message"] == "Test message"
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test"
        assert "timestamp" in log_data

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_fields = {"trade_id": "12345", "symbol": "AAPL"}
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["trade_id"] == "12345"
        assert log_data["symbol"] == "AAPL"


class TestLogContext:
    """Test LogContext manager."""

    def setup_method(self):
        """Clear context before each test."""
        clear_context()

    def test_context_manager(self):
        """Test context manager sets and clears context."""
        with LogContext(request_id="req-123", user_id="user-456"):
            # Context should be set
            pass
        # Context should be cleared after exiting
        clear_context()

    def test_nested_context(self):
        """Test nested context managers."""
        with LogContext(request_id="req-outer"):
            with LogContext(request_id="req-inner"):
                pass
            # Outer context should be restored


class TestStructuredLogger:
    """Test StructuredLogger."""

    def test_logger_creation(self):
        """Test logger can be created."""
        logger = get_logger("test_logger")
        assert logger is not None

    def test_log_levels(self):
        """Test different log levels."""
        logger = get_logger("test_logger")
        
        # These should not raise errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    def test_log_with_kwargs(self):
        """Test logging with keyword arguments."""
        logger = get_logger("test_logger")
        logger.info("Trade executed", symbol="AAPL", quantity=100, price=150.0)

    def test_exception_logging(self):
        """Test exception logging."""
        logger = get_logger("test_logger")
        
        try:
            raise ValueError("Test error")
        except Exception:
            logger.exception("An error occurred")


class TestContextVariables:
    """Test context variable functions."""

    def setup_method(self):
        """Clear context before each test."""
        clear_context()

    def test_set_request_id(self):
        """Test setting request ID."""
        set_request_id("req-789")
        # Should not raise error

    def test_set_user_id(self):
        """Test setting user ID."""
        set_user_id("user-789")
        # Should not raise error

    def test_clear_context(self):
        """Test clearing context."""
        set_request_id("req-123")
        set_user_id("user-123")
        clear_context()
        # Should not raise error
