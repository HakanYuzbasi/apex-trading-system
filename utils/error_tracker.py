"""Error tracking and reporting utility for production environments."""

import sys
import traceback
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from functools import wraps

try:
    from utils.structured_logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class ErrorTracker:
    """Track and report errors with context and metadata."""

    def __init__(self, error_log_path: Optional[str] = None):
        """Initialize error tracker.
        
        Args:
            error_log_path: Path to error log file (default: logs/errors.json)
        """
        self.error_log_path = error_log_path or "logs/errors.json"
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Ensure log directory exists."""
        log_dir = Path(self.error_log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    def capture_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        level: str = "error"
    ) -> str:
        """Capture and log exception with context.
        
        Args:
            exception: Exception to capture
            context: Additional context data
            user_id: User ID if applicable
            request_id: Request ID for correlation
            level: Error level (error, warning, critical)
        
        Returns:
            Error ID for tracking
        """
        error_id = self._generate_error_id()
        
        error_data = {
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "type": exception.__class__.__name__,
            "message": str(exception),
            "traceback": traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__
            ),
            "context": context or {},
            "user_id": user_id,
            "request_id": request_id,
            "python_version": sys.version,
        }

        # Log to structured logger
        logger.error(
            f"Exception captured: {exception.__class__.__name__}",
            error_id=error_id,
            exception_type=exception.__class__.__name__,
            exception_message=str(exception),
            exc_info=True,
            **context or {}
        )

        # Write to error log file
        self._write_error_log(error_data)

        return error_id

    def _generate_error_id(self) -> str:
        """Generate unique error ID.
        
        Returns:
            Unique error identifier
        """
        import uuid
        return f"err_{uuid.uuid4().hex[:12]}"

    def _write_error_log(self, error_data: Dict[str, Any]):
        """Write error to log file.
        
        Args:
            error_data: Error data to write
        """
        try:
            with open(self.error_log_path, "a") as f:
                f.write(json.dumps(error_data) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write error log: {e}")

    def get_recent_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent errors from log.
        
        Args:
            limit: Maximum number of errors to return
        
        Returns:
            List of recent errors
        """
        errors = []
        try:
            if not Path(self.error_log_path).exists():
                return errors

            with open(self.error_log_path, "r") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        errors.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read error log: {e}")

        return errors

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        errors = self.get_recent_errors(limit=1000)
        
        stats = {
            "total_errors": len(errors),
            "by_type": {},
            "by_level": {},
            "recent_24h": 0,
        }

        from datetime import timedelta
        now = datetime.utcnow()
        day_ago = now - timedelta(days=1)

        for error in errors:
            # Count by type
            error_type = error.get("type", "unknown")
            stats["by_type"][error_type] = stats["by_type"].get(error_type, 0) + 1

            # Count by level
            level = error.get("level", "error")
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

            # Count recent errors
            try:
                error_time = datetime.fromisoformat(error["timestamp"].replace("Z", ""))
                if error_time > day_ago:
                    stats["recent_24h"] += 1
            except Exception:
                pass

        return stats


# Global error tracker instance
_global_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Get global error tracker instance.
    
    Returns:
        Global ErrorTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ErrorTracker()
    return _global_tracker


def track_errors(context: Optional[Dict[str, Any]] = None):
    """Decorator to automatically track errors in functions.
    
    Args:
        context: Additional context to include with errors
    
    Example:
        @track_errors(context={"module": "trading"})
        def execute_trade(symbol, quantity):
            # Trade logic here
            pass
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    **(context or {})
                }
                error_id = get_error_tracker().capture_exception(
                    e,
                    context=error_context
                )
                logger.error(f"Error tracked with ID: {error_id}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    **(context or {})
                }
                error_id = get_error_tracker().capture_exception(
                    e,
                    context=error_context
                )
                logger.error(f"Error tracked with ID: {error_id}")
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Example usage:
# 
# # Manual error tracking
# try:
#     risky_operation()
# except Exception as e:
#     error_id = get_error_tracker().capture_exception(
#         e,
#         context={"operation": "trade_execution", "symbol": "AAPL"},
#         user_id="user_123",
#         request_id="req_456"
#     )
#     print(f"Error tracked: {error_id}")
# 
# # Automatic error tracking with decorator
# @track_errors(context={"service": "order_manager"})
# async def place_order(symbol: str, quantity: int):
#     # Order logic here
#     pass
# 
# # Get error statistics
# stats = get_error_tracker().get_error_stats()
# print(f"Total errors: {stats['total_errors']}")
# print(f"Errors in last 24h: {stats['recent_24h']}")
