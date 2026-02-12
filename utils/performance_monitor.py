"""Performance monitoring and profiling utility for tracking execution times."""

import time
import functools
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import logging

try:
    from utils.structured_logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


class PerformanceMetric:
    """Store performance metrics for a function or operation."""

    def __init__(self, name: str):
        self.name = name
        self.execution_times: List[float] = []
        self.error_count = 0
        self.success_count = 0
        self.last_execution: Optional[datetime] = None

    def add_execution(self, duration: float, success: bool = True):
        """Record an execution."""
        self.execution_times.append(duration)
        self.last_execution = datetime.utcnow()
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        # Keep only last 1000 executions
        if len(self.execution_times) > 1000:
            self.execution_times = self.execution_times[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.execution_times:
            return {
                "name": self.name,
                "executions": 0,
                "success_count": 0,
                "error_count": 0,
            }

        return {
            "name": self.name,
            "executions": len(self.execution_times),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0,
            "avg_duration_ms": statistics.mean(self.execution_times) * 1000,
            "min_duration_ms": min(self.execution_times) * 1000,
            "max_duration_ms": max(self.execution_times) * 1000,
            "median_duration_ms": statistics.median(self.execution_times) * 1000,
            "p95_duration_ms": self._percentile(self.execution_times, 95) * 1000,
            "p99_duration_ms": self._percentile(self.execution_times, 99) * 1000,
            "last_execution": self.last_execution.isoformat() + "Z" if self.last_execution else None,
        }

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[min(index, len(sorted_data) - 1)]


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetric] = defaultdict(lambda: PerformanceMetric(""))
        self.start_time = datetime.utcnow()

    def record(self, name: str, duration: float, success: bool = True):
        """Record a performance metric.
        
        Args:
            name: Metric name
            duration: Execution duration in seconds
            success: Whether execution was successful
        """
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetric(name)
        self.metrics[name].add_execution(duration, success)

    def get_metric(self, name: str) -> Dict[str, Any]:
        """Get statistics for a specific metric.
        
        Args:
            name: Metric name
        
        Returns:
            Performance statistics
        """
        if name in self.metrics:
            return self.metrics[name].get_stats()
        return {"name": name, "executions": 0}

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all performance metrics.
        
        Returns:
            List of all metric statistics
        """
        return [metric.get_stats() for metric in self.metrics.values()]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.
        
        Returns:
            Summary statistics
        """
        all_metrics = self.get_all_metrics()
        
        total_executions = sum(m["executions"] for m in all_metrics)
        total_errors = sum(m["error_count"] for m in all_metrics)
        
        uptime = datetime.utcnow() - self.start_time
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "total_operations": total_executions,
            "total_errors": total_errors,
            "unique_operations": len(self.metrics),
            "metrics": sorted(all_metrics, key=lambda x: x.get("avg_duration_ms", 0), reverse=True)[:10]  # Top 10 slowest
        }


# Global performance monitor
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance.
    
    Returns:
        Global PerformanceMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def monitor_performance(name: Optional[str] = None):
    """Decorator to monitor function performance.
    
    Args:
        name: Custom metric name (defaults to function name)
    
    Example:
        @monitor_performance()
        async def fetch_market_data(symbol):
            # Implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                get_performance_monitor().record(metric_name, duration, success)
                
                # Log slow operations (> 1 second)
                if duration > 1.0:
                    logger.warning(
                        f"Slow operation detected: {metric_name}",
                        duration_ms=duration * 1000,
                        function=func.__name__
                    )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                get_performance_monitor().record(metric_name, duration, success)
                
                # Log slow operations (> 1 second)
                if duration > 1.0:
                    logger.warning(
                        f"Slow operation detected: {metric_name}",
                        duration_ms=duration * 1000,
                        function=func.__name__
                    )

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class PerformanceTimer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str):
        """Initialize timer.
        
        Args:
            name: Metric name
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        success = exc_type is None
        
        get_performance_monitor().record(self.name, duration, success)
        
        if duration > 1.0:
            logger.warning(
                f"Slow operation: {self.name}",
                duration_ms=duration * 1000
            )

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


# Example usage:
# 
# # Using decorator
# @monitor_performance(name="trade_execution")
# async def execute_trade(symbol: str, quantity: int):
#     # Trade logic
#     pass
# 
# # Using context manager
# with PerformanceTimer("data_processing"):
#     process_large_dataset()
# 
# # Get metrics
# monitor = get_performance_monitor()
# summary = monitor.get_summary()
# print(f"Total operations: {summary['total_operations']}")
# print(f"Uptime: {summary['uptime_seconds']} seconds")
# 
# # Get specific metric
# trade_stats = monitor.get_metric("trade_execution")
# print(f"Avg execution time: {trade_stats['avg_duration_ms']}ms")
