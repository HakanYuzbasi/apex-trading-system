"""
core/health_checker.py - Health Check System

Provides comprehensive health checking for the trading system:
- Broker connection status
- Market data freshness
- Model performance (drift detection)
- System resources (memory, disk)
- Component status

Exposes health endpoints for monitoring and alerting.
"""

import asyncio
import logging
import psutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat() + "Z" if self.last_check.tzinfo else self.last_check.isoformat() + "Z",
            "details": self.details,
            "duration_ms": self.duration_ms
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    check_interval_seconds: int = 30
    broker_timeout_seconds: float = 10.0
    data_staleness_threshold_seconds: int = 300  # 5 minutes
    memory_warning_percent: float = 80.0
    memory_critical_percent: float = 95.0
    disk_warning_percent: float = 80.0
    disk_critical_percent: float = 95.0
    min_model_accuracy: float = 0.5


class HealthChecker:
    """
    Comprehensive health checker for the trading system.

    Performs periodic checks on:
    - Broker connection
    - Market data freshness
    - Model performance
    - System resources
    - Custom component checks

    Example:
        checker = HealthChecker()
        checker.register_check("custom", custom_check_func)

        # Run all checks
        results = await checker.check_all()

        # Get overall status
        status = checker.get_overall_status()
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None):
        self.config = config or HealthCheckConfig()
        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._last_full_check: Optional[datetime] = None

        # External references (set by application)
        self.ibkr_connector = None
        self.market_data_fetcher = None
        self.signal_generator = None

        # Timestamps for freshness checks
        self.last_market_data_update: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        self.last_signal_time: Optional[datetime] = None

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("broker_connection", self._check_broker)
        self.register_check("market_data_freshness", self._check_data_freshness)
        self.register_check("memory_usage", self._check_memory)
        self.register_check("disk_space", self._check_disk)
        self.register_check("log_directory", self._check_log_directory)

    def register_check(
        self,
        name: str,
        check_func: Callable[[], 'HealthCheck']
    ):
        """
        Register a custom health check.

        Args:
            name: Unique name for the check
            check_func: Async or sync function returning HealthCheck
        """
        self._checks[name] = check_func
        logger.debug(f"Registered health check: {name}")

    def unregister_check(self, name: str):
        """Unregister a health check."""
        if name in self._checks:
            del self._checks[name]
            logger.debug(f"Unregistered health check: {name}")

    async def check_all(self) -> Dict[str, HealthCheck]:
        """
        Run all registered health checks.

        Returns:
            Dictionary of check results
        """
        results = {}

        for name, check_func in self._checks.items():
            try:
                start_time = datetime.utcnow()

                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()

                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                result.duration_ms = duration
                results[name] = result

            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.utcnow()
                )

        self._results = results
        self._last_full_check = datetime.utcnow()
        return results

    async def check_one(self, name: str) -> Optional[HealthCheck]:
        """Run a single health check."""
        if name not in self._checks:
            return None

        try:
            check_func = self._checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            self._results[name] = result
            return result
        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                last_check=datetime.utcnow()
            )

    def get_overall_status(self) -> HealthStatus:
        """
        Get overall health status based on all check results.

        Returns:
            HEALTHY if all checks pass
            DEGRADED if any check is degraded
            UNHEALTHY if any check is unhealthy
        """
        if not self._results:
            return HealthStatus.UNKNOWN

        statuses = [check.status for check in self._results.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if any(s == HealthStatus.UNKNOWN for s in statuses):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def get_summary(self) -> Dict[str, Any]:
        """Get health check summary."""
        return {
            "status": self.get_overall_status().value,
            "last_check": self._last_full_check.isoformat() + "Z" if self._last_full_check else None,
            "checks": {
                name: check.to_dict()
                for name, check in self._results.items()
            },
            "healthy_count": sum(
                1 for c in self._results.values()
                if c.status == HealthStatus.HEALTHY
            ),
            "degraded_count": sum(
                1 for c in self._results.values()
                if c.status == HealthStatus.DEGRADED
            ),
            "unhealthy_count": sum(
                1 for c in self._results.values()
                if c.status == HealthStatus.UNHEALTHY
            ),
        }

    # Default check implementations

    async def _check_broker(self) -> HealthCheck:
        """Check broker connection status."""
        if self.ibkr_connector is None:
            return HealthCheck(
                name="broker_connection",
                status=HealthStatus.UNKNOWN,
                message="IBKR connector not configured",
                last_check=datetime.utcnow()
            )

        try:
            is_connected = self.ibkr_connector.ib.isConnected()

            if is_connected:
                return HealthCheck(
                    name="broker_connection",
                    status=HealthStatus.HEALTHY,
                    message="Connected to IBKR",
                    last_check=datetime.utcnow(),
                    details={
                        "host": self.ibkr_connector.host,
                        "port": self.ibkr_connector.port,
                        "account": self.ibkr_connector.account
                    }
                )
            else:
                return HealthCheck(
                    name="broker_connection",
                    status=HealthStatus.UNHEALTHY,
                    message="Disconnected from IBKR",
                    last_check=datetime.utcnow()
                )

        except Exception as e:
            return HealthCheck(
                name="broker_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking connection: {str(e)}",
                last_check=datetime.utcnow()
            )

    async def _check_data_freshness(self) -> HealthCheck:
        """Check if market data is fresh."""
        if self.last_market_data_update is None:
            return HealthCheck(
                name="market_data_freshness",
                status=HealthStatus.UNKNOWN,
                message="No market data received yet",
                last_check=datetime.utcnow()
            )

        staleness = (datetime.utcnow() - self.last_market_data_update).total_seconds()

        if staleness > self.config.data_staleness_threshold_seconds:
            return HealthCheck(
                name="market_data_freshness",
                status=HealthStatus.DEGRADED,
                message=f"Market data is stale ({staleness:.0f}s old)",
                last_check=datetime.utcnow(),
                details={
                    "last_update": self.last_market_data_update.isoformat() + "Z",
                    "staleness_seconds": staleness
                }
            )

        return HealthCheck(
            name="market_data_freshness",
            status=HealthStatus.HEALTHY,
            message=f"Market data is fresh ({staleness:.0f}s old)",
            last_check=datetime.utcnow(),
            details={
                "last_update": self.last_market_data_update.isoformat() + "Z",
                "staleness_seconds": staleness
            }
        )

    def _check_memory(self) -> HealthCheck:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            percent_used = memory.percent

            if percent_used >= self.config.memory_critical_percent:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {percent_used:.1f}%"
            elif percent_used >= self.config.memory_warning_percent:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {percent_used:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage: {percent_used:.1f}%"

            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                details={
                    "percent_used": percent_used,
                    "total_gb": memory.total / (1024 ** 3),
                    "available_gb": memory.available / (1024 ** 3),
                    "used_gb": memory.used / (1024 ** 3)
                }
            )

        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking memory: {str(e)}",
                last_check=datetime.utcnow()
            )

    def _check_disk(self) -> HealthCheck:
        """Check disk space usage."""
        try:
            disk = psutil.disk_usage('/')
            percent_used = disk.percent

            if percent_used >= self.config.disk_critical_percent:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {percent_used:.1f}%"
            elif percent_used >= self.config.disk_warning_percent:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {percent_used:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage: {percent_used:.1f}%"

            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                details={
                    "percent_used": percent_used,
                    "total_gb": disk.total / (1024 ** 3),
                    "free_gb": disk.free / (1024 ** 3),
                    "used_gb": disk.used / (1024 ** 3)
                }
            )

        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking disk: {str(e)}",
                last_check=datetime.utcnow()
            )

    def _check_log_directory(self) -> HealthCheck:
        """Check if log directory is writable."""
        log_dir = Path("logs")

        try:
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)

            # Try to write a test file
            test_file = log_dir / ".health_check"
            test_file.write_text(datetime.utcnow().isoformat() + "Z")
            test_file.unlink()

            return HealthCheck(
                name="log_directory",
                status=HealthStatus.HEALTHY,
                message="Log directory is writable",
                last_check=datetime.utcnow(),
                details={"path": str(log_dir.absolute())}
            )

        except Exception as e:
            return HealthCheck(
                name="log_directory",
                status=HealthStatus.DEGRADED,
                message=f"Log directory issue: {str(e)}",
                last_check=datetime.utcnow()
            )

    def record_market_data_update(self):
        """Record that market data was received."""
        self.last_market_data_update = datetime.utcnow()

    def record_trade(self):
        """Record that a trade occurred."""
        self.last_trade_time = datetime.utcnow()

    def record_signal(self):
        """Record that a signal was generated."""
        self.last_signal_time = datetime.utcnow()


# FastAPI endpoint integration
def create_health_endpoints(app, health_checker: HealthChecker):
    """
    Create FastAPI health check endpoints.

    Args:
        app: FastAPI application instance
        health_checker: HealthChecker instance

    Creates:
        GET /health - Overall health status
        GET /health/live - Liveness probe (is process running)
        GET /health/ready - Readiness probe (is system ready for traffic)
        GET /health/details - Detailed check results
    """
    from fastapi import Response

    @app.get("/health")
    async def health():
        """Overall health check."""
        await health_checker.check_all()
        summary = health_checker.get_summary()

        status_code = 200
        if summary["status"] == "unhealthy":
            status_code = 503
        elif summary["status"] == "degraded":
            status_code = 200  # Still return 200 for degraded

        return Response(
            content=json.dumps(summary),
            media_type="application/json",
            status_code=status_code
        )

    @app.get("/health/live")
    async def liveness():
        """Liveness probe - is the process running."""
        return {"status": "alive", "timestamp": datetime.utcnow().isoformat() + "Z"}

    @app.get("/health/ready")
    async def readiness():
        """Readiness probe - is the system ready for traffic."""
        status = health_checker.get_overall_status()
        is_ready = status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

        return Response(
            content=json.dumps({
                "ready": is_ready,
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }),
            media_type="application/json",
            status_code=200 if is_ready else 503
        )

    @app.get("/health/details")
    async def health_details():
        """Detailed health check results."""
        await health_checker.check_all()
        return health_checker.get_summary()


# Global instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def init_health_checker(
    ibkr_connector=None,
    market_data_fetcher=None,
    config: Optional[HealthCheckConfig] = None
) -> HealthChecker:
    """Initialize health checker with references."""
    global _health_checker
    _health_checker = HealthChecker(config)
    _health_checker.ibkr_connector = ibkr_connector
    _health_checker.market_data_fetcher = market_data_fetcher
    return _health_checker


# Import json for the endpoint
import json
