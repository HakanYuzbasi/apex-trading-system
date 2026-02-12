"""Health check endpoints for monitoring system status."""

from fastapi import APIRouter, Response, status
from datetime import datetime
import psutil
import asyncio
from typing import Dict, Any
import sys

try:
    from utils.db_pool import get_db_pool
except ImportError:
    get_db_pool = None

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint.
    
    Returns:
        Simple status response
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/liveness")
async def liveness() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint.
    
    Checks if the application is running.
    Returns 200 if alive, 503 if not.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/readiness")
async def readiness(response: Response) -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint.
    
    Checks if the application is ready to accept traffic.
    Validates critical dependencies like database.
    """
    checks = {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": {}
    }
    
    # Check database connection if available
    if get_db_pool:
        try:
            pool = get_db_pool()
            db_healthy = await pool.health_check()
            checks["checks"]["database"] = "healthy" if db_healthy else "unhealthy"
            if not db_healthy:
                checks["status"] = "not_ready"
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        except Exception as e:
            checks["checks"]["database"] = f"error: {str(e)}"
            checks["status"] = "not_ready"
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return checks


@router.get("/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with system metrics.
    
    Returns comprehensive system health information including:
    - CPU usage
    - Memory usage
    - Disk usage
    - Database connection pool status
    - Python version
    """
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "system": {
            "python_version": sys.version,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_count": psutil.cpu_count(),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent,
            },
        },
        "services": {}
    }
    
    # Database health
    if get_db_pool:
        try:
            pool = get_db_pool()
            db_healthy = await pool.health_check()
            pool_status = pool.get_pool_status()
            health_data["services"]["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "pool": pool_status
            }
        except Exception as e:
            health_data["services"]["database"] = {
                "status": "error",
                "error": str(e)
            }
    
    return health_data


@router.get("/metrics")
async def health_metrics() -> Dict[str, Any]:
    """Prometheus-style health metrics.
    
    Returns metrics in a format suitable for monitoring systems.
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_bytes": memory.available,
            "memory_total_bytes": memory.total,
            "disk_usage_percent": disk.percent,
            "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total,
        }
    }
    
    # Add database pool metrics if available
    if get_db_pool:
        try:
            pool = get_db_pool()
            pool_status = pool.get_pool_status()
            if pool_status.get("initialized"):
                metrics["metrics"]["db_pool_size"] = pool_status.get("size", 0)
                metrics["metrics"]["db_pool_checked_in"] = pool_status.get("checked_in", 0)
                metrics["metrics"]["db_pool_checked_out"] = pool_status.get("checked_out", 0)
                metrics["metrics"]["db_pool_overflow"] = pool_status.get("overflow", 0)
        except Exception:
            pass
    
    return metrics


# Example usage in server.py:
# from api.health import router as health_router
# app.include_router(health_router)
