"""
automation - Fully Automated Trading System Components

Modules:
- watchdog: Process supervisor with auto-restart
- alerts: Multi-channel notification system
"""

from .watchdog import TradingWatchdog, AlertManager, HealthMonitor

__all__ = ['TradingWatchdog', 'AlertManager', 'HealthMonitor']
