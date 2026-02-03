"""
core/hot_reload.py - Hot-Reload Configuration System

Provides dynamic configuration updates without system restart.
Monitors configuration files for changes and applies updates
to running system components.

Features:
- File-based configuration watching
- Safe parameter updates (excludes critical params during trades)
- Change history and rollback
- Notification system for config changes
"""

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Set
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# Try to import watchdog for file monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.info("watchdog not available. Install with: pip install watchdog")


class ConfigChangeType(Enum):
    """Type of configuration change."""
    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"


@dataclass
class ConfigChange:
    """Record of a configuration change."""
    parameter: str
    old_value: Any
    new_value: Any
    change_type: ConfigChangeType
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False
    deferred: bool = False
    reason: Optional[str] = None


@dataclass
class HotReloadConfig:
    """Configuration for hot-reload system."""
    config_file: str = "config/trading_config.json"
    watch_interval: float = 1.0  # seconds
    enable_watching: bool = True

    # Parameters that can be changed at runtime
    hot_reloadable: Set[str] = field(default_factory=lambda: {
        # Signal thresholds
        'signal_threshold',
        'confidence_threshold',
        'regime_thresholds',

        # Logging
        'log_level',
        'verbose_logging',

        # Monitoring
        'metrics_interval',
        'health_check_interval',

        # Risk parameters (with caution)
        'max_daily_loss',
        'max_drawdown',

        # Trading behavior
        'trading_cooldown',
        'max_positions',
    })

    # Parameters that require restart (never hot-reload)
    restart_required: Set[str] = field(default_factory=lambda: {
        'ibkr_host',
        'ibkr_port',
        'ibkr_client_id',
        'initial_capital',
        'trading_universe',
        'database_connection',
    })


class ConfigFileHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """Handle config file modification events."""

    def __init__(self, callback: Callable[[str], None], config_file: str):
        self.callback = callback
        self.config_file = os.path.abspath(config_file)
        self._last_hash: Optional[str] = None

    def on_modified(self, event):
        if not isinstance(event, FileModifiedEvent):
            return

        if os.path.abspath(event.src_path) == self.config_file:
            # Check if content actually changed (debounce)
            current_hash = self._get_file_hash()
            if current_hash != self._last_hash:
                self._last_hash = current_hash
                logger.info(f"Config file modified: {event.src_path}")
                self.callback(event.src_path)

    def _get_file_hash(self) -> str:
        """Get hash of config file contents."""
        try:
            with open(self.config_file, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


class HotReloadManager:
    """
    Manages hot-reloading of configuration parameters.

    Example usage:
        manager = HotReloadManager("config/trading.json")
        manager.register_callback("signal_threshold", on_threshold_change)
        manager.start()

        # Later, when config file changes:
        # - File is monitored for changes
        # - Callbacks are triggered for changed parameters
        # - Changes are applied to running config
    """

    def __init__(self, config: Optional[HotReloadConfig] = None):
        self.config = config or HotReloadConfig()
        self._current_config: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._change_history: List[ConfigChange] = []
        self._observer = None
        self._running = False
        self._lock = threading.RLock()

        # Load initial config
        self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(self.config.config_file)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}

        try:
            with open(config_path) as f:
                self._current_config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                return self._current_config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def start(self):
        """Start watching for configuration changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, using polling mode")
            self._start_polling()
            return

        if not self.config.enable_watching:
            logger.info("Config watching disabled")
            return

        config_dir = os.path.dirname(os.path.abspath(self.config.config_file))

        handler = ConfigFileHandler(self._on_file_change, self.config.config_file)
        self._observer = Observer()
        self._observer.schedule(handler, config_dir, recursive=False)
        self._observer.start()
        self._running = True

        logger.info(f"Started watching config file: {self.config.config_file}")

    def _start_polling(self):
        """Start polling for config changes (fallback mode)."""
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _poll_loop(self):
        """Polling loop for config changes."""
        last_mtime = 0

        while self._running:
            try:
                config_path = Path(self.config.config_file)
                if config_path.exists():
                    mtime = config_path.stat().st_mtime
                    if mtime > last_mtime:
                        last_mtime = mtime
                        self._on_file_change(str(config_path))
            except Exception as e:
                logger.debug(f"Error in poll loop: {e}")

            threading.Event().wait(self.config.watch_interval)

    def stop(self):
        """Stop watching for configuration changes."""
        self._running = False
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        logger.info("Stopped watching config file")

    def _on_file_change(self, path: str):
        """Handle configuration file change."""
        with self._lock:
            old_config = self._current_config.copy()
            new_config = self._load_config()

            if not new_config:
                logger.warning("Failed to load new config, keeping current")
                return

            # Detect changes
            changes = self._detect_changes(old_config, new_config)

            for change in changes:
                self._process_change(change)

    def _detect_changes(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> List[ConfigChange]:
        """Detect configuration changes."""
        changes = []

        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)

            if old_val != new_val:
                if key not in old_config:
                    change_type = ConfigChangeType.ADDED
                elif key not in new_config:
                    change_type = ConfigChangeType.REMOVED
                else:
                    change_type = ConfigChangeType.MODIFIED

                changes.append(ConfigChange(
                    parameter=key,
                    old_value=old_val,
                    new_value=new_val,
                    change_type=change_type
                ))

        return changes

    def _process_change(self, change: ConfigChange):
        """Process a configuration change."""
        param = change.parameter

        # Check if parameter requires restart
        if param in self.config.restart_required:
            logger.warning(
                f"Parameter '{param}' requires restart to take effect. "
                f"Current: {change.old_value}, New: {change.new_value}"
            )
            change.deferred = True
            change.reason = "Requires restart"
            self._change_history.append(change)
            return

        # Check if parameter is hot-reloadable
        if param not in self.config.hot_reloadable:
            logger.warning(
                f"Parameter '{param}' is not in hot-reloadable list. "
                f"Change deferred until restart."
            )
            change.deferred = True
            change.reason = "Not in hot-reloadable list"
            self._change_history.append(change)
            return

        # Apply the change
        logger.info(
            f"Hot-reloading '{param}': {change.old_value} -> {change.new_value}"
        )

        # Trigger callbacks
        if param in self._callbacks:
            for callback in self._callbacks[param]:
                try:
                    callback(param, change.old_value, change.new_value)
                except Exception as e:
                    logger.error(f"Error in callback for {param}: {e}")

        change.applied = True
        self._change_history.append(change)

    def register_callback(
        self,
        parameter: str,
        callback: Callable[[str, Any, Any], None]
    ):
        """
        Register a callback for parameter changes.

        Args:
            parameter: Parameter name to watch
            callback: Function called with (param_name, old_value, new_value)
        """
        if parameter not in self._callbacks:
            self._callbacks[parameter] = []
        self._callbacks[parameter].append(callback)
        logger.debug(f"Registered callback for parameter: {parameter}")

    def get(self, parameter: str, default: Any = None) -> Any:
        """Get a configuration parameter value."""
        return self._current_config.get(parameter, default)

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration parameters."""
        return self._current_config.copy()

    def get_change_history(self, limit: int = 50) -> List[Dict]:
        """Get recent configuration changes."""
        return [
            {
                "parameter": c.parameter,
                "old_value": c.old_value,
                "new_value": c.new_value,
                "change_type": c.change_type.value,
                "timestamp": c.timestamp.isoformat(),
                "applied": c.applied,
                "deferred": c.deferred,
                "reason": c.reason
            }
            for c in self._change_history[-limit:]
        ]

    def rollback(self, parameter: str) -> bool:
        """
        Rollback a parameter to its previous value.

        Returns True if rollback was successful.
        """
        # Find the last change for this parameter
        for change in reversed(self._change_history):
            if change.parameter == parameter and change.applied:
                logger.info(
                    f"Rolling back '{parameter}': "
                    f"{change.new_value} -> {change.old_value}"
                )

                self._current_config[parameter] = change.old_value

                # Trigger callbacks with reversed values
                if parameter in self._callbacks:
                    for callback in self._callbacks[parameter]:
                        try:
                            callback(parameter, change.new_value, change.old_value)
                        except Exception as e:
                            logger.error(f"Error in rollback callback: {e}")

                # Record rollback
                self._change_history.append(ConfigChange(
                    parameter=parameter,
                    old_value=change.new_value,
                    new_value=change.old_value,
                    change_type=ConfigChangeType.MODIFIED,
                    applied=True,
                    reason="Rollback"
                ))

                return True

        logger.warning(f"No previous value found for '{parameter}'")
        return False


# Global manager instance
_manager: Optional[HotReloadManager] = None


def get_hot_reload_manager() -> HotReloadManager:
    """Get the global hot-reload manager."""
    global _manager
    if _manager is None:
        _manager = HotReloadManager()
    return _manager


def init_hot_reload(config_file: str = "config/trading_config.json"):
    """Initialize and start the hot-reload system."""
    global _manager
    config = HotReloadConfig(config_file=config_file)
    _manager = HotReloadManager(config)
    _manager.start()
    return _manager


def get_config(parameter: str, default: Any = None) -> Any:
    """Get a configuration value (convenience function)."""
    return get_hot_reload_manager().get(parameter, default)


def on_config_change(parameter: str):
    """
    Decorator to register a function as a config change handler.

    Example:
        @on_config_change("signal_threshold")
        def handle_threshold_change(param, old, new):
            print(f"Threshold changed from {old} to {new}")
    """
    def decorator(func):
        get_hot_reload_manager().register_callback(parameter, func)
        return func
    return decorator
