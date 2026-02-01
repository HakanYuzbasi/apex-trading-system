"""
core/feature_flags.py - Feature Flag System

Provides runtime feature toggles for enabling/disabling system components
without code changes. Supports environment variables and file-based config.

Features:
- Environment variable overrides
- JSON file configuration
- Hot-reload support (optional)
- Default values with type safety
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlags:
    """
    Feature flags for the APEX Trading System.

    All flags can be overridden via environment variables prefixed with FF_.
    Example: FF_ML_SIGNALS=false disables ML signals.
    """

    # Signal Generation
    use_ml_signals: bool = True           # Use ML-based signal generation
    use_ensemble_signals: bool = True     # Use ensemble of multiple models
    use_regime_detection: bool = True     # Enable market regime detection

    # Trading Features
    enable_live_trading: bool = True      # Enable actual order execution
    enable_short_selling: bool = True     # Allow short positions
    enable_options_trading: bool = False  # Enable options strategies

    # Execution
    use_smart_order_routing: bool = True  # Use smart order router
    use_twap_vwap: bool = True           # Use TWAP/VWAP for large orders
    use_limit_orders: bool = True        # Use limit orders when appropriate

    # Risk Management
    use_correlation_manager: bool = True  # Enable correlation monitoring
    use_circuit_breaker: bool = True      # Enable automatic trading halt
    use_dynamic_position_sizing: bool = True  # ATR-based position sizing

    # Portfolio
    enable_auto_rebalance: bool = True    # Enable automatic rebalancing
    enable_portfolio_optimizer: bool = True  # Use portfolio optimization

    # Monitoring
    enable_prometheus_metrics: bool = True   # Export Prometheus metrics
    enable_structured_logging: bool = True   # Use JSON structured logs
    enable_distributed_tracing: bool = False # OpenTelemetry tracing

    # Resilience
    use_bulkhead_pattern: bool = True     # Enable failure isolation
    use_timeout_protection: bool = True   # Enable operation timeouts

    # Development/Debug
    debug_mode: bool = False              # Enable debug logging
    dry_run_mode: bool = False            # Log orders but don't execute
    paper_trading_mode: bool = False      # Force paper trading

    @classmethod
    def from_env(cls) -> 'FeatureFlags':
        """
        Load feature flags from environment variables.

        Environment variables are prefixed with FF_ and use uppercase.
        Example: FF_USE_ML_SIGNALS=true
        """
        flags = cls()

        for field_name in flags.__dataclass_fields__:
            env_name = f"FF_{field_name.upper()}"
            env_value = os.getenv(env_name)

            if env_value is not None:
                # Parse boolean
                if env_value.lower() in ('true', '1', 'yes', 'on'):
                    setattr(flags, field_name, True)
                elif env_value.lower() in ('false', '0', 'no', 'off'):
                    setattr(flags, field_name, False)
                else:
                    logger.warning(
                        f"Invalid value for {env_name}: {env_value}, using default"
                    )

        return flags

    @classmethod
    def from_file(cls, path: str) -> 'FeatureFlags':
        """
        Load feature flags from JSON file.

        Args:
            path: Path to JSON config file

        Returns:
            FeatureFlags instance
        """
        flags = cls()

        try:
            with open(path) as f:
                data = json.load(f)

            for key, value in data.items():
                if hasattr(flags, key):
                    setattr(flags, key, bool(value))
                else:
                    logger.warning(f"Unknown feature flag in config: {key}")

        except FileNotFoundError:
            logger.warning(f"Feature flags file not found: {path}, using defaults")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing feature flags file: {e}")

        return flags

    @classmethod
    def from_env_with_file_fallback(
        cls,
        config_path: Optional[str] = None
    ) -> 'FeatureFlags':
        """
        Load flags from file, then override with environment variables.

        This allows file-based defaults with env var overrides.
        """
        if config_path and Path(config_path).exists():
            flags = cls.from_file(config_path)
        else:
            flags = cls()

        # Override with env vars
        env_flags = cls.from_env()
        for field_name in flags.__dataclass_fields__:
            env_name = f"FF_{field_name.upper()}"
            if os.getenv(env_name) is not None:
                setattr(flags, field_name, getattr(env_flags, field_name))

        return flags

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary."""
        return asdict(self)

    def save_to_file(self, path: str):
        """Save current flags to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Feature flags saved to {path}")

    def is_enabled(self, flag_name: str) -> bool:
        """
        Check if a flag is enabled.

        Args:
            flag_name: Name of the flag

        Returns:
            True if enabled, False otherwise

        Raises:
            AttributeError: If flag doesn't exist
        """
        if not hasattr(self, flag_name):
            raise AttributeError(f"Unknown feature flag: {flag_name}")
        return getattr(self, flag_name)

    def enable(self, flag_name: str):
        """Enable a flag at runtime."""
        if hasattr(self, flag_name):
            setattr(self, flag_name, True)
            logger.info(f"Feature flag enabled: {flag_name}")
        else:
            raise AttributeError(f"Unknown feature flag: {flag_name}")

    def disable(self, flag_name: str):
        """Disable a flag at runtime."""
        if hasattr(self, flag_name):
            setattr(self, flag_name, False)
            logger.info(f"Feature flag disabled: {flag_name}")
        else:
            raise AttributeError(f"Unknown feature flag: {flag_name}")

    def get_enabled_flags(self) -> list:
        """Get list of enabled flag names."""
        return [
            name for name in self.__dataclass_fields__
            if getattr(self, name)
        ]

    def get_disabled_flags(self) -> list:
        """Get list of disabled flag names."""
        return [
            name for name in self.__dataclass_fields__
            if not getattr(self, name)
        ]


class FeatureFlagManager:
    """
    Manager for feature flags with hot-reload support.

    Example:
        manager = FeatureFlagManager()
        manager.load_from_env()

        if manager.is_enabled("use_ml_signals"):
            signal = ml_generator.generate(data)
        else:
            signal = technical_generator.generate(data)
    """

    def __init__(self):
        self._flags = FeatureFlags()
        self._config_path: Optional[str] = None
        self._last_loaded: Optional[datetime] = None
        self._callbacks: Dict[str, list] = {}

    @property
    def flags(self) -> FeatureFlags:
        """Get current feature flags."""
        return self._flags

    def load_from_env(self):
        """Load flags from environment variables."""
        self._flags = FeatureFlags.from_env()
        self._last_loaded = datetime.now()
        logger.info("Feature flags loaded from environment")

    def load_from_file(self, path: str):
        """Load flags from file."""
        self._flags = FeatureFlags.from_file(path)
        self._config_path = path
        self._last_loaded = datetime.now()
        logger.info(f"Feature flags loaded from {path}")

    def load(self, config_path: Optional[str] = None):
        """Load flags from file (if provided) and environment."""
        self._flags = FeatureFlags.from_env_with_file_fallback(config_path)
        self._config_path = config_path
        self._last_loaded = datetime.now()
        logger.info("Feature flags loaded")

    def reload(self):
        """Reload flags from original source."""
        if self._config_path:
            self.load_from_file(self._config_path)
        else:
            self.load_from_env()

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a flag is enabled."""
        return self._flags.is_enabled(flag_name)

    def enable(self, flag_name: str):
        """Enable a flag and trigger callbacks."""
        old_value = getattr(self._flags, flag_name, None)
        self._flags.enable(flag_name)
        self._trigger_callbacks(flag_name, old_value, True)

    def disable(self, flag_name: str):
        """Disable a flag and trigger callbacks."""
        old_value = getattr(self._flags, flag_name, None)
        self._flags.disable(flag_name)
        self._trigger_callbacks(flag_name, old_value, False)

    def on_change(self, flag_name: str, callback: Callable[[bool, bool], None]):
        """
        Register callback for flag changes.

        Args:
            flag_name: Name of flag to watch
            callback: Function(old_value, new_value) to call on change
        """
        if flag_name not in self._callbacks:
            self._callbacks[flag_name] = []
        self._callbacks[flag_name].append(callback)

    def _trigger_callbacks(self, flag_name: str, old_value: bool, new_value: bool):
        """Trigger callbacks for a flag change."""
        if old_value == new_value:
            return

        callbacks = self._callbacks.get(flag_name, [])
        for callback in callbacks:
            try:
                callback(old_value, new_value)
            except Exception as e:
                logger.error(f"Error in flag change callback: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get status summary."""
        return {
            "enabled_flags": self._flags.get_enabled_flags(),
            "disabled_flags": self._flags.get_disabled_flags(),
            "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
            "config_path": self._config_path,
        }


# Global instance
_manager: Optional[FeatureFlagManager] = None


def get_feature_flags() -> FeatureFlags:
    """Get global feature flags instance."""
    global _manager
    if _manager is None:
        _manager = FeatureFlagManager()
        _manager.load()
    return _manager.flags


def get_flag_manager() -> FeatureFlagManager:
    """Get global feature flag manager."""
    global _manager
    if _manager is None:
        _manager = FeatureFlagManager()
        _manager.load()
    return _manager


def is_enabled(flag_name: str) -> bool:
    """Quick check if a flag is enabled."""
    return get_feature_flags().is_enabled(flag_name)


def feature_flag(flag_name: str, fallback: Optional[Callable] = None):
    """
    Decorator to conditionally execute function based on feature flag.

    Args:
        flag_name: Name of the feature flag
        fallback: Optional function to call if flag is disabled

    Example:
        @feature_flag("use_ml_signals")
        async def generate_ml_signal(data):
            ...

        @feature_flag("use_ml_signals", fallback=generate_simple_signal)
        async def generate_signal(data):
            # Uses ML if enabled, otherwise calls generate_simple_signal
            ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            if is_enabled(flag_name):
                return await func(*args, **kwargs)
            elif fallback:
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                return fallback(*args, **kwargs)
            return None

        def sync_wrapper(*args, **kwargs):
            if is_enabled(flag_name):
                return func(*args, **kwargs)
            elif fallback:
                return fallback(*args, **kwargs)
            return None

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
