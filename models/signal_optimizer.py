"""
models/signal_optimizer.py - Signal Generation Optimizer
Optimizes signal generation for maximum accuracy and minimum latency.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import wraps
import time
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for signal optimization."""
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    enable_parallel: bool = True
    max_workers: int = 4
    enable_feature_selection: bool = True
    top_k_features: int = 50
    enable_ensemble_optimization: bool = True
    latency_threshold_ms: float = 100.0
    accuracy_target: float = 0.60


@dataclass
class SignalMetrics:
    """Metrics for signal generation performance."""
    generation_time_ms: float
    cache_hit_rate: float
    feature_count: int
    signal_strength: float
    confidence: float
    accuracy_estimate: float


class SignalOptimizer:
    """
    Optimizes signal generation for accuracy and speed.
    
    Features:
    - Feature caching and memoization
    - Parallel processing for multiple symbols
    - Dynamic feature selection
    - Ensemble optimization
    - Latency monitoring
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.feature_cache = {}
        self.signal_cache = {}
        self.feature_importance = {}
        self.performance_history = []
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_generations = 0
        
        logger.info(f"SignalOptimizer initialized with config: {config}")
    
    def optimize_signal_generation(
        self,
        data: pd.DataFrame,
        symbol: str,
        signal_generator,
        **kwargs
    ) -> Tuple[Any, SignalMetrics]:
        """
        Generate optimized signal with performance tracking.
        
        Args:
            data: Market data DataFrame
            symbol: Symbol identifier
            signal_generator: Signal generator instance
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (signal, metrics)
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = self._get_cache_key(data, symbol, kwargs)
        cached_signal = self._get_cached_signal(cache_key)
        
        if cached_signal is not None:
            generation_time = (time.perf_counter() - start_time) * 1000
            metrics = SignalMetrics(
                generation_time_ms=generation_time,
                cache_hit_rate=self._get_cache_hit_rate(),
                feature_count=len(getattr(cached_signal, '__dict__', {})),
                signal_strength=getattr(cached_signal, 'signal', 0),
                confidence=getattr(cached_signal, 'confidence', 0),
                accuracy_estimate=self._estimate_accuracy(cached_signal)
            )
            return cached_signal, metrics
        
        # Generate signal with optimizations
        signal = self._generate_optimized_signal(data, symbol, signal_generator, **kwargs)
        
        # Cache the result
        self._cache_signal(cache_key, signal)
        
        # Update metrics
        generation_time = (time.perf_counter() - start_time) * 1000
        metrics = SignalMetrics(
            generation_time_ms=generation_time,
            cache_hit_rate=self._get_cache_hit_rate(),
            feature_count=len(getattr(signal, '__dict__', {})),
            signal_strength=getattr(signal, 'signal', 0),
            confidence=getattr(signal, 'confidence', 0),
            accuracy_estimate=self._estimate_accuracy(signal)
        )
        
        # Track performance
        self.performance_history.append(metrics)
        
        # Auto-optimize if latency is high
        if generation_time > self.config.latency_threshold_ms:
            self._trigger_optimization()
        
        return signal, metrics
    
    def _generate_optimized_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        signal_generator,
        **kwargs
    ) -> Any:
        """Generate signal with all optimizations applied."""
        
        # Feature selection optimization
        if self.config.enable_feature_selection:
            data = self._select_optimal_features(data, symbol)
        
        # Generate signal
        if hasattr(signal_generator, 'generate_signal'):
            signal = signal_generator.generate_signal(data, symbol, **kwargs)
        elif hasattr(signal_generator, 'predict'):
            signal = signal_generator.predict(data, **kwargs)
        else:
            raise ValueError("Signal generator must have generate_signal or predict method")
        
        # Ensemble optimization
        if self.config.enable_ensemble_optimization and hasattr(signal_generator, 'ensemble'):
            signal = self._optimize_ensemble(signal_generator, data, symbol)
        
        return signal
    
    def _select_optimal_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Select most important features for the symbol."""
        if symbol not in self.feature_importance:
            # Initialize with all features
            self.feature_importance[symbol] = {
                col: 1.0 for col in data.columns
            }
            return data
        
        # Get top features
        importance = self.feature_importance[symbol]
        top_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.config.top_k_features]
        
        feature_names = [f[0] for f in top_features]
        
        # Ensure we have the features
        available_features = [f for f in feature_names if f in data.columns]
        
        if available_features:
            return data[available_features]
        
        return data
    
    def _optimize_ensemble(self, signal_generator, data: pd.DataFrame, symbol: str) -> Any:
        """Optimize ensemble weights for better performance."""
        if not hasattr(signal_generator, 'ensemble'):
            return None
        
        # Get ensemble predictions
        ensemble_predictions = []
        for model in signal_generator.ensemble.models:
            pred = model.predict(data)
            ensemble_predictions.append(pred)
        
        # Optimize weights based on recent performance
        weights = self._get_optimal_weights(symbol, len(ensemble_predictions))
        
        # Weighted combination
        if hasattr(ensemble_predictions[0], 'signal'):
            # SignalOutput objects
            weighted_signal = sum(
                pred.signal * weight 
                for pred, weight in zip(ensemble_predictions, weights)
            )
            
            # Create combined signal
            combined = ensemble_predictions[0]
            combined.signal = weighted_signal
            combined.confidence = np.mean([
                pred.confidence for pred in ensemble_predictions
            ])
            return combined
        else:
            # Numeric predictions
            return sum(pred * weight for pred, weight in zip(ensemble_predictions, weights))
    
    def _get_optimal_weights(self, symbol: str, n_models: int) -> np.ndarray:
        """Get optimal weights for ensemble models."""
        # Simple equal weights for now, can be enhanced with ML
        return np.ones(n_models) / n_models
    
    async def optimize_batch_signals(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signal_generator,
        **kwargs
    ) -> Dict[str, Tuple[Any, SignalMetrics]]:
        """
        Optimize signal generation for multiple symbols in parallel.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            signal_generator: Signal generator instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of symbol -> (signal, metrics)
        """
        if not self.config.enable_parallel:
            # Sequential processing
            results = {}
            for symbol, data in data_dict.items():
                signal, metrics = self.optimize_signal_generation(
                    data, symbol, signal_generator, **kwargs
                )
                results[symbol] = (signal, metrics)
            return results
        
        # Parallel processing
        tasks = []
        for symbol, data in data_dict.items():
            task = asyncio.create_task(
                asyncio.to_thread(
                    self.optimize_signal_generation,
                    data, symbol, signal_generator, **kwargs
                )
            )
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                signal, metrics = await task
                results[symbol] = (signal, metrics)
            except Exception as e:
                logger.error(f"Failed to generate signal for {symbol}: {e}")
                results[symbol] = (None, None)
        
        return results
    
    def _get_cache_key(self, data: pd.DataFrame, symbol: str, kwargs: dict) -> str:
        """Generate cache key for data and parameters."""
        # Use last few rows and hash for key
        if len(data) > 0:
            data_hash = hashlib.md5(
                str(data.tail(5).values.tobytes()).encode()
            ).hexdigest()
        else:
            data_hash = "empty"
        
        kwargs_hash = hashlib.md5(
            str(sorted(kwargs.items())).encode()
        ).hexdigest()
        
        return f"{symbol}_{data_hash}_{kwargs_hash}"
    
    def _get_cached_signal(self, cache_key: str) -> Optional[Any]:
        """Get cached signal if valid."""
        if not self.config.enable_caching:
            return None
        
        if cache_key in self.signal_cache:
            cached_time, signal = self.signal_cache[cache_key]
            
            # Check TTL
            if time.time() - cached_time < self.config.cache_ttl_seconds:
                self.cache_hits += 1
                return signal
            else:
                # Expired, remove
                del self.signal_cache[cache_key]
        
        self.cache_misses += 1
        return None
    
    def _cache_signal(self, cache_key: str, signal: Any):
        """Cache signal with timestamp."""
        if not self.config.enable_caching:
            return
        
        # Limit cache size
        if len(self.signal_cache) > 1000:
            # Remove oldest 25%
            items = sorted(self.signal_cache.items(), key=lambda x: x[1][0])
            for key, _ in items[:250]:
                del self.signal_cache[key]
        
        self.signal_cache[cache_key] = (time.time(), signal)
    
    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def _estimate_accuracy(self, signal: Any) -> float:
        """Estimate signal accuracy based on confidence and historical performance."""
        if not hasattr(signal, 'confidence'):
            return 0.5
        
        confidence = getattr(signal, 'confidence', 0.5)
        
        # Adjust based on historical performance
        if self.performance_history:
            avg_confidence = np.mean([
                m.confidence for m in self.performance_history[-20:]
            ])
            accuracy_adjustment = min(avg_confidence / confidence, 1.2)
        else:
            accuracy_adjustment = 1.0
        
        return min(confidence * accuracy_adjustment, 1.0)
    
    def _trigger_optimization(self):
        """Trigger auto-optimization when performance degrades."""
        logger.warning("Signal generation latency exceeded threshold, triggering optimization")
        
        # Reduce feature count
        if self.config.top_k_features > 20:
            self.config.top_k_features = max(20, self.config.top_k_features - 10)
            logger.info(f"Reduced feature count to {self.config.top_k_features}")
        
        # Clear some cache
        if len(self.signal_cache) > 500:
            items = sorted(self.signal_cache.items(), key=lambda x: x[1][0])
            for key, _ in items[:200]:
                del self.signal_cache[key]
            logger.info("Cleared 200 old cache entries")
    
    def update_feature_importance(self, symbol: str, importance: Dict[str, float]):
        """Update feature importance for a symbol."""
        if symbol not in self.feature_importance:
            self.feature_importance[symbol] = {}
        
        # Exponential moving average update
        alpha = 0.1  # Learning rate
        for feature, score in importance.items():
            if feature in self.feature_importance[symbol]:
                self.feature_importance[symbol][feature] = (
                    alpha * score + 
                    (1 - alpha) * self.feature_importance[symbol][feature]
                )
            else:
                self.feature_importance[symbol][feature] = score
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-100:]  # Last 100 generations
        
        return {
            'total_generations': self.total_generations,
            'cache_hit_rate': self._get_cache_hit_rate(),
            'avg_generation_time_ms': np.mean([
                m.generation_time_ms for m in recent_metrics
            ]),
            'p95_generation_time_ms': np.percentile([
                m.generation_time_ms for m in recent_metrics
            ], 95),
            'avg_confidence': np.mean([
                m.confidence for m in recent_metrics
            ]),
            'avg_accuracy_estimate': np.mean([
                m.accuracy_estimate for m in recent_metrics
            ]),
            'cache_size': len(self.signal_cache),
            'feature_importance_size': len(self.feature_importance),
            'latency_target_met_pct': np.mean([
                1.0 if m.generation_time_ms <= self.config.latency_threshold_ms else 0.0
                for m in recent_metrics
            ]) * 100
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.signal_cache.clear()
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("All caches cleared")
    
    def shutdown(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("SignalOptimizer shutdown complete")


# Decorator for automatic signal optimization
def optimize_signal(config: OptimizationConfig = None):
    """
    Decorator to automatically optimize signal generation functions.
    
    Args:
        config: Optimization configuration
    """
    if config is None:
        config = OptimizationConfig()
    
    optimizer = SignalOptimizer(config)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract data and symbol from arguments
            data = kwargs.get('data') or args[0] if args else None
            symbol = kwargs.get('symbol') or args[1] if len(args) > 1 else 'UNKNOWN'
            
            if data is None or not isinstance(data, pd.DataFrame):
                return func(*args, **kwargs)
            
            # Use optimizer
            signal, metrics = optimizer.optimize_signal_generation(
                data, symbol, func, *args, **kwargs
            )
            
            # Store metrics for monitoring
            wrapper._last_metrics = metrics
            
            return signal
        
        # Attach optimizer for access
        wrapper._optimizer = optimizer
        wrapper._last_metrics = None
        
        return wrapper
    
    return decorator


# Global optimizer instance
_global_optimizer: Optional[SignalOptimizer] = None


def get_global_optimizer() -> SignalOptimizer:
    """Get global signal optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = SignalOptimizer(OptimizationConfig())
    return _global_optimizer


def optimize_global_signal(data: pd.DataFrame, symbol: str, signal_generator, **kwargs):
    """Optimize signal using global optimizer."""
    optimizer = get_global_optimizer()
    return optimizer.optimize_signal_generation(data, symbol, signal_generator, **kwargs)
