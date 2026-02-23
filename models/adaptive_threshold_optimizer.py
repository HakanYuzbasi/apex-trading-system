"""
models/adaptive_threshold_optimizer.py - Per-Symbol Threshold Optimization

Walk-forward optimization of entry/exit thresholds per symbol using
historical signal outcomes. Replaces static regime-based thresholds
with data-driven, symbol-specific parameters.

Key features:
- Walk-forward grid search (no look-ahead bias)
- Per-symbol entry thresholds, confidence thresholds, sizing multipliers
- Feature importance analysis per symbol
- Graceful fallback to regime defaults when insufficient data
- Symbol volatility regime classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Grid search parameter space
ENTRY_THRESHOLD_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
CONFIDENCE_THRESHOLD_GRID = [0.20, 0.30, 0.40, 0.50]


@dataclass
class SymbolThresholds:
    """Optimized thresholds for a specific symbol."""
    symbol: str
    entry_threshold: float
    high_conviction_threshold: float
    confidence_threshold: float
    position_size_multiplier: float
    volatility_regime: str
    top_features: List[str]
    expected_sharpe: float
    last_optimized: datetime
    sample_count: int
    is_default: bool = False


@dataclass
class OptimizationResult:
    """Result of threshold optimization for a symbol."""
    symbol: str
    thresholds: SymbolThresholds
    walk_forward_sharpe: float
    walk_forward_accuracy: float
    parameter_stability: float
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveThresholdOptimizer:
    """
    Per-symbol walk-forward threshold optimization.

    Uses historical signal outcomes to find optimal entry/exit thresholds
    for each symbol. Falls back to regime defaults when insufficient data.
    """

    def __init__(
        self,
        outcome_tracker: Any = None,
        default_thresholds: Optional[Dict[str, float]] = None,
        min_signals: int = 30,
        optimization_interval_hours: int = 24,
    ):
        self.outcome_tracker = outcome_tracker
        self.default_thresholds = default_thresholds or {
            "bull": 0.23,
            "bear": 0.25,
            "neutral": 0.28,
            "volatile": 0.32,
            "strong_bull": 0.20,
            "strong_bear": 0.23,
            "high_volatility": 0.32,
        }
        self.min_signals = min_signals
        self.optimization_interval_hours = optimization_interval_hours

        # Cached optimized thresholds per symbol
        self._optimized: Dict[str, SymbolThresholds] = {}
        self._last_optimization: Dict[str, datetime] = {}

        logger.info(
            f"AdaptiveThresholdOptimizer initialized: "
            f"min_signals={min_signals}, interval={optimization_interval_hours}h"
        )

    def get_thresholds(self, symbol: str, regime: str) -> SymbolThresholds:
        """
        Get thresholds for a symbol.

        Returns optimized thresholds if available and fresh,
        otherwise returns regime defaults.

        Args:
            symbol: Stock symbol
            regime: Current market regime

        Returns:
            SymbolThresholds (optimized or default)
        """
        # Check if we have fresh optimized thresholds
        if symbol in self._optimized:
            last_opt = self._last_optimization.get(symbol, datetime.min)
            age_hours = (datetime.now() - last_opt).total_seconds() / 3600

            if age_hours < self.optimization_interval_hours:
                return self._optimized[symbol]

        # Fall back to regime defaults
        return self._default_thresholds(symbol, regime)

    def optimize_symbol(
        self,
        symbol: str,
        signal_history: pd.DataFrame,
    ) -> Optional[OptimizationResult]:
        """
        Optimize thresholds for a single symbol using walk-forward analysis.

        Args:
            symbol: Stock symbol
            signal_history: DataFrame with columns:
                signal, confidence, return_5d, regime, timestamp

        Returns:
            OptimizationResult or None if insufficient data
        """
        # Filter to this symbol
        if "symbol" in signal_history.columns:
            sym_data = signal_history[signal_history["symbol"] == symbol].copy()
        else:
            sym_data = signal_history.copy()

        if len(sym_data) < self.min_signals:
            return None

        # Ensure required columns
        required = {"signal", "confidence", "return_5d"}
        if not required.issubset(set(sym_data.columns)):
            missing = required - set(sym_data.columns)
            logger.debug(f"Missing columns for {symbol}: {missing}")
            return None

        # Drop rows with missing returns
        sym_data = sym_data.dropna(subset=["signal", "confidence", "return_5d"])
        if len(sym_data) < self.min_signals:
            return None

        # Run walk-forward optimization
        best_entry, best_conf, wf_sharpe, wf_accuracy, stability = (
            self._walk_forward_optimize(sym_data)
        )

        # Compute symbol volatility regime
        vol_regime = self._compute_symbol_vol_regime(sym_data)

        # Compute position size multiplier
        size_mult = self._adjust_position_size(vol_regime, wf_sharpe)

        # Get top features if available
        top_features = self._get_top_features(sym_data)

        thresholds = SymbolThresholds(
            symbol=symbol,
            entry_threshold=best_entry,
            high_conviction_threshold=max(best_entry + 0.15, 0.50),
            confidence_threshold=best_conf,
            position_size_multiplier=size_mult,
            volatility_regime=vol_regime,
            top_features=top_features,
            expected_sharpe=wf_sharpe,
            last_optimized=datetime.now(),
            sample_count=len(sym_data),
        )

        # Cache
        self._optimized[symbol] = thresholds
        self._last_optimization[symbol] = datetime.now()

        return OptimizationResult(
            symbol=symbol,
            thresholds=thresholds,
            walk_forward_sharpe=wf_sharpe,
            walk_forward_accuracy=wf_accuracy,
            parameter_stability=stability,
        )

    def optimize_all(
        self, signal_history: pd.DataFrame
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize thresholds for all symbols with sufficient data.

        Args:
            signal_history: Complete signal history DataFrame

        Returns:
            Dict of symbol -> OptimizationResult
        """
        results = {}

        if "symbol" not in signal_history.columns:
            return results

        symbols = signal_history["symbol"].unique()

        for symbol in symbols:
            try:
                result = self.optimize_symbol(symbol, signal_history)
                if result is not None:
                    results[symbol] = result
            except Exception as e:
                logger.debug(f"Optimization failed for {symbol}: {e}")

        if results:
            logger.info(
                f"Optimized thresholds for {len(results)} symbols "
                f"(avg Sharpe: {np.mean([r.walk_forward_sharpe for r in results.values()]):.2f})"
            )

        return results

    # ─── Internal Methods ──────────────────────────────────────────

    def _walk_forward_optimize(
        self,
        data: pd.DataFrame,
        train_window: int = 60,
        test_window: int = 20,
    ) -> Tuple[float, float, float, float, float]:
        """
        Walk-forward grid search over entry/confidence thresholds.

        Returns:
            (best_entry_threshold, best_conf_threshold,
             avg_test_sharpe, avg_test_accuracy, parameter_stability)
        """
        n = len(data)
        if n < train_window + test_window:
            # Insufficient data for walk-forward
            return 0.25, 0.25, 0.0, 0.50, 0.0

        # Track best parameters per fold
        fold_results: List[Dict] = []

        step = max(test_window, 10)
        start = 0

        while start + train_window + test_window <= n:
            train_end = start + train_window
            test_end = min(train_end + test_window, n)

            train = data.iloc[start:train_end]
            test = data.iloc[train_end:test_end]

            if len(test) < 5:
                break

            # Grid search on training set
            best_sharpe = -999
            best_params = (0.25, 0.25)

            for entry_thresh in ENTRY_THRESHOLD_GRID:
                for conf_thresh in CONFIDENCE_THRESHOLD_GRID:
                    sharpe = self._evaluate_threshold(
                        train, entry_thresh, conf_thresh
                    )
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = (entry_thresh, conf_thresh)

            # Evaluate best params on test set
            test_sharpe = self._evaluate_threshold(
                test, best_params[0], best_params[1]
            )
            test_accuracy = self._evaluate_accuracy(
                test, best_params[0], best_params[1]
            )

            fold_results.append({
                "entry_threshold": best_params[0],
                "confidence_threshold": best_params[1],
                "test_sharpe": test_sharpe,
                "test_accuracy": test_accuracy,
            })

            start += step

        if not fold_results:
            return 0.25, 0.25, 0.0, 0.50, 0.0

        # Average best parameters across folds
        avg_entry = float(np.mean([f["entry_threshold"] for f in fold_results]))
        avg_conf = float(np.mean([f["confidence_threshold"] for f in fold_results]))
        avg_sharpe = float(np.mean([f["test_sharpe"] for f in fold_results]))
        avg_accuracy = float(np.mean([f["test_accuracy"] for f in fold_results]))

        # Parameter stability (low std = stable)
        entry_std = float(np.std([f["entry_threshold"] for f in fold_results]))
        stability = max(0.0, 1.0 - entry_std / 0.15)

        # Snap to nearest grid value
        avg_entry = min(ENTRY_THRESHOLD_GRID, key=lambda x: abs(x - avg_entry))
        avg_conf = min(CONFIDENCE_THRESHOLD_GRID, key=lambda x: abs(x - avg_conf))

        return avg_entry, avg_conf, avg_sharpe, avg_accuracy, stability

    def _evaluate_threshold(
        self,
        data: pd.DataFrame,
        entry_threshold: float,
        confidence_threshold: float,
    ) -> float:
        """Evaluate a threshold combination by computing simulated Sharpe."""
        # Filter signals by thresholds
        mask = (
            (data["signal"].abs() >= entry_threshold) &
            (data["confidence"] >= confidence_threshold)
        )
        filtered = data[mask]

        if len(filtered) < 3:
            return -1.0

        # Simulated returns: sign(signal) * return_5d
        sim_returns = np.sign(filtered["signal"].values) * filtered["return_5d"].values

        if sim_returns.std() < 1e-10:
            return 0.0

        return float(sim_returns.mean() / sim_returns.std() * np.sqrt(52))  # weekly

    def _evaluate_accuracy(
        self,
        data: pd.DataFrame,
        entry_threshold: float,
        confidence_threshold: float,
    ) -> float:
        """Evaluate directional accuracy for a threshold combination."""
        mask = (
            (data["signal"].abs() >= entry_threshold) &
            (data["confidence"] >= confidence_threshold)
        )
        filtered = data[mask]

        if len(filtered) < 3:
            return 0.50

        correct = (
            np.sign(filtered["signal"].values) ==
            np.sign(filtered["return_5d"].values)
        )
        return float(correct.mean())

    def _compute_symbol_vol_regime(self, data: pd.DataFrame) -> str:
        """Classify symbol's volatility regime from its return distribution."""
        if "return_5d" not in data.columns or len(data) < 10:
            return "normal_vol"

        returns = data["return_5d"].dropna()
        if len(returns) < 10:
            return "normal_vol"

        vol = float(returns.std() * np.sqrt(52))  # Annualized from weekly

        if vol > 0.50:
            return "high_vol"
        elif vol < 0.15:
            return "low_vol"
        return "normal_vol"

    def _adjust_position_size(
        self, vol_regime: str, expected_sharpe: float
    ) -> float:
        """
        Compute position size multiplier based on vol regime and Sharpe.

        High vol + low Sharpe = smaller positions.
        Low vol + high Sharpe = larger positions.
        """
        if vol_regime == "high_vol" and expected_sharpe < 1.0:
            return 0.5
        elif vol_regime == "high_vol":
            return 0.7
        elif vol_regime == "low_vol" and expected_sharpe > 1.5:
            return 1.3
        elif vol_regime == "low_vol" and expected_sharpe > 1.0:
            return 1.15
        return 1.0

    def _get_top_features(self, data: pd.DataFrame) -> List[str]:
        """
        Get top features for this symbol if feature data is available.

        Falls back to generic list if not.
        """
        # If we have feature columns, compute correlations with return
        feature_cols = [
            c for c in data.columns
            if c not in ("symbol", "signal", "confidence", "return_5d",
                         "return_1d", "return_10d", "return_20d",
                         "regime", "timestamp", "entry_price")
        ]

        if not feature_cols or "return_5d" not in data.columns:
            return ["momentum", "trend", "mean_reversion", "volatility"]

        correlations = {}
        for col in feature_cols[:20]:  # Limit to first 20
            try:
                corr = data[col].corr(data["return_5d"])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except (ValueError, TypeError):
                continue

        if not correlations:
            return ["momentum", "trend", "mean_reversion", "volatility"]

        sorted_features = sorted(correlations, key=correlations.get, reverse=True)
        return sorted_features[:5]

    def _default_thresholds(self, symbol: str, regime: str) -> SymbolThresholds:
        """Create default thresholds from regime configuration."""
        entry = self.default_thresholds.get(regime, 0.28)
        return SymbolThresholds(
            symbol=symbol,
            entry_threshold=entry,
            high_conviction_threshold=max(entry + 0.15, 0.50),
            confidence_threshold=0.25,
            position_size_multiplier=1.0,
            volatility_regime="normal_vol",
            top_features=["momentum", "trend", "mean_reversion", "volatility"],
            expected_sharpe=0.0,
            last_optimized=datetime.min,
            sample_count=0,
            is_default=True,
        )

    def get_diagnostics(self) -> Dict:
        """Return optimizer state for monitoring."""
        return {
            "optimized_symbols": len(self._optimized),
            "symbol_details": {
                sym: {
                    "entry_threshold": t.entry_threshold,
                    "confidence_threshold": t.confidence_threshold,
                    "expected_sharpe": t.expected_sharpe,
                    "size_multiplier": t.position_size_multiplier,
                    "sample_count": t.sample_count,
                    "vol_regime": t.volatility_regime,
                }
                for sym, t in self._optimized.items()
            },
        }
