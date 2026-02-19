"""
Model Performance Tracker

Tracks live model predictions vs actual outcomes to detect model degradation.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """Track model predictions vs actual outcomes."""
    
    def __init__(self, data_dir: Path, window_days: int = 30):
        self.data_dir = data_dir
        self.window_days = window_days
        self.predictions_file = data_dir / "model_predictions.jsonl"
        
        # In-memory cache for fast lookups
        self.predictions: deque = deque(maxlen=10000)
        self._load_recent_predictions()
        
    def _load_recent_predictions(self):
        """Load recent predictions from file."""
        if not self.predictions_file.exists():
            return
            
        cutoff = datetime.now() - timedelta(days=self.window_days)
        
        try:
            with open(self.predictions_file, 'r') as f:
                for line in f:
                    try:
                        pred = json.loads(line)
                        pred_time = datetime.fromisoformat(pred['timestamp'])
                        if pred_time > cutoff:
                            self.predictions.append(pred)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
    
    def log_prediction(
        self,
        symbol: str,
        prediction: float,
        regime: str,
        model_version: str = "current"
    ):
        """Log a model prediction."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "prediction": float(prediction),
            "regime": regime,
            "model_version": model_version,
            "actual_return": None,  # To be filled later
            "outcome_timestamp": None
        }
        
        self.predictions.append(record)
        
        # Append to file
        try:
            with open(self.predictions_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error writing prediction: {e}")
    
    def update_outcome(self, symbol: str, actual_return: float, lookback_hours: int = 24):
        """Update prediction with actual outcome."""
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        
        # Find matching prediction
        for pred in reversed(self.predictions):
            if pred['symbol'] == symbol and pred['actual_return'] is None:
                pred_time = datetime.fromisoformat(pred['timestamp'])
                if pred_time > cutoff:
                    pred['actual_return'] = float(actual_return)
                    pred['outcome_timestamp'] = datetime.now().isoformat()
                    logger.debug(
                        f"Updated outcome for {symbol}: "
                        f"predicted={pred['prediction']:.4f}, actual={actual_return:.4f}"
                    )
                    break
    
    def get_accuracy(
        self,
        days: int = 7,
        regime: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> Dict:
        """Calculate directional accuracy over last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        correct = 0
        total = 0
        predictions_list = []
        actuals_list = []
        
        for pred in self.predictions:
            # Filter by time
            pred_time = datetime.fromisoformat(pred['timestamp'])
            if pred_time < cutoff:
                continue
            
            # Filter by regime
            if regime and pred['regime'] != regime:
                continue
            
            # Filter by model version
            if model_version and pred['model_version'] != model_version:
                continue
            
            # Skip if no outcome yet
            if pred['actual_return'] is None:
                continue
            
            # Check directional accuracy
            pred_direction = pred['prediction'] > 0
            actual_direction = pred['actual_return'] > 0
            
            if pred_direction == actual_direction:
                correct += 1
            total += 1
            
            predictions_list.append(pred['prediction'])
            actuals_list.append(pred['actual_return'])
        
        if total == 0:
            return {
                "accuracy": 0.0,
                "total_predictions": 0,
                "correct_predictions": 0,
                "mse": 0.0,
                "mae": 0.0
            }
        
        # Calculate metrics
        predictions_arr = np.array(predictions_list)
        actuals_arr = np.array(actuals_list)
        
        mse = np.mean((predictions_arr - actuals_arr) ** 2)
        mae = np.mean(np.abs(predictions_arr - actuals_arr))
        
        return {
            "accuracy": correct / total,
            "total_predictions": total,
            "correct_predictions": correct,
            "mse": float(mse),
            "mae": float(mae),
            "days": days,
            "regime": regime,
            "model_version": model_version
        }
    
    def get_regime_breakdown(self, days: int = 7) -> Dict[str, Dict]:
        """Get accuracy breakdown by regime."""
        regimes = set(pred['regime'] for pred in self.predictions if pred.get('regime'))
        
        breakdown = {}
        for regime in regimes:
            breakdown[regime] = self.get_accuracy(days=days, regime=regime)
        
        return breakdown
    
    def check_degradation(
        self,
        baseline_accuracy: float = 0.54,
        threshold_drop: float = 0.05,
        days: int = 7
    ) -> bool:
        """Check if model has degraded significantly."""
        current = self.get_accuracy(days=days)
        
        if current['total_predictions'] < 20:
            # Not enough data
            return False
        
        accuracy_drop = baseline_accuracy - current['accuracy']
        
        if accuracy_drop > threshold_drop:
            logger.warning(
                f"⚠️ Model degradation detected! "
                f"Accuracy dropped from {baseline_accuracy:.2%} to {current['accuracy']:.2%} "
                f"({accuracy_drop:.2%} drop over {days} days)"
            )
            return True
        
        return False
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        return {
            "7_day": self.get_accuracy(days=7),
            "30_day": self.get_accuracy(days=30),
            "by_regime": self.get_regime_breakdown(days=7),
            "total_tracked": len(self.predictions),
            "degradation_detected": self.check_degradation()
        }


# Global instance
_tracker: Optional[ModelPerformanceTracker] = None


def get_tracker(data_dir: Path) -> ModelPerformanceTracker:
    """Get or create global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ModelPerformanceTracker(data_dir)
    return _tracker
