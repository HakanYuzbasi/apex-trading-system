"""
Feature Drift Detection

Monitors feature distributions and alerts when they drift from baseline.
"""

import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from monitoring.alert_aggregator import send_alert, AlertSeverity

logger = logging.getLogger(__name__)


class FeatureDriftDetector:
    """Detect when feature distributions drift from baseline."""
    
    def __init__(
        self,
        baseline_stats_file: Path,
        drift_threshold_sigma: float = 3.0
    ):
        self.baseline_stats_file = baseline_stats_file
        self.drift_threshold = drift_threshold_sigma
        
        self.baseline_stats: Dict = {}
        self.drift_history: List[Dict] = []
        
        self._load_baseline_stats()
    
    def _load_baseline_stats(self):
        """Load baseline feature statistics."""
        if not self.baseline_stats_file.exists():
            logger.warning(f"Baseline stats file not found: {self.baseline_stats_file}")
            return
        
        try:
            with open(self.baseline_stats_file, 'r') as f:
                self.baseline_stats = json.load(f)
            logger.info(f"Loaded baseline stats for {len(self.baseline_stats)} features")
        except Exception as e:
            logger.error(f"Error loading baseline stats: {e}")
    
    def save_baseline_stats(self, features: Dict[str, np.ndarray]):
        """Save current features as baseline."""
        stats = {}
        
        for feature_name, values in features.items():
            if len(values) == 0:
                continue
            
            stats[feature_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "count": int(len(values))
            }
        
        try:
            with open(self.baseline_stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.baseline_stats = stats
            logger.info(f"Saved baseline stats for {len(stats)} features")
        except Exception as e:
            logger.error(f"Error saving baseline stats: {e}")
    
    def detect_drift(
        self,
        current_features: Dict[str, float],
        regime: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Detect feature drift from baseline.
        
        Returns:
            Dictionary of drifted features with drift metrics
        """
        if not self.baseline_stats:
            logger.warning("No baseline stats available for drift detection")
            return {}
        
        drifted_features = {}
        
        for feature_name, current_value in current_features.items():
            if feature_name not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[feature_name]
            baseline_mean = baseline['mean']
            baseline_std = baseline['std']
            
            # Skip if std is too small (constant feature)
            if baseline_std < 1e-6:
                continue
            
            # Calculate z-score
            z_score = abs(current_value - baseline_mean) / baseline_std
            
            if z_score > self.drift_threshold:
                drift_info = {
                    "current_value": float(current_value),
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "z_score": float(z_score),
                    "threshold": self.drift_threshold,
                    "regime": regime
                }
                
                drifted_features[feature_name] = drift_info
                
                # Send alert
                send_alert(
                    alert_type="FEATURE_DRIFT",
                    message=f"Feature '{feature_name}' drifted: "
                           f"z-score={z_score:.2f} (current={current_value:.4f}, "
                           f"baseline={baseline_mean:.4f}±{baseline_std:.4f})",
                    severity=AlertSeverity.WARNING if z_score < 5 else AlertSeverity.ERROR,
                    metadata=drift_info
                )
                
                # Record drift event
                self.drift_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "feature": feature_name,
                    **drift_info
                })
        
        if drifted_features:
            logger.warning(
                f"⚠️ Detected drift in {len(drifted_features)} features: "
                f"{', '.join(drifted_features.keys())}"
            )
        
        return drifted_features
    
    def get_drift_summary(self, days: int = 7) -> Dict:
        """Get summary of recent drift events."""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_drifts = [
            d for d in self.drift_history
            if datetime.fromisoformat(d['timestamp']) > cutoff
        ]
        
        # Count by feature
        drift_counts = {}
        for drift in recent_drifts:
            feature = drift['feature']
            drift_counts[feature] = drift_counts.get(feature, 0) + 1
        
        return {
            "total_drift_events": len(recent_drifts),
            "unique_features_drifted": len(drift_counts),
            "drift_by_feature": drift_counts,
            "days": days,
            "threshold_sigma": self.drift_threshold
        }
    
    def should_retrain(self, max_drifted_features: int = 5) -> bool:
        """Check if model should be retrained based on drift."""
        recent_summary = self.get_drift_summary(days=1)
        
        if recent_summary['unique_features_drifted'] >= max_drifted_features:
            logger.warning(
                f"🔄 Recommending model retrain: "
                f"{recent_summary['unique_features_drifted']} features drifted "
                f"(threshold: {max_drifted_features})"
            )
            return True
        
        return False


# Global instance
_detector: Optional[FeatureDriftDetector] = None


def get_drift_detector(
    baseline_stats_file: Path,
    drift_threshold: float = 3.0
) -> FeatureDriftDetector:
    """Get or create global drift detector."""
    global _detector
    if _detector is None:
        _detector = FeatureDriftDetector(baseline_stats_file, drift_threshold)
    return _detector
