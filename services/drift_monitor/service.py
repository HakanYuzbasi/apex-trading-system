"""
Drift Monitor Service - wraps ModelDriftMonitor and FeatureDriftDetector for API use.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from config import ApexConfig
from monitoring.model_drift_monitor import ModelDriftMonitor
from monitoring.feature_drift_detector import FeatureDriftDetector

logger = logging.getLogger(__name__)


class DriftMonitorService:
    """Thin wrapper that exposes drift monitoring data to the API layer."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir or ApexConfig.DATA_DIR
        self._model_monitor = ModelDriftMonitor(data_dir=self._data_dir)
        self._feature_detector = FeatureDriftDetector(
            baseline_stats_file=self._data_dir / "feature_baseline_stats.json",
        )

    # -- Model drift --------------------------------------------------------

    def get_status(self) -> dict:
        """Return current model health status as a plain dict."""
        status = self._model_monitor.get_status()
        return status.to_dict()

    def get_report(self) -> dict:
        """Return detailed drift report including window history."""
        return self._model_monitor.get_report()

    # -- Feature drift ------------------------------------------------------

    def get_feature_drift_summary(self, days: int = 7) -> dict:
        """Return summary of recent feature drift events."""
        return self._feature_detector.get_drift_summary(days=days)
