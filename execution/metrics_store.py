"""
execution/metrics_store.py
Persistent storage for execution metrics (slippage, commissions).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ExecutionMetricsStore:
    """Persistent store for execution metrics."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.metrics = {
            'total_trades': 0,
            'total_slippage': 0.0,
            'total_commission': 0.0,
            'slippage_history': []  # List of dicts
        }
        self._load()

    def _load(self):
        """Load metrics from file."""
        if not self.file_path.exists():
            return
        
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                # Merge safely
                for key in self.metrics:
                    if key in data:
                        self.metrics[key] = data[key]
            logger.info(f"Loaded execution metrics from {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load metrics from {self.file_path}: {e}")

    def save(self):
        """Save metrics to file."""
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.file_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics to {self.file_path}: {e}")

    def record_metrics(self, slippage_bps: float, commission: float, trade_details: Dict):
        """Record execution metrics for a trade and persist."""
        self.metrics['total_trades'] += 1
        self.metrics['total_slippage'] += abs(float(slippage_bps))
        self.metrics['total_commission'] += float(commission)
        
        # Add timestamp if missing
        if 'timestamp' not in trade_details:
             trade_details['timestamp'] = datetime.now().isoformat()

        self.metrics['slippage_history'].append(trade_details)
        
        # Keep history manageable (last 1000 trades)
        if len(self.metrics['slippage_history']) > 1000:
             self.metrics['slippage_history'] = self.metrics['slippage_history'][-1000:]
             
        self.save()

    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return self.metrics
    
    def get_avg_slippage(self) -> float:
        """Get average slippage per trade."""
        if self.metrics['total_trades'] > 0:
            return self.metrics['total_slippage'] / self.metrics['total_trades']
        return 0.0
