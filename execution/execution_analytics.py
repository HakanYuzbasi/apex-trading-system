"""
Execution Analytics

Tracks execution quality metrics including slippage, fill rates, and algo performance.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionAnalytics:
    """Track and analyze execution quality."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.executions_file = data_dir / "execution_analytics.jsonl"
        
        # In-memory tracking
        self.executions: List[Dict] = []
        self.slippage_by_algo: Dict[str, List[float]] = defaultdict(list)
        self.fills_by_algo: Dict[str, int] = defaultdict(int)
        self.orders_by_algo: Dict[str, int] = defaultdict(int)
        
        self._load_recent_executions()
    
    def _load_recent_executions(self, days: int = 30):
        """Load recent executions from file."""
        if not self.executions_file.exists():
            return
        
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            with open(self.executions_file, 'r') as f:
                for line in f:
                    try:
                        exec_data = json.loads(line)
                        exec_time = datetime.fromisoformat(exec_data['timestamp'])
                        if exec_time > cutoff:
                            self.executions.append(exec_data)
                            
                            # Update in-memory stats
                            algo = exec_data.get('algo_type', 'MARKET')
                            if exec_data.get('slippage_bps') is not None:
                                self.slippage_by_algo[algo].append(exec_data['slippage_bps'])
                            if exec_data.get('filled', False):
                                self.fills_by_algo[algo] += 1
                            self.orders_by_algo[algo] += 1
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Error loading executions: {e}")
    
    def track_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        arrival_price: float,
        fill_price: float,
        algo_type: str = "MARKET",
        filled: bool = True,
        execution_time_ms: Optional[float] = None
    ):
        """Track an execution."""
        # Calculate slippage in basis points
        if arrival_price > 0:
            slippage_bps = (fill_price - arrival_price) / arrival_price * 10000
            if side == "SELL":
                slippage_bps = -slippage_bps  # Reverse for sells
        else:
            slippage_bps = 0.0
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "arrival_price": float(arrival_price),
            "fill_price": float(fill_price),
            "slippage_bps": float(slippage_bps),
            "algo_type": algo_type,
            "filled": filled,
            "execution_time_ms": execution_time_ms
        }
        
        self.executions.append(record)
        
        # Update in-memory stats
        if slippage_bps is not None:
            self.slippage_by_algo[algo_type].append(slippage_bps)
        if filled:
            self.fills_by_algo[algo_type] += 1
        self.orders_by_algo[algo_type] += 1
        
        # Append to file
        try:
            with open(self.executions_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error writing execution: {e}")
        
        # Log if slippage is excessive
        if abs(slippage_bps) > 50:  # 50 bps = 0.5%
            logger.warning(
                f"⚠️ High slippage on {symbol}: {slippage_bps:.1f} bps "
                f"({arrival_price:.2f} → {fill_price:.2f})"
            )
    
    def get_algo_performance(self, algo_type: str, days: int = 7) -> Dict:
        """Get performance metrics for a specific algo."""
        cutoff = datetime.now() - timedelta(days=days)
        
        slippages = []
        fills = 0
        orders = 0
        exec_times = []
        
        for exec_data in self.executions:
            exec_time = datetime.fromisoformat(exec_data['timestamp'])
            if exec_time < cutoff:
                continue
            
            if exec_data['algo_type'] != algo_type:
                continue
            
            orders += 1
            if exec_data.get('filled', False):
                fills += 1
            
            if exec_data.get('slippage_bps') is not None:
                slippages.append(exec_data['slippage_bps'])
            
            if exec_data.get('execution_time_ms') is not None:
                exec_times.append(exec_data['execution_time_ms'])
        
        if not slippages:
            return {
                "algo_type": algo_type,
                "days": days,
                "total_orders": orders,
                "fills": fills,
                "fill_rate": fills / orders if orders > 0 else 0,
                "avg_slippage_bps": 0,
                "median_slippage_bps": 0,
                "slippage_std_bps": 0,
                "avg_execution_time_ms": 0
            }
        
        return {
            "algo_type": algo_type,
            "days": days,
            "total_orders": orders,
            "fills": fills,
            "fill_rate": fills / orders if orders > 0 else 0,
            "avg_slippage_bps": float(np.mean(slippages)),
            "median_slippage_bps": float(np.median(slippages)),
            "slippage_std_bps": float(np.std(slippages)),
            "avg_execution_time_ms": float(np.mean(exec_times)) if exec_times else 0
        }
    
    def get_summary(self, days: int = 7) -> Dict:
        """Get overall execution summary."""
        algos = set(exec_data['algo_type'] for exec_data in self.executions)
        
        algo_stats = {}
        for algo in algos:
            algo_stats[algo] = self.get_algo_performance(algo, days)
        
        # Overall stats
        cutoff = datetime.now() - timedelta(days=days)
        recent_execs = [
            e for e in self.executions
            if datetime.fromisoformat(e['timestamp']) > cutoff
        ]
        
        all_slippages = [
            e['slippage_bps'] for e in recent_execs
            if e.get('slippage_bps') is not None
        ]
        
        return {
            "days": days,
            "total_executions": len(recent_execs),
            "by_algo": algo_stats,
            "overall_avg_slippage_bps": float(np.mean(all_slippages)) if all_slippages else 0,
            "overall_median_slippage_bps": float(np.median(all_slippages)) if all_slippages else 0
        }


# Global instance
_analytics: Optional[ExecutionAnalytics] = None


def get_execution_analytics(data_dir: Path) -> ExecutionAnalytics:
    """Get or create global execution analytics."""
    global _analytics
    if _analytics is None:
        _analytics = ExecutionAnalytics(data_dir)
    return _analytics
