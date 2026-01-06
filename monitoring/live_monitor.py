"""
monitoring/live_monitor.py - Export trading state for dashboard
Run this alongside main.py to feed data to Streamlit
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class LiveMonitor:
    """Export live trading state for dashboard."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.state_file = self.data_dir / "trading_state.json"
        self.trades_file = self.data_dir / "trades.csv"
        self.equity_file = self.data_dir / "equity_curve.csv"
        
        # Initialize CSV files
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        if not self.trades_file.exists():
            with open(self.trades_file, 'w') as f:
                f.write("timestamp,symbol,side,quantity,price,pnl\n")
        
        if not self.equity_file.exists():
            with open(self.equity_file, 'w') as f:
                f.write("timestamp,equity,drawdown\n")
    
    def update_state(self, state: Dict):
        """Update current state."""
        try:
            state['timestamp'] = datetime.now().isoformat()
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug("State updated for dashboard")
        except Exception as e:
            logger.error(f"Failed to update state: {e}")
    
    def log_trade(self, symbol: str, side: str, quantity: int, price: float, pnl: float = 0):
        """Log a trade."""
        try:
            timestamp = datetime.now().isoformat()
            line = f"{timestamp},{symbol},{side},{quantity},{price},{pnl}\n"
            
            with open(self.trades_file, 'a') as f:
                f.write(line)
            
            logger.info(f"Trade logged: {side} {quantity} {symbol} @ ${price}")
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    def log_equity(self, equity: float, drawdown: float):
        """Log equity point."""
        try:
            timestamp = datetime.now().isoformat()
            line = f"{timestamp},{equity},{drawdown}\n"
            
            with open(self.equity_file, 'a') as f:
                f.write(line)
        except Exception as e:
            logger.error(f"Failed to log equity: {e}")
