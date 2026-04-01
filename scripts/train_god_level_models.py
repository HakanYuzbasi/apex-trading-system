#!/usr/bin/env python3
"""
God Level Model Continuous Retraining Script
=============================================
Fetches daily OHLCV data and retrains the advanced ML ensemble,
including the Synthetic Order Flow and Parkinson features.
Designed to be run as an automated scheduled background daemon.
The daemon cadence is controlled externally via APEX_RETRAIN_INTERVAL_SECONDS.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import yfinance as yf

from models.god_level_signal_generator import GodLevelSignalGenerator
from core.logging_config import setup_logging

logger = setup_logging(level="INFO", log_file="logs/god_level_training.log", 
                       json_format=False, console_output=True)

DEFAULT_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY', 'QQQ',
    'BTC-USD', 'ETH-USD'
]

def fetch_historical_data(symbols: List[str], days: int = 1500) -> Dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data for training using yfinance."""
    historical_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for symbol in symbols:
        try:
            logger.info(f"  Fetching {symbol} for training...")
            
            # Convert crypto symbols for yfinance
            yf_symbol = symbol.replace('/', '-') if '/' in symbol else symbol
            
            # Fetch data
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if df is not None and len(df) > 100:
                df = df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
                historical_data[symbol] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                logger.info(f"    ✅ {symbol}: {len(df)} bars")
            else:
                logger.warning(f"    ⚠️  {symbol}: Insufficient data")
                
        except Exception as e:
            logger.error(f"    ❌ {symbol}: {e}")
            
    logger.info(f"Fetched data for {len(historical_data)}/{len(symbols)} symbols")
    return historical_data

def main():
    parser = argparse.ArgumentParser(description="Train God Level models")
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--days', type=int, default=1500, help='Days of historical data')
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')] if args.symbols else DEFAULT_SYMBOLS
    
    logger.info("="*80)
    logger.info("GOD LEVEL AUTOMATED RETRAINING CYCLE STARTED")
    logger.info("="*80)
    
    historical_data = fetch_historical_data(symbols, args.days)
    
    if len(historical_data) < 2:
        logger.error("Insufficient data to train God Level models.")
        return 1
        
    generator = GodLevelSignalGenerator()
    generator.train_models(historical_data)
    
    logger.info("✅ GOD LEVEL RETRAINING CYCLE COMPLETE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
