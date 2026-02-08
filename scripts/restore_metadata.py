
import json
import pandas as pd
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta

from config import ApexConfig

def restore_metadata():
    metadata_path = ApexConfig.DATA_DIR / "position_metadata.json"
    if not metadata_path.exists():
        return
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Get standard symbols from metadata
    symbols = list(metadata.keys())
    print(f"Restoring {len(symbols)} symbols...")
    
    # Fetch yesterday's close (Feb 5, 2026)
    # We use a 3-day window to be safe and get the last trading day's close
    data = yf.download(symbols, start="2026-02-04", end="2026-02-06", interval="1d", group_by="ticker")
    
    for symbol in symbols:
        try:
            # Get the last close before today (Feb 6)
            if symbol in data:
                close_prices = data[symbol]['Close'].dropna()
                if not close_prices.empty:
                    # Use the last known close before today
                    yesterday_close = float(close_prices.iloc[0]) # Since Feb 6 is 'end', iloc[0] should be Feb 5
                    metadata[symbol]['entry_price'] = yesterday_close
                    print(f"  {symbol}: Restored to ${yesterday_close:.2f}")
        except Exception as e:
            print(f"  {symbol}: Failed to restore: {e}")
            
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("DONE")

if __name__ == "__main__":
    restore_metadata()
