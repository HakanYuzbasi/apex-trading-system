#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
from datetime import datetime

def clean_history():
    history_file = Path("data/performance_history.json")
    backup_file = Path(f"data/performance_history.json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if not history_file.exists():
        print(f"❌ File {history_file} not found.")
        return

    print(f"📦 Backing up {history_file} to {backup_file}...")
    shutil.copy2(history_file, backup_file)

    with open(history_file, 'r') as f:
        data = json.load(f)

    curve = data.get('equity_curve', [])
    if not curve:
        print("ℹ️ Equity curve is empty.")
        return

    original_count = len(curve)
    
    # 🔎 Filter logic:
    # A value is an outlier if it's < 50% of the LAST good value (or the peak)
    # Since we know the outlier is ~100k and good values are > 1M, we use a simple floor.
    CLEAN_FLOOR = 500000 
    
    cleaned_curve = [point for point in curve if point[1] >= CLEAN_FLOOR]
    removed_count = original_count - len(cleaned_curve)

    if removed_count > 0:
        data['equity_curve'] = cleaned_curve
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Successfully removed {removed_count} outliers (values < ${CLEAN_FLOOR:,}).")
        
        # Verify min value now
        values = [v for t, v in cleaned_curve]
        if values:
            print(f"📊 New Min: ${min(values):,.2f}")
            print(f"📊 New Max: ${max(values):,.2f}")
    else:
        print("ℹ️ No outliers found below the threshold.")

if __name__ == "__main__":
    clean_history()
