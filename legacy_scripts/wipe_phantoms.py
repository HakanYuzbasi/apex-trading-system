import json
from pathlib import Path

# Look in both global and multi-tenant admin folders
target_files = [
    Path("data/trading_state.json"),
    Path("data/position_metadata.json"),
    Path("data/users/admin/trading_state.json"),
    Path("data/users/admin/position_metadata.json")
]

for file_path in target_files:
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            modified = False
            # If it's the trading state file
            if "positions" in data:
                data["positions"] = {}
                data["open_positions"] = 0
                modified = True
            
            # If it's the metadata file (which is just a dict of symbols)
            elif isinstance(data, dict) and "positions" not in data:
                data = {} # wipe all phantom entry prices
                modified = True
                
            if modified:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"🧹 Scrubbed phantom positions from {file_path}")
        except Exception as e:
            print(f"⚠️ Could not process {file_path}: {e}")

print("👻 Ghost positions eliminated! Safe to restart.")
