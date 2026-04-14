import os
import re
from pathlib import Path

# --- 1. Relax the Data Watchdog in main.py ---
main_path = "main.py"
if os.path.exists(main_path):
    with open(main_path, "r", encoding="utf-8") as f:
        main_content = f.read()
    
    # Push the watchdog timeout to 15 minutes (900 seconds) for active positions
    main_content = re.sub(r'active_symbol_timeout=\d+', 'active_symbol_timeout=900', main_content)
    
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(main_content)
    print("✅ Relaxed the Data Watchdog to prevent false halts on slow Crypto feeds.")

# --- 2. Tune config.py for Maximum Alpha ---
config_path = "config.py"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Lower the entry thresholds so the AI is allowed to trade
    content = re.sub(r"'bull':\s*0\.20", "'bull': 0.12", content)
    content = re.sub(r"'bear':\s*0\.20", "'bear': 0.12", content)
    content = re.sub(r"'neutral':\s*0\.20", "'neutral': 0.15", content)
    content = re.sub(r"'volatile':\s*0\.25", "'volatile': 0.18", content)
    content = re.sub(r"'strong_bull':\s*0\.18", "'strong_bull': 0.10", content)
    content = re.sub(r"'strong_bear':\s*0\.18", "'strong_bear': 0.10", content)

    # Lower the confidence requirement
    content = re.sub(r'MIN_CONFIDENCE\s*=\s*0\.\d+', 'MIN_CONFIDENCE = 0.55', content)
    
    # Lower the Consensus Engine strictness so it agrees faster
    content = re.sub(r'MIN_CONSENSUS_AGREEMENT\s*=\s*0\.\d+', 'MIN_CONSENSUS_AGREEMENT = 0.60', content)
    
    # Increase Position Sizing to 10% of portfolio ($10,000 per trade on a $100k account)
    content = re.sub(r'POSITION_SIZE_USD\s*=\s*\d+', 'POSITION_SIZE_USD = 10000', content)
    
    # Increase the maximum positions allowed
    content = re.sub(r'MAX_POSITIONS\s*=\s*\d+', 'MAX_POSITIONS = 10', content)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Applied Aggressive Tuning to config.py. The engine is unleashed.")

# --- 3. Wipe the local hallucination state ---
target_files = [
    Path("data/trading_state.json"),
    Path("data/position_metadata.json"),
    Path("data/users/admin/trading_state.json"),
    Path("data/users/admin/position_metadata.json")
]

import json
for file_path in target_files:
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            modified = False
            if "positions" in data:
                data["positions"] = {}
                data["open_positions"] = 0
                modified = True
            elif isinstance(data, dict) and "positions" not in data:
                data = {} 
                modified = True
            if modified:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass
print("✅ Scrubbed phantom positions.")

