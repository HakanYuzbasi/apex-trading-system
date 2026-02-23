import os

filepath = "core/symbols.py"
if not os.path.exists(filepath):
    print("⚠️ Could not find core/symbols.py")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

patch = """
# --- FIX: MARKET HOURS OVERRIDE ---
import pytz
from datetime import datetime

def custom_is_market_open(symbol: str, timestamp=None) -> bool:
    if timestamp is None:
        timestamp = datetime.utcnow()
    try:
        parsed = parse_symbol(symbol)
        
        # Crypto is 24/7
        if parsed.asset_class.name == 'CRYPTO' or parsed.asset_class.value == 'CRYPTO': 
            return True
            
        # Forex is 24/5 (Mon-Fri)
        if parsed.asset_class.name == 'FOREX' or parsed.asset_class.value == 'FOREX': 
            return timestamp.weekday() < 5
            
        # Equities / Options (Mon-Fri)
        if timestamp.weekday() >= 5: 
            return False
            
        # Calculate precise New York time
        eastern = pytz.timezone('America/New_York')
        now_est = datetime.now(pytz.UTC).astimezone(eastern)
        est_hour = now_est.hour + now_est.minute / 60.0
        
        # 9.5 = 9:30 AM EST | 16.0 = 4:00 PM EST
        return 9.5 <= est_hour <= 16.0
    except Exception:
        # If it fails to parse, assume it's open to be safe
        return True

# Override the buggy library globally
is_market_open = custom_is_market_open
# ----------------------------------
"""

if "custom_is_market_open" not in content:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n" + patch + "\n")
    print("✅ Market Calendar bypassed successfully inside core/symbols.py! Equities UNLOCKED.")
else:
    print("⚡ Market hours fix is already applied in core/symbols.py.")
