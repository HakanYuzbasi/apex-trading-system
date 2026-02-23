import os

filepath = "main.py"
if not os.path.exists(filepath):
    print("⚠️ Could not find main.py")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

patch = """
# --- FIX: MARKET HOURS OVERRIDE ---
import pytz
from datetime import datetime
def custom_is_market_open(symbol, timestamp):
    try:
        from core.symbols import parse_symbol, AssetClass
        parsed = parse_symbol(symbol)
        
        if parsed.asset_class == AssetClass.CRYPTO: 
            return True
        if parsed.asset_class == AssetClass.FOREX: 
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
        return True

# Override the buggy library globally
import core.symbols
core.symbols.is_market_open = custom_is_market_open
is_market_open = custom_is_market_open
# ----------------------------------
"""

if "MARKET HOURS OVERRIDE" not in content:
    # Inject it right below the imports
    import_idx = content.find("class ApexTradingSystem:")
    if import_idx != -1:
        content = content[:import_idx] + patch + "\n" + content[import_idx:]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Market Calendar successfully bypassed! Equities are now unlocked.")
    else:
        print("⚠️ Could not find injection point.")
else:
    print("⚡ Market hours fix is already applied.")

