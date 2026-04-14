import os
import re

filepath = "config.py"
if not os.path.exists(filepath):
    print("⚠️ Could not find config.py")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Turn on API routing
if "LIVE_TRADING = False" in content:
    content = content.replace("LIVE_TRADING = False", "LIVE_TRADING = True")
    print("✅ Flipped LIVE_TRADING to True (API Routing Enabled)")
else:
    print("⚡ LIVE_TRADING is already True or dynamically set.")

# 2. Ensure Alpaca is enabled in the broker mode
if "BROKER_MODE =" in content:
    # Safely update the broker mode to 'both' or 'alpaca'
    content = re.sub(r'BROKER_MODE\s*=\s*["\']ibkr["\']', 'BROKER_MODE = "both"', content)
    print("✅ Ensured BROKER_MODE allows Alpaca.")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("🚀 Configuration updated. Ready for broker execution!")
