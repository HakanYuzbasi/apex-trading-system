import os
import re

filepath = "config.py"
if not os.path.exists(filepath):
    print("⚠️ Could not find config.py")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Force LIVE_TRADING to True
content = re.sub(r'LIVE_TRADING\s*=\s*False', 'LIVE_TRADING = True', content)

# 2. Force BROKER_MODE to "both" (Overwriting "ibkr" or "alpaca" if they exist)
if "BROKER_MODE =" in content:
    content = re.sub(r'BROKER_MODE\s*=\s*["\'].*?["\']', 'BROKER_MODE = "both"', content)
else:
    # If it's missing entirely, append it
    content += '\n    BROKER_MODE = "both"\n'

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Dual-Broker Auto-Routing is LOCKED IN. (Crypto -> Alpaca, TradFi -> IBKR)")
