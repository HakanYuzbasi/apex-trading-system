import os
import re

filepath = "config.py"
if not os.path.exists(filepath):
    print("⚠️ Could not find config.py")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Force the KILL_SWITCH_ENABLED toggle to False permanently
content = re.sub(r'KILL_SWITCH_ENABLED\s*=\s*True', 'KILL_SWITCH_ENABLED = False', content)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Kill-Switch permanently disabled in config.")
