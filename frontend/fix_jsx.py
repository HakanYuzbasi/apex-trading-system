import os
import re

filepath = "components/Dashboard.tsx"
if not os.path.exists(filepath):
    print("⚠️ Could not find Dashboard.tsx")
    exit(1)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# This regex matches the exact leftover dangling tags and removes them
bad_syntax = r'\s*<span className="live-ping-dot"></span>\s*</div>\s*\{isDisconnected \? "System Offline" : isStale \? "Feed Stale" : "Live Engine"\}\s*</span>'

new_content = re.sub(bad_syntax, '', content)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(new_content)

print("✅ Fixed dangling JSX syntax error!")
