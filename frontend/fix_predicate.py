import os
import re

def fix_dashboard_predicate():
    path = "components/Dashboard.tsx"
    if not os.path.exists(path):
        print(f"⚠️ Could not find {path}.")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace the strict type predicate with a standard filter and array cast
    content = re.sub(
        r'\}\)\.filter\(\(row\):\s*row\s*is\s*CockpitPosition\s*=>\s*row\s*!==\s*null\);',
        r'}).filter((row) => row !== null) as CockpitPosition[];',
        content
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Fixed strict type predicate in Dashboard.tsx")

if __name__ == "__main__":
    fix_dashboard_predicate()
