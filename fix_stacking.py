import os

def fix_z_index():
    path = "frontend/app/globals.css"
    if not os.path.exists(path):
        path = "app/globals.css"
        if not os.path.exists(path):
            print("⚠️ Could not find globals.css")
            return
            
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 1. Force the cards to stay low in the stack
    if "z-index: 1 !important;" not in content and ".glass-card {" in content:
        content = content.replace(
            ".glass-card {",
            ".glass-card {\n  z-index: 1 !important;\n  position: relative;"
        )
        
    # 2. Force all popups, portals, and fixed headers to the absolute maximum integer
    if "2147483647" not in content:
        nuclear_css = """
/* Ultimate Z-Index Hammer for Command Palette & Modals */
div[role="dialog"],
div[data-radix-portal],
div[cmdk-dialog],
[cmdk-overlay],
.fixed {
  z-index: 2147483647 !important;
}
"""
        content += nuclear_css
        
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Stacking contexts rebalanced! Popups forced to the front.")

if __name__ == "__main__":
    fix_z_index()
