import os

def fix_palette_z_index():
    path = "frontend/app/globals.css"
    if not os.path.exists(path):
        # Fallback if run from inside frontend folder
        path = "app/globals.css"
        if not os.path.exists(path):
            print("⚠️ Could not find globals.css")
            return
            
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    if "data-radix-portal" not in content:
        fix_css = """
/* Force Command Palette and Modals to the Absolute Top */
[data-radix-portal], 
[role="dialog"], 
[data-state="open"] > .fixed,
.fixed.inset-0 {
  z-index: 99999 !important;
}
"""
        with open(path, "a", encoding="utf-8") as f:
            f.write(fix_css)
        print("✅ Fixed Command Palette z-index!")
    else:
        print("⚡ Z-index fix already applied.")

if __name__ == "__main__":
    fix_palette_z_index()
