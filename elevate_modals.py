import os
import glob

def boost_ui_z_index():
    # Find the UI folder
    ui_path = "frontend/components/ui"
    if not os.path.exists(ui_path):
        ui_path = "components/ui" # fallback if run from inside frontend
        if not os.path.exists(ui_path):
            print("⚠️ Could not find components/ui directory.")
            return

    # Shadcn components that use overlays and modals
    target_files = [
        "dialog.tsx",
        "command.tsx",
        "popover.tsx",
        "select.tsx",
        "sheet.tsx",
        "dropdown-menu.tsx",
        "tooltip.tsx"
    ]
    
    for file in target_files:
        filepath = os.path.join(ui_path, file)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Upgrade Tailwind's standard z-50 to a massive z-[99999]
            if "z-50" in content:
                content = content.replace("z-50", "z-[99999]")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Elevated z-index in {file}")

if __name__ == "__main__":
    print("🚀 Elevating Command Palette and Modals...")
    boost_ui_z_index()
