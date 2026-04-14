import os

def fix_globals_css():
    path = "frontend/app/globals.css"
    if not os.path.exists(path):
        print(f"⚠️ Could not find {path}.")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = "@apply bg-background text-foreground antialiased;"
    replacement = """/* Replaced @apply for Tailwind v4 compatibility */
    background-color: hsl(var(--background));
    color: hsl(var(--foreground));
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;"""
    
    if target in content:
        content = content.replace(target, replacement)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Fixed @apply bg-background in {path}")
    else:
        print("⚠️ Target @apply string not found or already fixed.")

if __name__ == "__main__":
    print("🛠️ Patching Tailwind v4 CSS compatibility...")
    fix_globals_css()
