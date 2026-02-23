import os
import subprocess
import re

def restore_globals_css():
    filepath = "frontend/app/globals.css"
    if not os.path.exists(filepath):
        print(f"⚠️ Could not find {filepath}")
        return

    try:
        # Get the commit history for this file
        commits = subprocess.check_output(["git", "log", "--pretty=format:%h", filepath]).decode('utf-8').strip().split('\n')
    except Exception as e:
        print(f"⚠️ Git error: {e}")
        return

    for commit in commits:
        if not commit: continue
        try:
            content = subprocess.check_output(["git", "show", f"{commit}:{filepath}"]).decode('utf-8')
            # If it does NOT contain my custom injection, it's your original working file!
            if "Glassmorphic Panels" not in content and "/* Replaced @apply" not in content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ Restored globals.css to your original pristine state (Commit {commit})")
                return
        except Exception:
            continue

    print("⚠️ Could not find the original globals.css in git history.")

def restore_dashboard_visuals():
    filepath = "frontend/components/Dashboard.tsx"
    if not os.path.exists(filepath):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Revert Title
    content = content.replace(
        '<h1 className="text-4xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-sky-400 via-indigo-400 to-emerald-400 drop-shadow-sm pb-1">Apex Trading Terminal</h1>',
        '<h1 className="text-3xl font-semibold tracking-tight text-foreground">Apex Dashboard</h1>'
    )

    # 2. Revert the Connection Badge
    badge_pattern = r'<span\s+className=\{`inline-flex items-center gap-2\.5[^>]+>.*?live-ping.*?</span>'
    original_badge = """<span
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${isDisconnected
                  ? "bg-destructive/15 text-destructive"
                  : isStale
                    ? "bg-warning/15 text-warning"
                    : "bg-positive/15 text-positive"
                  }`}
              >
                {isDisconnected ? <WifiOff className="h-3.5 w-3.5" /> : <Wifi className="h-3.5 w-3.5" />}
                {isDisconnected ? "Disconnected" : isStale ? "Connected (Stale)" : "Connected"}
              </span>"""
    
    content = re.sub(badge_pattern, original_badge, content, flags=re.DOTALL)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Restored original Dashboard headers")

def fix_xai_chart_styles():
    filepath = "frontend/components/dashboard/ExplainableAIChart.tsx"
    if not os.path.exists(filepath): return
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Swap out the custom class for standard Shadcn UI classes
    content = content.replace('className="apex-panel ', 'className="bg-card border border-border ')
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Adapted Explainable AI Chart to match your standard theme")

if __name__ == "__main__":
    print("🔄 Rolling back custom UI injections...")
    restore_globals_css()
    restore_dashboard_visuals()
    fix_xai_chart_styles()
