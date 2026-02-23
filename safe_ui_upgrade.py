import os
import re

def upgrade_ui_safely():
    # 1. Safely Append CSS (Does not overwrite existing Shadcn variables)
    css_path = "frontend/app/globals.css"
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

        if "/* APEX PREMIUM UI */" not in css_content:
            premium_css = """

/* APEX PREMIUM UI - Appended Safely */
body::before {
  content: "";
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    linear-gradient(to right, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  mask-image: radial-gradient(circle at 50% 0%, rgba(0,0,0,1) 0%, transparent 80%);
  -webkit-mask-image: radial-gradient(circle at 50% 0%, rgba(0,0,0,1) 0%, transparent 80%);
  z-index: -1;
  pointer-events: none;
}

.glass-card {
  background-color: hsl(var(--card) / 0.4) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid hsl(var(--border) / 0.5) !important;
  box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.3) !important;
  transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
  border-color: hsl(var(--border) / 0.9) !important;
  transform: translateY(-2px);
}
"""
            with open(css_path, "a", encoding="utf-8") as f:
                f.write(premium_css)
            print("✅ Subtle CSS Grid and Glassmorphic classes safely appended.")
        else:
            print("⚡ Premium CSS already present.")

    # 2. Safely Inject into Dashboard.tsx
    tsx_path = "frontend/components/Dashboard.tsx"
    if os.path.exists(tsx_path):
        with open(tsx_path, "r", encoding="utf-8") as f:
            tsx_content = f.read()

        # Add glass-card to existing Cards safely
        if "glass-card" not in tsx_content:
            new_tsx = re.sub(r'<Card className="([^"]*)"', r'<Card className="\1 glass-card"', tsx_content)
            new_tsx = new_tsx.replace("<Card>", '<Card className="glass-card">')

            with open(tsx_path, "w", encoding="utf-8") as f:
                f.write(new_tsx)
            print("✅ Dashboard cards upgraded to premium glassmorphism.")
        else:
            print("⚡ Dashboard already upgraded.")

if __name__ == "__main__":
    print("✨ Applying Non-Destructive Premium UI...")
    upgrade_ui_safely()
