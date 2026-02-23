import os

def restore_globals_css():
    path = "frontend/app/globals.css"
    if not os.path.exists(path):
        # Fallback if run from inside frontend folder
        path = "app/globals.css"
        if not os.path.exists(path):
            print("⚠️ Could not find globals.css")
            return

    css_content = """@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    /* Core Shadcn UI Variables */
    --background: 222 47% 4%;
    --foreground: 210 40% 98%;
    --card: 222 47% 6%;
    --card-foreground: 210 40% 98%;
    --popover: 222 47% 6%;
    --popover-foreground: 210 40% 98%;
    --primary: 199 89% 48%;
    --primary-foreground: 210 40% 98%;
    --secondary: 217 33% 17%;
    --secondary-foreground: 210 40% 98%;
    --muted: 217 33% 17%;
    --muted-foreground: 215 20.2% 65.1%;
    --accent: 217 33% 17%;
    --accent-foreground: 210 40% 98%;
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
    --border: 217 33% 17%;
    --input: 217 33% 17%;
    --ring: 199 89% 48%;
    --radius: 0.75rem;

    /* Custom Apex Colors */
    --positive: 142 71% 45%;
    --negative: 348 100% 61%;
    --warning: 38 92% 50%;
  }

  .dark {
    --background: 222 47% 4%;
    --foreground: 210 40% 98%;
    /* Same variables mapped for explicit dark mode */
  }
}

@layer base {
  * {
    border-color: hsl(var(--border));
  }
  body {
    background-color: hsl(var(--background));
    color: hsl(var(--foreground));
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background-image: 
      radial-gradient(circle at 50% -20%, rgba(14, 165, 233, 0.15) 0%, transparent 60%),
      linear-gradient(to bottom, rgba(10, 15, 30, 1) 0%, rgba(2, 6, 23, 1) 100%);
    background-attachment: fixed;
  }
}

@layer components {
  /* The Animated Background Grid */
  .apex-shell {
    position: relative;
  }
  .apex-shell::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
      linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: -1;
    mask-image: linear-gradient(to bottom, black 20%, transparent 100%);
  }

  /* Glassmorphic Panels */
  .apex-panel {
    position: relative;
    overflow: hidden;
    background-color: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(51, 65, 85, 0.5);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  }
  
  /* Glowing Hover Effects */
  .apex-interactive {
    transition: all 0.3s;
    cursor: pointer;
  }
  .apex-interactive:hover {
    transform: translateY(-4px);
    background-color: rgba(30, 41, 59, 0.6);
    border-color: rgba(71, 85, 105, 0.8);
  }

  /* Institutional Data Typography */
  .apex-kpi-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    letter-spacing: -0.05em;
    font-variant-numeric: tabular-nums;
  }

  /* Neon Glowing Accents */
  .text-positive { color: #4ade80; text-shadow: 0 0 15px rgba(74, 222, 128, 0.35); }
  .text-negative { color: #f87171; text-shadow: 0 0 15px rgba(248, 113, 113, 0.35); }
  .bg-positive { background-color: #4ade80; box-shadow: 0 0 12px rgba(74,222,128,0.5); }
  .bg-negative { background-color: #f87171; box-shadow: 0 0 12px rgba(248,113,113,0.5); }

  /* Live Pulse Indicator */
  .live-ping {
    position: relative;
    display: flex;
    height: 0.625rem;
    width: 0.625rem;
  }
  .live-ping-anim {
    animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;
    position: absolute;
    display: inline-flex;
    height: 100%;
    width: 100%;
    border-radius: 9999px;
    background-color: currentColor;
    opacity: 0.75;
  }
  @keyframes ping {
    75%, 100% { transform: scale(2.5); opacity: 0; }
  }
  .live-ping-dot {
    position: relative;
    display: inline-flex;
    border-radius: 9999px;
    height: 0.625rem;
    width: 0.625rem;
    background-color: currentColor;
  }
}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(css_content)
    print("✨ UI properly restored! All Shadcn variables injected.")

if __name__ == "__main__":
    restore_globals_css()
