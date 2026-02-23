import os

def fix_dashboard_ts():
    path = "components/Dashboard.tsx"
    if not os.path.exists(path):
        print(f"⚠️ Could not find {path}.")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Apply the double-cast (as unknown as Record<string, unknown>) suggested by TS
    content = content.replace(
        "(position as Record<string, unknown>)",
        "(position as unknown as Record<string, unknown>)"
    )
    
    # Also fix the XAI Shap data mock casting which might complain similarly
    content = content.replace(
        "(wsData as any)?.shap_values",
        "((wsData as unknown as Record<string, unknown>)?.shap_values as Record<string, number>)"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Fixed strict TypeScript casts in Dashboard.tsx")

if __name__ == "__main__":
    fix_dashboard_ts()
