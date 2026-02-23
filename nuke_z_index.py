import os

def drop_z_index_nuke():
    path = "frontend/app/globals.css"
    if not os.path.exists(path):
        # Fallback if inside frontend dir
        path = "app/globals.css"
        if not os.path.exists(path):
            print("⚠️ Could not find globals.css")
            return
            
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace any existing glass-card z-index rules we tried before
    content = content.replace("z-index: 1 !important;", "z-index: 0 !important;")
        
    nuke_css = """

/* ========================================= */
/* ABSOLUTE Z-INDEX OVERRIDE FOR MODALS      */
/* ========================================= */

/* 1. Force Shadcn's default modal layer to the absolute maximum */
.z-50, 
[data-radix-portal], 
div[role="dialog"] { 
  z-index: 2147483647 !important; 
}

/* 2. Force the glass cards to stay flat on the bottom layer */
.glass-card { 
  z-index: 0 !important; 
  isolation: isolate; 
  transform: translateZ(0); 
}
"""
    
    if "ABSOLUTE Z-INDEX OVERRIDE FOR MODALS" not in content:
        with open(path, "a", encoding="utf-8") as f:
            f.write(nuke_css)
        print("✅ Dropped the CSS Z-Index Nuke! Modals forced to the front.")
    else:
        print("⚡ Nuke already applied.")

if __name__ == "__main__":
    drop_z_index_nuke()
