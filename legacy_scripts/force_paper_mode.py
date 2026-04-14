import sqlite3
import os
import re
from pathlib import Path

# 1. Force the SQLite Database to Paper Mode
db_path = Path("data/apex_saas.db")
if db_path.exists():
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Find the exact table name for broker connections
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in c.fetchall()]
        
        for table in tables:
            if "broker" in table.lower() or "connection" in table.lower():
                try:
                    # Force ALL connections to paper and active
                    c.execute(f"UPDATE {table} SET environment = 'paper', is_active = 1")
                    conn.commit()
                    print(f"✅ DB Table '{table}' forced to PAPER mode.")
                except Exception as e:
                    pass
        conn.close()
    except Exception as e:
        print(f"⚠️ DB Error: {e}")
else:
    print("⚠️ DB not found at data/apex_saas.db")

# 2. Hardcode Config.py just in case
config_path = "config.py"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Aggressively replace any LIVE_TRADING variable
    content = re.sub(r'^LIVE_TRADING\s*=.*$', 'LIVE_TRADING = True', content, flags=re.MULTILINE)
    content = re.sub(r'^BROKER_MODE\s*=.*$', 'BROKER_MODE = "both"', content, flags=re.MULTILINE)
    
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ config.py hardcoded to LIVE_TRADING = True and BROKER_MODE = 'both'")

# 3. Hardcode .env just in case
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = re.sub(r'^LIVE_TRADING=.*$', 'LIVE_TRADING=True', content, flags=re.MULTILINE)
    content = re.sub(r'^BROKER_MODE=.*$', 'BROKER_MODE=both', content, flags=re.MULTILINE)
    
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ .env hardcoded to LIVE_TRADING=True and BROKER_MODE=both")

