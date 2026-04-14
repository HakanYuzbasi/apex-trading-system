import sqlite3
import os
import json
import re

print("🔄 Syncing IBKR Ports to 7497...")

# 1. Update the Database Credentials
db_path = "data/apex_saas.db"
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Look for the credentials column in the broker connections table
        c.execute("SELECT id, credentials FROM broker_connections WHERE broker_type LIKE '%ibkr%'")
        rows = c.fetchall()
        
        for row in rows:
            row_id = row[0]
            try:
                creds = json.loads(row[1]) if row[1] else {}
            except:
                creds = {}
                
            # Force the port to 7497
            creds["port"] = 7497
            creds["paper_port"] = 7497
            creds["host"] = "127.0.0.1"
            
            c.execute("UPDATE broker_connections SET credentials = ? WHERE id = ?", (json.dumps(creds), row_id))
            
        conn.commit()
        conn.close()
        print("✅ Database updated: IBKR connection locked to port 7497.")
    except Exception as e:
        print(f"⚠️ Could not update DB: {e}")

# 2. Update .env file 
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        env_content = f.read()
    
    # Update or append the port settings
    if "IBKR_PORT" in env_content:
        env_content = re.sub(r'IBKR_PORT\s*=\s*\d+', 'IBKR_PORT=7497', env_content)
    else:
        env_content += "\nIBKR_PORT=7497\n"
        
    if "IBKR_PAPER_PORT" in env_content:
        env_content = re.sub(r'IBKR_PAPER_PORT\s*=\s*\d+', 'IBKR_PAPER_PORT=7497', env_content)
    else:
        env_content += "IBKR_PAPER_PORT=7497\n"

    with open(env_path, "w") as f:
        f.write(env_content)
    print("✅ .env file updated: Defaults set to port 7497.")

# 3. Update config.py just in case
config_path = "config.py"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_content = f.read()
        
    config_content = re.sub(r'IBKR_PORT\s*=\s*\d+', 'IBKR_PORT = 7497', config_content)
    config_content = re.sub(r'IBKR_PAPER_PORT\s*=\s*\d+', 'IBKR_PAPER_PORT = 7497', config_content)
    
    with open(config_path, "w") as f:
        f.write(config_content)
    print("✅ config.py updated.")

print("🚀 IBKR Port Sync Complete!")
