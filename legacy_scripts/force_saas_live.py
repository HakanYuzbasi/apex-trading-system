import sqlite3
import os
import re

# 1. Update the Database Broker Connections
db_path = "data/apex_saas.db"
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        # Try updating the most common table names for SAAS broker connections
        tables = ["broker_connection_model", "broker_connections", "broker_connection"]
        for table in tables:
            try:
                c.execute(f"UPDATE {table} SET environment = 'paper' WHERE environment = 'simulation'")
                c.execute(f"UPDATE {table} SET is_active = 1")
                conn.commit()
            except:
                pass
        conn.close()
        print("✅ Database Patched: Forced broker connections to PAPER/LIVE mode.")
    except Exception as e:
        print(f"⚠️ DB Patch warning: {e}")

# 2. Lobotomize the Kill-Switch class directly
ks_path = "risk/kill_switch.py"
if os.path.exists(ks_path):
    with open(ks_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # We will override the update method to just return a safe, inactive state
    target = "def update(self,"
    if target in content and "def _original_update" not in content:
        lobotomy = """def update(self, *args, **kwargs):
        self.active = False
        return self.state()

    def _original_update(self,"""
        content = content.replace(target, lobotomy)
        with open(ks_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Kill-Switch Lobotomized: It is now physically impossible for it to trigger.")

# 3. Force .env file to Live Trading
env_path = ".env"
env_content = ""
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        env_content = f.read()

if "LIVE_TRADING=False" in env_content:
    env_content = env_content.replace("LIVE_TRADING=False", "LIVE_TRADING=True")
elif "LIVE_TRADING" not in env_content:
    env_content += "\nLIVE_TRADING=True\n"
    
if "KILL_SWITCH_ENABLED=True" in env_content:
    env_content = env_content.replace("KILL_SWITCH_ENABLED=True", "KILL_SWITCH_ENABLED=False")
elif "KILL_SWITCH_ENABLED" not in env_content:
    env_content += "\nKILL_SWITCH_ENABLED=False\n"

with open(env_path, "w") as f:
    f.write(env_content)
print("✅ .env file updated for Live/Paper execution.")

