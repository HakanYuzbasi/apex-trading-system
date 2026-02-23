import sqlite3
import os
import json

db_path = "data/apex_saas.db"
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute("SELECT id, credentials FROM broker_connections WHERE broker_type LIKE '%ibkr%'")
        rows = c.fetchall()
        
        for row in rows:
            row_id = row[0]
            try:
                creds = json.loads(row[1]) if row[1] else {}
            except:
                creds = {}
                
            # Assign a high client ID to avoid any lingering ghost sessions
            creds["client_id"] = 99
            
            c.execute("UPDATE broker_connections SET credentials = ? WHERE id = ?", (json.dumps(creds), row_id))
            
        conn.commit()
        conn.close()
        print("✅ Assigned new Client ID (99) to APEX for IBKR.")
    except Exception as e:
        print(f"⚠️ Could not update DB: {e}")
else:
    print("⚠️ DB not found.")
