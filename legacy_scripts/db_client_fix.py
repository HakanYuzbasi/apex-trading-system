import sqlite3
import os

db_path = "data/apex_saas.db"
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Update the dedicated client_id column for IBKR
        c.execute("UPDATE broker_connections SET client_id = 99 WHERE broker_type LIKE '%ibkr%'")
        
        conn.commit()
        conn.close()
        print("✅ Database surgically updated! IBKR client_id is now 99.")
    except Exception as e:
        print(f"⚠️ Could not update DB: {e}")
else:
    print("⚠️ DB not found.")
