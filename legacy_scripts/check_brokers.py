import sqlite3
import os

db_path = "data/apex_saas.db"
if not os.path.exists(db_path):
    print("⚠️ DB not found!")
    exit(1)

conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%broker%';")
tables = [t[0] for t in c.fetchall()]

for table in tables:
    print(f"\n--- Reading from {table} ---")
    try:
        # Looking for broker connections
        c.execute(f"SELECT id, user_id, broker_type, environment, is_active FROM {table}")
        rows = c.fetchall()
        if not rows:
            print("No connections found in this table.")
        for row in rows:
            print(f"ID: {row[0]} | User: {row[1]} | Broker: {row[2].upper()} | Mode: {row[3].upper()} | Active: {'✅ YES' if row[4] else '❌ NO'}")
    except Exception as e:
        print(f"Could not read {table}: {e}")

conn.close()
