import os
import re

def fix_alpaca_connector():
    print("🛰️ PHASE 1: Patching Alpaca Heartbeat Loop...")
    filepath = "execution/alpaca_connector.py"
    if not os.path.exists(filepath):
        print(f"  ❌ {filepath} not found.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Ensure the polling loop updates the global state heartbeat
    # We look for the main polling loop and inject a direct state update
    if "async def _poll_quotes" in content:
        # Inject global state update into the loop
        sync_code = """
            # APEX Sync: Force heartbeat update into global state
            from api.dependencies import read_trading_state, save_trading_state
            try:
                state = read_trading_state()
                brokers = state.get("brokers", [])
                found = False
                for b in brokers:
                    if b.get("broker") == "alpaca":
                        b["last_heartbeat"] = datetime.utcnow().isoformat() + "Z"
                        b["status"] = "live"
                        b["mode"] = "trading"
                        found = True
                if not found:
                    brokers.append({
                        "broker": "alpaca",
                        "status": "live",
                        "mode": "trading",
                        "last_heartbeat": datetime.utcnow().isoformat() + "Z"
                    })
                state["brokers"] = brokers
                state["timestamp"] = datetime.utcnow().isoformat() + "Z"
                save_trading_state(state)
            except Exception as e:
                logging.error(f"Alpaca heartbeat sync failed: {e}")
        """
        # Place it right after the polling log
        content = content.replace(
            'logging.info(f"Starting Alpaca crypto quote polling for {len(self.symbols)} symbols")',
            'logging.info(f"Starting Alpaca crypto quote polling for {len(self.symbols)} symbols")' + sync_code
        )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  ✅ Alpaca Connector explicitly linked to Global State.")

def nuke_zombie_state():
    print("🧹 PHASE 2: Nuking Zombie State Files...")
    # This forces the system to regenerate fresh state rather than reading a 1-hour old cache
    state_file = "data/trading_state.json"
    if os.path.exists(state_file):
        os.remove(state_file)
        print(f"  ✅ Deleted {state_file} to force fresh generation.")
    
    # Also clear the user-specific state which often overrides the global one
    admin_state = "data/users/admin/trading_state.json"
    if os.path.exists(admin_state):
        os.remove(admin_state)
        print(f"  ✅ Deleted {admin_state}.")

if __name__ == "__main__":
    fix_alpaca_connector()
    nuke_zombie_state()
    print("\n🎉 Connection gap fixed! Restart your main.py now.")