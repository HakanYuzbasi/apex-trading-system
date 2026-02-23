import json
from pathlib import Path
cmd_file = Path("data/users/admin/trading_control_commands.json")
cmd_file.parent.mkdir(parents=True, exist_ok=True)
with open(cmd_file, "w", encoding="utf-8") as f:
    json.dump({
        "kill_switch_reset_requested": True,
        "requested_by": "admin",
        "request_id": "final_cold_start",
        "reason": "Clearing state after math fix"
    }, f)
print("✅ Issued final reset command!")
