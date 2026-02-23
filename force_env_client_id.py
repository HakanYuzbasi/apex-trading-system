import os
import re

print("🔄 Forcing IBKR Client ID 99 via .env and config.py...")

# 1. Update .env file 
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        env_content = f.read()
    
    if "IBKR_CLIENT_ID" in env_content:
        env_content = re.sub(r'IBKR_CLIENT_ID\s*=\s*\d+', 'IBKR_CLIENT_ID=99', env_content)
    else:
        env_content += "\nIBKR_CLIENT_ID=99\n"

    with open(env_path, "w") as f:
        f.write(env_content)
    print("✅ .env file updated: IBKR_CLIENT_ID=99")

# 2. Update config.py
config_path = "config.py"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config_content = f.read()
        
    if "IBKR_CLIENT_ID" in config_content:
        config_content = re.sub(r'IBKR_CLIENT_ID\s*=\s*\d+', 'IBKR_CLIENT_ID = 99', config_content)
    else:
        config_content += "\nIBKR_CLIENT_ID = 99\n"
    
    with open(config_path, "w") as f:
        f.write(config_content)
    print("✅ config.py updated: IBKR_CLIENT_ID = 99")

