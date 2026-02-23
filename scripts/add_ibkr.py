import asyncio
import os
import sys

# Add the project directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ApexConfig
from models.broker import BrokerType
from services.broker.service import broker_service

async def add_ibkr():
    print("🌱 Adding IBKR tenant broker connection...")
    
    # Check if we already have it
    connections = await broker_service.list_connections("default_user")
    has_ibkr = any(c.broker_type == BrokerType.IBKR for c in connections)
    if has_ibkr:
        print("✅ IBKR connection already exists.")
        return

    try:
        # Create a connection for the default user
        conn = await broker_service.create_connection(
            user_id="default_user",
            broker_type=BrokerType.IBKR,
            name="System Config IBKR",
            credentials={
                "host": getattr(ApexConfig, "IBKR_HOST", "127.0.0.1"),
                "port": int(getattr(ApexConfig, "IBKR_PORT", 7497)),
                "client_id": int(getattr(ApexConfig, "IBKR_CLIENT_ID", 1))
            },
            client_id=int(getattr(ApexConfig, "IBKR_CLIENT_ID", 1)),
            environment="paper"
        )
        print(f"✅ Successfully created IBKR tenant broker connection: {conn.id}")
    except Exception as e:
        print(f"❌ Failed to add IBKR tenant: {e}")

if __name__ == "__main__":
    asyncio.run(add_ibkr())
