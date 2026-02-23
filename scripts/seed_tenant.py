import asyncio
import os
import sys

# Add the project directory to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ApexConfig
from models.broker import BrokerType
from services.broker.service import broker_service

async def seed_tenant():
    # Only seed if there are no existing tenants
    tenants = await broker_service.list_tenant_ids()
    if tenants:
        print(f"✅ Database already has tenants: {tenants}. Skipping seed.")
        return
        
    api_key = getattr(ApexConfig, "ALPACA_API_KEY", "")
    secret_key = getattr(ApexConfig, "ALPACA_SECRET_KEY", "")
    
    if not api_key or not secret_key:
        print("❌ ALPACA_API_KEY or ALPACA_SECRET_KEY not fully configured in environment. Skipping seed.")
        return
        
    print("🌱 Seeding initial tenant broker connection...")
    
    try:
        # Create a default connection for the default user
        conn = await broker_service.create_connection(
            user_id="default_user",
            broker_type=BrokerType.ALPACA,
            name="System Config Alpaca",
            credentials={
                "api_key": api_key,
                "secret_key": secret_key
            },
            environment="paper"  # Force paper to allow validation to pass since the key is a paper key (starts with PK)
        )
        print(f"✅ Successfully created tenant broker connection: {conn.id}")
        print("🪐 ExecutionManager should pick this up automatically within 10 seconds.")
    except Exception as e:
        print(f"❌ Failed to seed tenant: {e}")

if __name__ == "__main__":
    asyncio.run(seed_tenant())
