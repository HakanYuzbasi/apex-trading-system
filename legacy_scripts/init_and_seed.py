import asyncio
import hashlib
import os
import sys

# Add the current directory to sys.path so we can import packages
sys.path.append(os.getcwd())

from services.common.db import init_db, db_session
from services.auth.models import UserModel, UserRoleModel, SubscriptionModel
from services.trading.models import BrokerConnectionModel, PortfolioModel, PositionModel, OrderModel, SignalModel
from services.common.schemas import SubscriptionTier

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

async def main():
    print("🚀 Initializing database tables...")
    await init_db()
    print("✅ Tables created.")

    async with db_session() as session:
        # Check if admin user exists
        from sqlalchemy import select
        result = await session.execute(select(UserModel).where(UserModel.username == "admin"))
        admin = result.scalar_one_or_none()

        if not admin:
            print("👤 Creating admin user...")
            admin = UserModel(
                username="admin",
                email="admin@apex.local",
                password_hash=hash_password("ApexAdmin!2026"),
                is_active=True
            )
            session.add(admin)
            await session.flush()  # Get admin.id

            # Add admin role
            role = UserRoleModel(user_id=admin.id, role="admin")
            session.add(role)

            # Add subscription
            sub = SubscriptionModel(user_id=admin.id, tier=SubscriptionTier.PRO)
            session.add(sub)
            
            print(f"✅ Admin user created (ID: {admin.id})")
        else:
            print("ℹ️ Admin user already exists.")

    print("✨ Database successfully initialized and seeded.")

if __name__ == "__main__":
    asyncio.run(main())
