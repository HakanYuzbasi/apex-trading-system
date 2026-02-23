import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all models to ensure they register with Base.metadata
from services.common.db import get_engine, Base

async def create_tables():
    engine = get_engine()
    async with engine.begin() as conn:
        print("Creating all missing tables...")
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully.")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(create_tables())
