"""Database connection pooling utility for efficient database access."""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
import logging

try:
    from sqlalchemy.ext.asyncio import (
        create_async_engine,
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
    )
    from sqlalchemy.pool import NullPool, QueuePool
except ImportError:
    # SQLAlchemy not installed - provide stub for type hints
    AsyncEngine = None
    AsyncSession = None
    async_sessionmaker = None

logger = logging.getLogger(__name__)


class DatabasePool:
    """Async database connection pool manager."""

    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ):
        """Initialize database pool.
        
        Args:
            database_url: Database connection URL
            pool_size: Number of connections to maintain
            max_overflow: Max connections beyond pool_size
            pool_timeout: Seconds to wait for connection
            pool_recycle: Seconds before recycling connection
            echo: Echo SQL statements (debug)
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        
        self._engine: Optional[AsyncEngine] = None
        self._session_factory = None
        self._initialized = False

    async def initialize(self):
        """Initialize database engine and session factory."""
        if self._initialized:
            return

        try:
            self._engine = create_async_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=self.echo,
                future=True,
            )

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            self._initialized = True
            logger.info("Database pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def dispose(self):
        """Dispose of database engine and connections."""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("Database pool disposed")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session from pool.
        
        Usage:
            async with pool.session() as session:
                result = await session.execute(query)
        """
        if not self._initialized:
            await self.initialize()

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def execute(self, query, params=None):
        """Execute a query directly.
        
        Args:
            query: SQL query or SQLAlchemy statement
            params: Query parameters
        
        Returns:
            Query result
        """
        async with self.session() as session:
            result = await session.execute(query, params or {})
            return result

    async def health_check(self) -> bool:
        """Check database connection health.
        
        Returns:
            True if database is accessible
        """
        try:
            if not self._initialized:
                return False

            async with self._engine.connect() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_pool_status(self) -> dict:
        """Get current pool status.
        
        Returns:
            Dict with pool statistics
        """
        if not self._engine or not self._initialized:
            return {"initialized": False}

        pool = self._engine.pool
        return {
            "initialized": True,
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "timeout": self.pool_timeout,
        }


# Global pool instance
_global_pool: Optional[DatabasePool] = None


def get_db_pool() -> DatabasePool:
    """Get global database pool instance.
    
    Returns:
        Global DatabasePool instance
    """
    global _global_pool
    if _global_pool is None:
        raise RuntimeError("Database pool not initialized. Call initialize_db_pool() first.")
    return _global_pool


def initialize_db_pool(
    database_url: str,
    pool_size: int = 10,
    max_overflow: int = 20,
    **kwargs
) -> DatabasePool:
    """Initialize global database pool.
    
    Args:
        database_url: Database connection URL
        pool_size: Number of connections to maintain
        max_overflow: Max connections beyond pool_size
        **kwargs: Additional DatabasePool arguments
    
    Returns:
        Initialized DatabasePool instance
    """
    global _global_pool
    _global_pool = DatabasePool(
        database_url=database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        **kwargs
    )
    return _global_pool


# Example usage:
# 
# # Initialize pool at startup
# pool = initialize_db_pool(
#     database_url="postgresql+asyncpg://user:pass@localhost/db",
#     pool_size=10,
#     max_overflow=20
# )
# await pool.initialize()
# 
# # Use in endpoints
# async with get_db_pool().session() as session:
#     result = await session.execute(select(Trade))
#     trades = result.scalars().all()
# 
# # Cleanup at shutdown
# await pool.dispose()
