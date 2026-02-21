"""
services/common/db.py - Async PostgreSQL database engine and session management.

Uses SQLAlchemy 2.0 async with asyncpg driver.
Falls back gracefully when PostgreSQL is unavailable (e.g. during tests).
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

def get_database_url() -> str:
    _raw_url = os.getenv("DATABASE_URL", "")
    if not _raw_url:
        db_path = os.path.join(os.getcwd(), "data", "apex_saas.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path}"
    return _raw_url

# ---------------------------------------------------------------------------
# ORM Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# ---------------------------------------------------------------------------
# Engine & Session Factory (lazy singletons)
# ---------------------------------------------------------------------------

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """Return the global async engine (creates on first call)."""
    global _engine
    if _engine is None:
        db_url = get_database_url()
        kwargs = {
            "echo": os.getenv("SQL_ECHO", "false").lower() == "true",
            "pool_pre_ping": True,
        }
        if not db_url.startswith("sqlite"):
            kwargs["pool_size"] = 10
            kwargs["max_overflow"] = 20
            
        _engine = create_async_engine(db_url, **kwargs)
        logger.info("Database engine created: %s", db_url.split("@")[-1])
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the global async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[Optional[AsyncSession], None]:
    """FastAPI dependency that yields an async DB session, or None if unavailable."""
    yielded_session = False
    try:
        factory = get_session_factory()
        async with factory() as session:
            try:
                yielded_session = True
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    except Exception as exc:
        # If an endpoint error occurred after we yielded a session, propagate it.
        if yielded_session:
            raise
        logger.debug("Database unavailable, yielding None: %s", exc)
        yield None


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for use outside FastAPI (scripts, tests)."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create all tables (for development/testing only - use Alembic in prod)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")


async def close_db() -> None:
    """Dispose the engine connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connection pool closed")
