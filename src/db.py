"""Database engine, session, and base model setup.

Works with both PostgreSQL (Render) and SQLite (local dev).
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from src.config import settings

_is_sqlite = "sqlite" in settings.database_url

if _is_sqlite:
    async_engine = create_async_engine(
        settings.database_url,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sync_engine = create_engine(
        settings.database_url_sync,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    async_engine = create_async_engine(settings.database_url, echo=False)
    sync_engine = create_engine(settings.database_url_sync, echo=False)

AsyncSessionLocal = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_tables():
    """Create all tables."""
    from src.models import (  # noqa: F401
        AgentCycle, Hypothesis, IntegrityCheck,
        Paper, Prediction, Protocol, Simulation,
        KnowledgeNode, KnowledgeEdge,
    )
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
