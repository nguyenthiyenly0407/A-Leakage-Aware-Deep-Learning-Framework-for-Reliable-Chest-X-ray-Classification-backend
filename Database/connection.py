from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://postgres:abc123@localhost:5432/AI-Dermatology-Diagnosis-and-Patient-Management"

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db():
    async with SessionLocal() as session:
        yield session

async def test_connection():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
