import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

# ✅ Đảm bảo tên DB này đã được tạo trong pgAdmin 4
DATABASE_URL = "postgresql+asyncpg://postgres:abc123@localhost:5432/AI-Dermatology-Diagnosis-and-Patient-Management"

engine = create_async_engine(DATABASE_URL, echo=True) # Để True để xem log SQL cho rõ
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db():
    async with SessionLocal() as session:
        yield session

async def test_connection():
    try:
        print("--- Đang kết nối thử nè Ly... ---")
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print(f"--- Kết nối THÀNH CÔNG! Kết quả: {result.fetchone()} ---")
    except Exception as e:
        print(f"--- LỖI RỒI MÁ ƠI: {e} ---")

# 👇 THÊM ĐOẠN NÀY ĐỂ CHẠY FILE ĐỘC LẬP
if __name__ == "__main__":
    asyncio.run(test_connection())