import asyncio
from Database.connection import engine
from Database.model import Base # Đảm bảo file models.py đã có class Base

async def init_db():
    print("--- Đang xây nhà... Chờ xíu nha Ly! ---")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("--- [XONG] Đã tạo xong các bảng login_doctor, patients, analysis! ---")

if __name__ == "__main__":
    asyncio.run(init_db())