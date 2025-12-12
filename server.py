from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from Database.connection import get_db, test_connection

app = FastAPI()

@app.on_event("startup")
async def startup():
    await test_connection()

@app.get("/db-ping")
async def db_ping(db: AsyncSession = Depends(get_db)):
    r = await db.execute(text("SELECT now()"))
    return {"db_time": str(r.scalar_one())}
