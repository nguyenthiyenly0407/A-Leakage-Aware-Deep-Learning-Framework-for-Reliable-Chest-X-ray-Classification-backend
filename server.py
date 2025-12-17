from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from Database.connection import get_db, test_connection
from API.auth.API_login import router as auth_router
from fastapi.middleware.cors import CORSMiddleware
from API.patient.API_patient import router as patient_router
from API.analysis.API_analysis import router as analysis_router
app = FastAPI()
app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(analysis_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def startup():
    await test_connection()

@app.get("/db-ping")
async def db_ping(db: AsyncSession = Depends(get_db)):
    r = await db.execute(text("SELECT now()"))
    return {"db_time": str(r.scalar_one())}
