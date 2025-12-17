from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List

from Database.connection import get_db
from .analysis_schema import AnalysisCreate, AnalysisOut

router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.post("", response_model=AnalysisOut)
async def add_analysis(payload: AnalysisCreate, db: AsyncSession = Depends(get_db)):
    q = text("""
        INSERT INTO skin_analysis
        (id_patient, id_doctor, image_url, ai_prediction, ai_confidence, doctor_diagnosis, note)
        VALUES
        (:id_patient, :id_doctor, :image_url, :ai_prediction, :ai_confidence, :doctor_diagnosis, :note)
        RETURNING id_analysis, id_patient, id_doctor, image_url, ai_prediction, ai_confidence, doctor_diagnosis, note, created_at;
    """)
    r = await db.execute(q, payload.model_dump())
    await db.commit()
    return r.mappings().first()

@router.get("", response_model=List[AnalysisOut])
async def get_all_analysis(db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_analysis, id_patient, id_doctor, image_url, ai_prediction, ai_confidence,
               doctor_diagnosis, note, created_at
        FROM skin_analysis
        ORDER BY id_analysis DESC;
    """)
    r = await db.execute(q)
    return r.mappings().all()

@router.get("/patient/{id_patient}", response_model=List[AnalysisOut])
async def get_analysis_by_patient(id_patient: int, db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_analysis, id_patient, id_doctor, image_url, ai_prediction, ai_confidence,
               doctor_diagnosis, note, created_at
        FROM skin_analysis
        WHERE id_patient = :id_patient
        ORDER BY id_analysis DESC;
    """)
    r = await db.execute(q, {"id_patient": id_patient})
    return r.mappings().all()

@router.get("/{id_analysis}", response_model=AnalysisOut)
async def get_one_analysis(id_analysis: int, db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_analysis, id_patient, id_doctor, image_url, ai_prediction, ai_confidence,
               doctor_diagnosis, note, created_at
        FROM skin_analysis
        WHERE id_analysis = :id_analysis
        LIMIT 1;
    """)
    r = await db.execute(q, {"id_analysis": id_analysis})
    row = r.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return row
