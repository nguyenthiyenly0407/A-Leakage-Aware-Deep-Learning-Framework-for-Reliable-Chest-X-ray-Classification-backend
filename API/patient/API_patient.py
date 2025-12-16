from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List

from Database.connection import get_db
from .patient_schema import PatientCreate, PatientOut

router = APIRouter(prefix="/patients", tags=["patients"])

# ✅ Add patient
@router.post("", response_model=PatientOut)
async def add_patient(payload: PatientCreate, db: AsyncSession = Depends(get_db)):
    q = text("""
        INSERT INTO patient (name_patient, date_of_birth_patient, gender, phone_number)
        VALUES (:name_patient, :date_of_birth_patient, :gender, :phone_number)
        RETURNING id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at;
    """)
    r = await db.execute(q, payload.model_dump())
    await db.commit()

    row = r.mappings().first()
    return row

# ✅ Get list patients
@router.get("", response_model=List[PatientOut])
async def get_patients(db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at
        FROM patient
        ORDER BY id_patient DESC;
    """)
    r = await db.execute(q)
    rows = r.mappings().all()
    return rows

# ✅ Get one patient by id
@router.get("/{id_patient}", response_model=PatientOut)
async def get_patient(id_patient: int, db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at
        FROM patient
        WHERE id_patient = :id_patient
        LIMIT 1;
    """)
    r = await db.execute(q, {"id_patient": id_patient})
    row = r.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Patient not found")

    return row
