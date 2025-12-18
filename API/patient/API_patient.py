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
        ORDER BY id_patient ASC;
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

@router.put("/{id_patient}", response_model=PatientOut)
async def update_patient(id_patient: int, payload: PatientCreate, db: AsyncSession = Depends(get_db)):
    # Câu lệnh Update
    q = text("""
        UPDATE patient
        SET name_patient = :name_patient,
            date_of_birth_patient = :date_of_birth_patient,
            gender = :gender,
            phone_number = :phone_number
        WHERE id_patient = :id_patient
        RETURNING id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at;
    """)
    
    # Gộp id vào params để bind vào câu query
    params = payload.model_dump()
    params["id_patient"] = id_patient

    r = await db.execute(q, params)
    await db.commit()

    # Lấy dòng vừa update trả về
    row = r.mappings().first()

    # Nếu không tìm thấy ID thì báo lỗi 404
    if not row:
        raise HTTPException(status_code=404, detail="Patient not found")

    return row


@router.delete("/{id_patient}")
async def delete_patient(id_patient: int, db: AsyncSession = Depends(get_db)):
    try:
        q = text("""
            DELETE FROM patient 
            WHERE id_patient = :id_patient 
            RETURNING id_patient;
        """)
        r = await db.execute(q, {"id_patient": id_patient})
        await db.commit()

        # Kiểm tra xem có xóa được dòng nào không
        deleted_id = r.scalar()
        
        if not deleted_id:
            raise HTTPException(status_code=404, detail="Patient not found")

        return {"message": "Deleted successfully", "id_patient": deleted_id}

    except Exception as e:
        # Bắt lỗi nếu bệnh nhân đang dính khóa ngoại (Foreign Key) bên bảng skin_analysis
        if "foreign key constraint" in str(e).lower():
            raise HTTPException(
                status_code=400, 
                detail="Không thể xóa bệnh nhân này vì đã có dữ liệu phân tích da (History)."
            )
        raise e