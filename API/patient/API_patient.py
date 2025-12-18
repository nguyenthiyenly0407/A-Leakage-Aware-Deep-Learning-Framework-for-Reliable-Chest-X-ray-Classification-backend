from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List

from Database.connection import get_db
from .patient_schema import PatientCreate, PatientOut
from typing import Optional
from fastapi import Query

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
async def get_patients(
    keyword: Optional[str] = Query(None), # Nhận tham số ?keyword=...
    db: AsyncSession = Depends(get_db)
):
    # Câu query gốc
    sql_query = """
        SELECT id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at
        FROM patient
    """
    
    # Nếu có keyword thì nối thêm điều kiện WHERE
    params = {}
    if keyword:
        # %keyword% nghĩa là tìm chuỗi con nằm ở bất cứ đâu
        search_term = f"%{keyword}%"
        params["search"] = search_term
        
        # Logic: Tên HOẶC SĐT HOẶC Giới tính HOẶC Năm sinh khớp với từ khóa
        # Lưu ý: EXTRACT(YEAR...) để lấy năm sinh ra so sánh
        sql_query += """
            WHERE 
                name_patient ILIKE :search 
                OR phone_number ILIKE :search
                OR gender ILIKE :search
                OR CAST(EXTRACT(YEAR FROM date_of_birth_patient) AS TEXT) ILIKE :search
        """

    # Luôn luôn sắp xếp tăng dần theo ID (như bạn yêu cầu trước đó)
    sql_query += " ORDER BY id_patient ASC;"

    q = text(sql_query)
    r = await db.execute(q, params)
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

