from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Optional

from Database.connection import get_db
from .patient_schema import PatientCreate, PatientOut

# 👇 IMPORT QUAN TRỌNG: Lấy thông tin bác sĩ đang đăng nhập
from API.auth.security import get_current_user 

router = APIRouter(prefix="/patients", tags=["patients"])

# =================================================================
# ✅ 1. ADD PATIENT (Tự động gán cho bác sĩ đang login)
# =================================================================
@router.post("", response_model=PatientOut)
async def add_patient(
    payload: PatientCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user) # 🔒 Bắt buộc đăng nhập
):
    # Lấy ID bác sĩ từ token
    # Lưu ý: check lại xem trong token bạn lưu key là 'id_account' hay 'id_doctor'
    # Nếu bảng login_doctor dùng id_doctor, thì chỗ này khả năng là current_user['id_doctor']
    # Nhưng mình giữ id_account theo code cũ của bạn, bạn tự check nhé.
    my_doctor_id = current_user.get('id_account') or current_user.get('id_doctor')

    q = text("""
        INSERT INTO patient (name_patient, date_of_birth_patient, gender, phone_number, id_doctor)
        VALUES (:name_patient, :date_of_birth_patient, :gender, :phone_number, :id_doctor)
        RETURNING id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at;
    """)
    
    params = payload.model_dump()
    params['id_doctor'] = my_doctor_id # Gán ID bác sĩ vào

    try:
        r = await db.execute(q, params)
        await db.commit()
        row = r.mappings().first()
        return row
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi khi thêm bệnh nhân: {str(e)}")

# =================================================================
# ✅ 2. GET LIST (Chỉ lấy bệnh nhân CỦA TÔI)
# =================================================================
@router.get("", response_model=List[PatientOut])
async def get_patients(
    keyword: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user) # 🔒 Bắt buộc đăng nhập
):
    my_doctor_id = current_user.get('id_account') or current_user.get('id_doctor')

    # 👇 CÂU QUERY QUAN TRỌNG: Thêm WHERE id_doctor
    sql_query = """
        SELECT id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at
        FROM patient
        WHERE id_doctor = :id_doctor 
    """
    
    params = {"id_doctor": my_doctor_id}

    # Logic tìm kiếm (giữ nguyên nhưng nối thêm vào WHERE)
    if keyword:
        search_term = f"%{keyword}%"
        params["search"] = search_term
        
        # Dùng AND ( ... OR ... ) để gom nhóm điều kiện tìm kiếm
        sql_query += """
            AND (
                name_patient ILIKE :search 
                OR phone_number ILIKE :search
                OR gender ILIKE :search
                OR CAST(EXTRACT(YEAR FROM date_of_birth_patient) AS TEXT) ILIKE :search
            )
        """

    sql_query += " ORDER BY id_patient ASC;"

    q = text(sql_query)
    r = await db.execute(q, params)
    rows = r.mappings().all()
    return rows

# =================================================================
# ✅ 3. GET ONE (Chặn xem trộm hồ sơ người khác)
# =================================================================
@router.get("/{id_patient}", response_model=PatientOut)
async def get_patient(
    id_patient: int, 
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    my_doctor_id = current_user.get('id_account') or current_user.get('id_doctor')

    q = text("""
        SELECT id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at
        FROM patient
        WHERE id_patient = :id_patient 
          AND id_doctor = :id_doctor  -- 🔒 CHỐT CHẶN BẢO MẬT
        LIMIT 1;
    """)
    
    r = await db.execute(q, {"id_patient": id_patient, "id_doctor": my_doctor_id})
    row = r.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân (hoặc bạn không có quyền xem).")

    return row

# =================================================================
# ✅ 4. UPDATE (Chỉ sửa được bệnh nhân của mình)
# =================================================================
@router.put("/{id_patient}", response_model=PatientOut)
async def update_patient(
    id_patient: int, 
    payload: PatientCreate, 
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    my_doctor_id = current_user.get('id_account') or current_user.get('id_doctor')

    q = text("""
        UPDATE patient
        SET name_patient = :name_patient,
            date_of_birth_patient = :date_of_birth_patient,
            gender = :gender,
            phone_number = :phone_number
        WHERE id_patient = :id_patient 
          AND id_doctor = :id_doctor -- 🔒 CHỈ SỬA CỦA MÌNH
        RETURNING id_patient, name_patient, date_of_birth_patient, gender, phone_number, created_at;
    """)
    
    params = payload.model_dump()
    params["id_patient"] = id_patient
    params["id_doctor"] = my_doctor_id

    r = await db.execute(q, params)
    await db.commit()
    row = r.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân để sửa.")

    return row

# =================================================================
# ✅ 5. DELETE (Chỉ xóa được bệnh nhân của mình)
# =================================================================
@router.delete("/{id_patient}")
async def delete_patient(
    id_patient: int, 
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    my_doctor_id = current_user.get('id_account') or current_user.get('id_doctor')

    try:
        q = text("""
            DELETE FROM patient 
            WHERE id_patient = :id_patient 
              AND id_doctor = :id_doctor -- 🔒 CHỈ XÓA CỦA MÌNH
            RETURNING id_patient;
        """)
        r = await db.execute(q, {"id_patient": id_patient, "id_doctor": my_doctor_id})
        await db.commit()

        deleted_id = r.scalar()
        
        if not deleted_id:
            raise HTTPException(status_code=404, detail="Không tìm thấy bệnh nhân để xóa.")

        return {"message": "Deleted successfully", "id_patient": deleted_id}

    except Exception as e:
        if "foreign key constraint" in str(e).lower():
            raise HTTPException(
                status_code=400, 
                detail="Không thể xóa vì bệnh nhân này đã có hồ sơ khám bệnh (History)."
            )
        raise e