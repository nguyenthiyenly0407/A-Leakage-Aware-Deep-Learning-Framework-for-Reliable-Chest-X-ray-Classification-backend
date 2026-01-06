from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from API.auth.security import get_current_user
from Database.connection import get_db
import math
from datetime import date # <--- Nhớ import date

router = APIRouter(prefix="/history", tags=["History"])

@router.get("")
async def get_history(
    keyword: str = None,
    status: str = None, 
    start_date: date = None, # <--- Thêm tham số ngày bắt đầu
    end_date: date = None,   # <--- Thêm tham số ngày kết thúc
    page: int = 1,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    uid = current_user.get('id_account') or current_user.get('id_doctor')
    offset = (page - 1) * limit
    
    # Base query logic
    base_conditions = "sa.id_doctor = :uid"
    params = {"uid": uid}

    # 1. Lọc Keyword (đã fix lỗi integer cast text)
    if keyword:
        base_conditions += " AND (p.name_patient ILIKE :kw OR sa.ai_prediction ILIKE :kw OR CAST(p.id_patient AS TEXT) ILIKE :kw)"
        params["kw"] = f"%{keyword}%"

    # 2. Lọc Status
    if status and status != 'all':
        if status == 'correct':
            base_conditions += " AND sa.doctor_diagnosis = sa.ai_prediction"
        elif status == 'incorrect':
            base_conditions += " AND sa.doctor_diagnosis IS NOT NULL AND sa.doctor_diagnosis != sa.ai_prediction"
        elif status == 'pending':
            base_conditions += " AND sa.doctor_diagnosis IS NULL"

    # 3. Lọc Ngày (Mới thêm)
    if start_date:
        base_conditions += " AND DATE(sa.created_at) >= :start_date"
        params["start_date"] = start_date
    if end_date:
        base_conditions += " AND DATE(sa.created_at) <= :end_date"
        params["end_date"] = end_date

    # Query Total Count
    count_query = f"""
        SELECT COUNT(*) 
        FROM skin_analysis sa
        JOIN patient p ON sa.id_patient = p.id_patient
        WHERE {base_conditions}
    """
    total_result = await db.execute(text(count_query), params)
    total_records = total_result.scalar()

    # Query Data
    data_query = f"""
        SELECT 
            sa.id_analysis,
            sa.created_at,
            sa.ai_prediction,
            sa.ai_confidence,
            sa.doctor_diagnosis,
            p.name_patient,
            p.id_patient,
            CASE 
                WHEN sa.doctor_diagnosis IS NULL THEN 'pending'
                WHEN sa.doctor_diagnosis = sa.ai_prediction THEN 'correct'
                ELSE 'incorrect'
            END as status
        FROM skin_analysis sa
        JOIN patient p ON sa.id_patient = p.id_patient
        WHERE {base_conditions}
        ORDER BY sa.created_at DESC 
        LIMIT :limit OFFSET :offset
    """
    
    params["limit"] = limit
    params["offset"] = offset
    
    result = await db.execute(text(data_query), params)
    rows = result.mappings().all()

    return {
        "data": rows,
        "pagination": {
            "page": page,
            "limit": limit,
            "total_records": total_records,
            "total_pages": math.ceil(total_records / limit) if limit > 0 else 0
        }
    }