from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from API.auth.security import get_current_user
from Database.connection import get_db

router = APIRouter(prefix="/stats", tags=["Statistics"])

@router.get("/chart-data")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    my_doctor_id = current_user.get('id_account') or current_user.get('id_doctor')

    # =========================================================
    # 1. FIX LỖI ĐẾM BỆNH NHÂN (Tách ra từng query cho chắc ăn)
    # =========================================================
    
    # Query 1: Đếm TẤT CẢ bệnh nhân đang quản lý (Để sửa lỗi hiển thị số 1)
    q_total = text("SELECT COUNT(*) FROM patient WHERE id_doctor = :uid")
    total_all = await db.scalar(q_total, {"uid": my_doctor_id})

    # Query 2: Đếm bệnh nhân MỚI trong tháng này (Để tính %)
    q_this_month = text("""
        SELECT COUNT(*) FROM patient 
        WHERE id_doctor = :uid 
        AND date_trunc('month', created_at) = date_trunc('month', CURRENT_DATE)
    """)
    this_month = await db.scalar(q_this_month, {"uid": my_doctor_id}) or 0

    # Query 3: Đếm bệnh nhân tháng trước
    q_last_month = text("""
        SELECT COUNT(*) FROM patient 
        WHERE id_doctor = :uid 
        AND date_trunc('month', created_at) = date_trunc('month', CURRENT_DATE - INTERVAL '1 month')
    """)
    last_month = await db.scalar(q_last_month, {"uid": my_doctor_id}) or 0

    # Tính % Tăng trưởng
    growth_txt = "+0%"
    if last_month > 0:
        percent = ((this_month - last_month) / last_month) * 100
        growth_txt = f"{percent:+.1f}%"
    elif this_month > 0:
        growth_txt = "+100%"

    # =========================================================
    # 2. ĐẾM SỐ CA AI ĐÚNG / SAI
    # =========================================================
    q_ai = text("""
        SELECT 
            COUNT(*) FILTER (WHERE doctor_diagnosis = ai_prediction) as correct,
            COUNT(*) FILTER (WHERE doctor_diagnosis IS NOT NULL AND doctor_diagnosis != ai_prediction) as incorrect
        FROM skin_analysis
        WHERE id_doctor = :uid
    """)
    res_ai = (await db.execute(q_ai, {"uid": my_doctor_id})).mappings().first()

    # =========================================================
    # 3. PIE CHART (Top bệnh)
    # =========================================================
    q_disease = text("""
        SELECT COALESCE(doctor_diagnosis, ai_prediction) as name, COUNT(*) as value
        FROM skin_analysis
        WHERE id_doctor = :uid
        GROUP BY name
        ORDER BY value DESC
        LIMIT 5
    """)
    res_disease = (await db.execute(q_disease, {"uid": my_doctor_id})).mappings().all()

    # =========================================================
    # 4. AREA CHART (7 Ngày - Full Width)
    # =========================================================
    q_traffic = text("""
        SELECT 
            TO_CHAR(day_series, 'DD/MM') as day,
            COUNT(sa.id_analysis) as visits
        FROM generate_series(CURRENT_DATE - INTERVAL '6 days', CURRENT_DATE, '1 day') as day_series
        LEFT JOIN skin_analysis sa 
            ON DATE(sa.created_at) = DATE(day_series) 
            AND sa.id_doctor = :uid
        GROUP BY day_series
        ORDER BY day_series ASC
    """)
    res_traffic = (await db.execute(q_traffic, {"uid": my_doctor_id})).mappings().all()

    # =========================================================
    # 5. RECENT LIST (Hoạt động gần đây)
    # =========================================================
    q_recent = text("""
        SELECT 
            p.id_patient as id,
            p.name_patient as name,
            COALESCE(sa.doctor_diagnosis, sa.ai_prediction) as disease,
            CASE 
                WHEN sa.doctor_diagnosis IS NULL THEN 'Chưa xác nhận'
                WHEN sa.doctor_diagnosis = sa.ai_prediction THEN 'AI Đúng'
                ELSE 'AI Sai'
            END as status,
            TO_CHAR(sa.created_at, 'HH24:MI DD/MM') as time
        FROM skin_analysis sa
        JOIN patient p ON sa.id_patient = p.id_patient
        WHERE sa.id_doctor = :uid
        ORDER BY sa.created_at DESC
        LIMIT 4
    """)
    res_recent = (await db.execute(q_recent, {"uid": my_doctor_id})).mappings().all()

    return {
        "stats": {
            "totalPatients": total_all,  # Giờ chắc chắn là tổng số
            "growth": growth_txt,
            "aiCorrect": res_ai['correct'],
            "aiIncorrect": res_ai['incorrect']
        },
        "diseaseData": [dict(row) for row in res_disease],
        "visitData": [dict(row) for row in res_traffic],
        "recentPatients": [dict(row) for row in res_recent]
    }