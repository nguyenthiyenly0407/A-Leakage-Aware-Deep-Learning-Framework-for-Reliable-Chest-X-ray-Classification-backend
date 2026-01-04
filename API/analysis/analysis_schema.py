from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# ==========================================
# 1. BASE SCHEMA & CREATE (Dùng cho nội bộ hoặc API cũ)
# ==========================================
class AnalysisCreate(BaseModel):
    id_patient: int
    id_doctor: int
    image_url: str
    ai_prediction: str
    ai_confidence: Optional[float] = None
    doctor_diagnosis: Optional[str] = None
    note: Optional[str] = None

# ==========================================
# 2. OUTPUT SCHEMA (Dùng để trả về JSON cho Frontend)
# ==========================================
class AnalysisOut(AnalysisCreate):
    id_analysis: int
    created_at: datetime

    class Config:
        from_attributes = True  # Để Pydantic đọc được dữ liệu từ SQLAlchemy model

# ==========================================
# 3. UPDATE SCHEMA (MỚI - Dùng cho bác sĩ chốt kết quả)
# ==========================================
class DoctorDiagnosisUpdate(BaseModel):
    confirm_ai: bool                       # True: Đồng ý với AI (Yes) | False: Không đồng ý (No)
    custom_diagnosis: Optional[str] = None # Nếu chọn False (No), bác sĩ điền tên bệnh đúng vào đây
    note: Optional[str] = None             # Ghi chú thêm của bác sĩ