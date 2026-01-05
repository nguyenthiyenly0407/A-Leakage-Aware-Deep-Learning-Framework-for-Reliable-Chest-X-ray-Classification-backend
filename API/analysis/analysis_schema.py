from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

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
class AnalysisOut(BaseModel):
    id_analysis: int
    id_patient: int
    id_doctor: int
    image_url: str
    heatmap_url: Optional[str] = None  # ✅ Chuẩn
    ai_prediction: str
    ai_confidence: float
    
    # 💡 Mẹo: Để là str thì Frontend phải tự parse. 
    # Nếu muốn xịn, bạn có thể định nghĩa kiểu List[Dict] nhưng cần xử lý ở API.
    # Hiện tại để Optional[str] là AN TOÀN NHẤT với code API hiện tại.
    top3_json: Optional[str] = None    # ✅ Chuẩn (Lưu ý: Frontend nhận về chuỗi JSON)
    
    doctor_diagnosis: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime

    # ⚠️ SỬA LỖI: Chỉ giữ 1 cái Config này thôi
    class Config:
        from_attributes = True

# ==========================================
# 3. UPDATE SCHEMA (MỚI - Dùng cho bác sĩ chốt kết quả)
# ==========================================
class DoctorDiagnosisUpdate(BaseModel):
    confirm_ai: bool                       # True: Đồng ý (Yes) | False: Không (No)
    custom_diagnosis: Optional[str] = None # Nếu No -> Bác sĩ nhập tên bệnh
    note: Optional[str] = None             # Ghi chú thêm