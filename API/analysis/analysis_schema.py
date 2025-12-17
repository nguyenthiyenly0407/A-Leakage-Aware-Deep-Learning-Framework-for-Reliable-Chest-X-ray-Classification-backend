from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class AnalysisCreate(BaseModel):
    id_patient: int
    id_doctor: int
    image_url: str
    ai_prediction: str
    ai_confidence: Optional[float] = None
    doctor_diagnosis: Optional[str] = None
    note: Optional[str] = None

class AnalysisOut(AnalysisCreate):
    id_analysis: int
    created_at: datetime
