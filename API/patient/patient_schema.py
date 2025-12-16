from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional

class PatientCreate(BaseModel):
    name_patient: str
    date_of_birth_patient: date
    gender: Optional[str] = None
    phone_number: Optional[str] = None

class PatientOut(BaseModel):
    id_patient: int
    name_patient: str
    date_of_birth_patient: date
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    created_at: datetime
