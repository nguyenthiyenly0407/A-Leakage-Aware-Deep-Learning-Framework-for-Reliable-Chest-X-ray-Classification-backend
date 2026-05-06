from sqlalchemy import Column, Integer, String, Text, DateTime, Date, ForeignKey, Float, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship
from datetime import datetime

class Base(DeclarativeBase):
    pass

# 1. BẢNG DOCTOR (Dựa trên SQL của mày)
class Doctor(Base):
    __tablename__ = "login_doctor"
    id_doctor = Column(Integer, primary_key=True, index=True)
    name_doctor = Column(String(100), nullable=False)
    date_of_birth_doctor = Column(Date, nullable=False)
    email_doctor = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    gender = Column(String(10))
    phone_number = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# 2. BẢNG ADMIN (Dựa trên SQL của mày)
class Admin(Base):
    __tablename__ = "login_admin"
    id_admin = Column(Integer, primary_key=True, index=True)
    name_admin = Column(String(100), nullable=False)
    email_admin = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# 3. BẢNG PATIENT (Dựa trên Pydantic PatientCreate/Out)
class Patient(Base):
    __tablename__ = "patients"
    id_patient = Column(Integer, primary_key=True, index=True)
    id_doctor = Column(Integer, ForeignKey("login_doctor.id_doctor"))
    name_patient = Column(String(100), nullable=False)
    date_of_birth_patient = Column(Date, nullable=False)
    gender = Column(String(10))
    phone_number = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analyses = relationship("Analysis", back_populates="patient")

# 4. BẢNG ANALYSIS (Dựa trên Pydantic AnalysisCreate/Out)
class Analysis(Base):
    __tablename__ = "analysis"
    id_analysis = Column(Integer, primary_key=True, index=True)
    id_patient = Column(Integer, ForeignKey("patients.id_patient"))
    id_doctor = Column(Integer, ForeignKey("login_doctor.id_doctor"))
    image_url = Column(String(255), nullable=False)
    heatmap_url = Column(String(255))
    ai_prediction = Column(String(100), nullable=False)
    ai_confidence = Column(Float)
    top3_json = Column(Text) # Lưu chuỗi JSON kết quả top 3
    doctor_diagnosis = Column(String(100))
    note = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="analyses")