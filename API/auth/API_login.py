from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from Database.connection import get_db
from .auth import LoginRequest, TokenResponse
from .security import verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/doctor/login", response_model=TokenResponse)
async def doctor_login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_doctor, email_doctor, password_hash, is_active
        FROM login_doctor
        WHERE email_doctor = :email
        LIMIT 1
    """)
    r = await db.execute(q, {"email": payload.email})
    row = r.mappings().first()

    if not row or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if row["is_active"] is False:
        raise HTTPException(status_code=403, detail="Account is inactive")

    token = create_access_token({
        "sub": row["email_doctor"],
        "role": "doctor",
        "doctor_id": row["id_doctor"],
    })
    return {"access_token": token, "token_type": "bearer"}


@router.post("/admin/login", response_model=TokenResponse)
async def admin_login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_admin, email_admin, password_hash, is_active
        FROM login_admin
        WHERE email_admin = :email
        LIMIT 1
    """)
    r = await db.execute(q, {"email": payload.email})
    row = r.mappings().first()

    if not row or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if row["is_active"] is False:
        raise HTTPException(status_code=403, detail="Account is inactive")

    token = create_access_token({
        "sub": row["email_admin"],
        "role": "admin",
        "admin_id": row["id_admin"],
    })
    return {"access_token": token, "token_type": "bearer"}
