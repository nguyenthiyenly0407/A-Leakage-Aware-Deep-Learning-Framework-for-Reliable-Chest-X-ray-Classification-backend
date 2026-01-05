from datetime import datetime, timedelta
from typing import Optional
# 👇 BỔ SUNG CÁC DÒNG IMPORT NÀY
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import bcrypt

# ==========================================
# 1. CẤU HÌNH
# ==========================================
# ⚠️ LƯU Ý: Key này phải GIỐNG HỆT key bên file API_login.py
SECRET_KEY = "DA_LIEU_BI_MAT_2026" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# 👇 DÒNG NÀY QUAN TRỌNG ĐỂ SWAGGER UI HIỆN NÚT LOGIN
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ==========================================
# 2. HÀM MÃ HÓA PASSWORD (BCRYPT)
# ==========================================
def verify_password(plain_password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        password_hash.encode("utf-8")
    )

def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(
        plain_password.encode("utf-8"),
        bcrypt.gensalt(rounds=12)
    ).decode("utf-8")

# ==========================================
# 3. HÀM TẠO TOKEN
# ==========================================
def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ==========================================
# 4. HÀM XÁC THỰC USER (GET CURRENT USER)
# ==========================================
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Hàm này sẽ được gọi trong các API cần bảo mật (như thêm/sửa/xóa bệnh nhân).
    Nó tự động lấy token, giải mã và trả về thông tin user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials (Token không hợp lệ hoặc đã hết hạn)",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Giải mã token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Lấy thông tin id_account từ payload
        # (Lưu ý: Bên file Login lúc tạo token bạn phải lưu key là 'id_account' hoặc 'id_doctor')
        id_account: int = payload.get("id_account")
        
        if id_account is None:
            raise credentials_exception
            
        # Trả về toàn bộ payload (chứa id_account, role, name...)
        return payload
        
    except JWTError:
        raise credentials_exception