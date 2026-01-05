import os
import io
import torch
import torch.nn as nn
import numpy as np
import base64
import json
import uuid
import timm
import cv2
from dotenv import load_dotenv
# --- Library Imports ---
from PIL import Image
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse # Import quan trọng để trả về ảnh
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from torchvision import models, transforms
from torch.nn import functional as F

# --- GradCAM & Crypto ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from cryptography.fernet import Fernet

# --- Local Imports ---
from .analysis_schema import AnalysisCreate, AnalysisOut, DoctorDiagnosisUpdate
from Database.connection import get_db

# ==========================================
# 1. CẤU HÌNH & CONSTANTS
# ==========================================
load_dotenv()
MODEL_PATH = "Model/Ensemble_Hybrid_ResNet.pth"
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Acne and Rosacea Photos', 'Atopic Dermatitis Photos', 'Eczema Photos',
    'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives', 'Warts Molluscum and other Viral Infections'
]


key_str = os.getenv("SECRET_KEY")

if not key_str:

    raise ValueError("❌ Lỗi: Chưa cấu hình SECRET_KEY trong file .env!")


SECRET_KEY = key_str.encode() 

try:
    cipher_suite = Fernet(SECRET_KEY)
    print("✅ [Crypto] Key mã hóa đã được tải thành công!")
except Exception as e:
    raise ValueError(f"❌ Key trong .env bị sai định dạng! Lỗi: {e}")
class HybridViTModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridViTModel, self).__init__()
        self.cnn = models.efficientnet_b0(weights=None)
        self.cnn.classifier = nn.Identity()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        self.vit.head = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(torch.cat((self.cnn(x), self.vit(x)), dim=1))

def build_resnet50(num_classes):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, num_classes))
    return m

class EnsemblePacked(nn.Module):
    def __init__(self, hybrid, resnet, weight_h=0.6, weight_r=0.4):
        super().__init__()
        self.hybrid = hybrid
        self.resnet = resnet
        self.weight_h = weight_h
        self.weight_r = weight_r
    def forward(self, x):
        return self.weight_h * self.hybrid(x) + self.weight_r * self.resnet(x)

# Biến toàn cục model
ai_model = None

# ==========================================
# 3. HELPER FUNCTIONS (Xử lý ảnh & Mã hóa)
# ==========================================

def encrypt_image_from_bytes(image_bytes: bytes) -> bytes:
    """Mã hóa bytes ảnh trước khi lưu DB"""
    return cipher_suite.encrypt(image_bytes)

def decrypt_image_to_bytes(encrypted_data: bytes) -> bytes:
    """Giải mã bytes từ DB để hiển thị"""
    return cipher_suite.decrypt(encrypted_data)

def generate_heatmap_bytes(input_tensor, img_float):
    """
    Tạo heatmap trên RAM và trả về BYTES (không lưu file).
    """
    try:
        # Target layer
        target_h = [ai_model.hybrid.cnn.features[-1]]
        target_r = [ai_model.resnet.layer4[-1]]
        
        cam_h = GradCAM(model=ai_model.hybrid, target_layers=target_h)
        cam_r = GradCAM(model=ai_model.resnet, target_layers=target_r)

        gray_h = cam_h(input_tensor=input_tensor)[0]
        gray_r = cam_r(input_tensor=input_tensor)[0]

        # Kết hợp heatmap
        combined = (0.6 * gray_h) + (0.4 * gray_r)
        
        # Tạo ảnh visualization
        vis = show_cam_on_image(img_float, combined, use_rgb=True)
        img_pil = Image.fromarray(vis)
        
        # --- Lưu vào RAM (BytesIO) ---
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=90)
        return buf.getvalue() # Trả về bytes
        
    except Exception as e:
        print(f"⚠️ Lỗi GradCAM: {e}")
        return None

# ==========================================
# 4. ROUTER SETUP & LOAD MODEL
# ==========================================
router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.on_event("startup")
async def load_ai_model():
    global ai_model
    try:
        print(f"🏥 [AI] Đang load model trên {device}...")
        hybrid = HybridViTModel(len(CLASS_NAMES))
        resnet = build_resnet50(len(CLASS_NAMES))
        ai_model = EnsemblePacked(hybrid, resnet).to(device)
        
        # Load weights
        state = torch.load(MODEL_PATH, map_location=device)
        ai_model.load_state_dict(state)
        ai_model.eval()
        print("✅ [AI] Model loaded successfully!")
    except Exception as e:
        print(f"❌ [AI] Load failed: {e}")

# ==========================================
# 5. API CHÍNH: DETECT (Đã sửa đổi để lưu Blob Mã Hóa)
# ==========================================
@router.post("/detect", response_model=AnalysisOut) 
async def detect_and_save_analysis(
    id_patient: int = Form(...),
    id_doctor: int = Form(...),
    note: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if ai_model is None:
        raise HTTPException(status_code=503, detail="AI Model chưa sẵn sàng")

    # --- B1: ĐỌC VÀ MÃ HÓA ẢNH GỐC ---
    contents = await file.read()
    
    # Mã hóa ngay lập tức
    encrypted_original = encrypt_image_from_bytes(contents)
    
    # Mở ảnh bằng PIL để AI xử lý (Load vào RAM)
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # --- B2: TIỀN XỬ LÝ (PRE-PROCESS) ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_float = np.float32(np.array(img_resized)) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(device)

    # --- B3: DỰ ĐOÁN (AI PREDICT) ---
    with torch.no_grad():
        outputs = ai_model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3)

    top1_conf = float(top3_prob[0][0].item())
    top1_idx = top3_idx[0][0].item()
    ai_pred = CLASS_NAMES[top1_idx]
    
    # Cảnh báo độ tin cậy thấp
    CONFIDENCE_THRESHOLD = 0.70
    if top1_conf < CONFIDENCE_THRESHOLD:
        note_warning = (
            f"[Cảnh báo AI] Độ tin cậy thấp ({top1_conf*100:.2f}%). "
            f"Khả năng cao là DA KHỎE hoặc chưa rõ ràng."
        )
        note = f"{note} | {note_warning}" if note else note_warning

    # Top-3 JSON
    top3_list = []
    for i in range(3):
        top3_list.append({
            "disease": CLASS_NAMES[top3_idx[0][i].item()],
            "confidence": f"{top3_prob[0][i].item() * 100:.2f}%"
        })
    top3_json_str = json.dumps(top3_list)

    # --- B3.1: TẠO HEATMAP & MÃ HÓA ---
    heatmap_bytes = generate_heatmap_bytes(input_tensor, img_float)
    encrypted_heatmap = None
    if heatmap_bytes:
        encrypted_heatmap = encrypt_image_from_bytes(heatmap_bytes)

    # --- B4: LƯU DATABASE (LƯU BINARY) ---
    # Chú ý: Cột image_data và heatmap_data phải là kiểu BYTEA (Postgres) hoặc BLOB (MySQL)
    q = text("""
        INSERT INTO skin_analysis
        (id_patient, id_doctor, image_data, heatmap_data, ai_prediction, ai_confidence, top3_json, doctor_diagnosis, note)
        VALUES
        (:id_patient, :id_doctor, :image_data, :heatmap_data, :ai_prediction, :ai_confidence, :top3_json, :doctor_diagnosis, :note)
        RETURNING id_analysis, id_patient, id_doctor, ai_prediction, ai_confidence, top3_json, doctor_diagnosis, note, created_at;
    """)
    
    # Không cần trả về cột data blob trong response JSON vì nó rất nặng
    params = {
        "id_patient": id_patient,
        "id_doctor": id_doctor,
        "image_data": encrypted_original,
        "heatmap_data": encrypted_heatmap,
        "ai_prediction": ai_pred,
        "ai_confidence": top1_conf * 100,
        "top3_json": top3_json_str,
        "doctor_diagnosis": None,
        "note": note
    }
    
    try:
        r = await db.execute(q, params)
        await db.commit()
        record = r.mappings().first()
        
        # Convert record to dict để thêm url ảo cho frontend
        result_dict = dict(record)
        # Tạo URL ảo để frontend gọi API lấy ảnh (Thay vì trả về URL tĩnh)
        result_dict["image_url"] = f"/analysis/image/{record.id_analysis}/original"
        result_dict["heatmap_url"] = f"/analysis/image/{record.id_analysis}/heatmap"
        
        return result_dict
        
    except Exception as e:
        await db.rollback()
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

# ==========================================
# 6. API MỚI: LẤY ẢNH TỪ DB (GIẢI MÃ & STREAM)
# ==========================================
@router.get("/image/{id_analysis}/{img_type}")
async def get_secure_image(id_analysis: int, img_type: str, db: AsyncSession = Depends(get_db)):
    """
    img_type: 'original' hoặc 'heatmap'
    API này sẽ query DB, giải mã và trả về stream ảnh.
    """
    if img_type not in ['original', 'heatmap']:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Chỉ select đúng cột cần thiết để tối ưu
    col_name = "image_data" if img_type == "original" else "heatmap_data"
    q = text(f"SELECT {col_name} FROM skin_analysis WHERE id_analysis = :id")
    
    result = await db.execute(q, {"id": id_analysis})
    record = result.mappings().first()
    
    if not record or not record[col_name]:
        # Trả về ảnh mặc định hoặc 404 nếu không có ảnh
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        # 1. Lấy dữ liệu mã hóa
        encrypted_bytes = record[col_name]
        
        # 2. Giải mã
        decrypted_bytes = decrypt_image_to_bytes(encrypted_bytes)
        
        # 3. Trả về dưới dạng file ảnh (Stream)
        return StreamingResponse(io.BytesIO(decrypted_bytes), media_type="image/jpeg")
    except Exception as e:
        print(f"Decryption Error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi giải mã ảnh.")

# ==========================================
# 7. CÁC API KHÁC (Tối ưu Query - Không select BLOB)
# ==========================================

# Helper để không lặp code Select
def get_select_query_without_blob():
    """Chỉ lấy thông tin text, không lấy cột dữ liệu ảnh nặng"""
    return """
        SELECT id_analysis, id_patient, id_doctor, ai_prediction, ai_confidence, 
               top3_json, doctor_diagnosis, note, created_at
        FROM skin_analysis 
    """

def process_records_with_urls(records):
    """Chèn URL ảo vào kết quả trả về"""
    results = []
    for row in records:
        r = dict(row)
        r["image_url"] = f"/analysis/image/{r['id_analysis']}/original"
        r["heatmap_url"] = f"/analysis/image/{r['id_analysis']}/heatmap"
        results.append(r)
    return results

@router.get("", response_model=List[AnalysisOut])
async def get_all_analysis(db: AsyncSession = Depends(get_db)):
    # Bỏ SELECT * để tránh lag DB vì load ảnh nặng
    q = text(get_select_query_without_blob() + " ORDER BY id_analysis DESC;")
    r = await db.execute(q)
    return process_records_with_urls(r.mappings().all())

@router.get("/patient/{id_patient}", response_model=List[AnalysisOut])
async def get_analysis_by_patient(id_patient: int, db: AsyncSession = Depends(get_db)):
    q = text(get_select_query_without_blob() + " WHERE id_patient = :id_patient ORDER BY id_analysis DESC;")
    r = await db.execute(q, {"id_patient": id_patient})
    return process_records_with_urls(r.mappings().all())

@router.get("/{id_analysis}", response_model=AnalysisOut)
async def get_one_analysis(id_analysis: int, db: AsyncSession = Depends(get_db)):
    q = text(get_select_query_without_blob() + " WHERE id_analysis = :id_analysis LIMIT 1;")
    r = await db.execute(q, {"id_analysis": id_analysis})
    row = r.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Process single record
    res = dict(row)
    res["image_url"] = f"/analysis/image/{res['id_analysis']}/original"
    res["heatmap_url"] = f"/analysis/image/{res['id_analysis']}/heatmap"
    return res

@router.put("/{id_analysis}/diagnose", response_model=AnalysisOut)
async def doctor_confirm_analysis(
    id_analysis: int, 
    payload: DoctorDiagnosisUpdate, 
    db: AsyncSession = Depends(get_db)
):
    # Lấy thông tin cũ (không cần load ảnh)
    q_select = text("SELECT ai_prediction FROM skin_analysis WHERE id_analysis = :id_analysis")
    r = await db.execute(q_select, {"id_analysis": id_analysis})
    record = r.mappings().first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ.")

    final_diagnosis = record['ai_prediction'] if payload.confirm_ai else payload.custom_diagnosis
    if not payload.confirm_ai and not payload.custom_diagnosis:
         raise HTTPException(status_code=400, detail="Vui lòng nhập chẩn đoán mới.")

    # Update
    q_update = text("""
        UPDATE skin_analysis
        SET doctor_diagnosis = :diagnosis,
            note = :note 
        WHERE id_analysis = :id_analysis
        RETURNING id_analysis, id_patient, id_doctor, ai_prediction, ai_confidence, top3_json, doctor_diagnosis, note, created_at;
    """)
    
    try:
        r_update = await db.execute(q_update, {
            "diagnosis": final_diagnosis,
            "note": payload.note,
            "id_analysis": id_analysis
        })
        await db.commit()
        
        updated_row = r_update.mappings().first()
        
        # Format lại response
        res = dict(updated_row)
        res["image_url"] = f"/analysis/image/{res['id_analysis']}/original"
        res["heatmap_url"] = f"/analysis/image/{res['id_analysis']}/heatmap"
        return res
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi Database: {str(e)}")