import os
import io
import torch
import torch.nn as nn
import numpy as np
import base64
import uuid
import timm
import cv2
from PIL import Image
from typing import List, Optional

# FastAPI & DB Imports
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from torchvision import models, transforms
from torch.nn import functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from .analysis_schema import AnalysisCreate, AnalysisOut, DoctorDiagnosisUpdate
# Local Imports (Giả sử file connection của bạn ở đây)
from Database.connection import get_db


# ==========================================
# 1. CẤU HÌNH AI & MODEL
# ==========================================
MODEL_PATH = "Model/Ensemble_Hybrid_ResNet.pth" # Sửa đường dẫn nếu cần
IMG_SIZE = 224
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True) # Tự tạo folder nếu chưa có

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Acne and Rosacea Photos', 'Atopic Dermatitis Photos', 'Eczema Photos',
    'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives', 'Warts Molluscum and other Viral Infections'
]

# --- Định nghĩa kiến trúc Model (Giữ nguyên như cũ) ---
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

# Biến toàn cục chứa model
ai_model = None

# ==========================================
# 2. KHỞI TẠO ROUTER
# ==========================================
router = APIRouter(prefix="/analysis", tags=["analysis"])

# Load model khi server khởi động
@router.on_event("startup")
async def load_ai_model():
    global ai_model
    try:
        print(f"🏥 [AI] Đang load model trên {device}...")
        hybrid = HybridViTModel(len(CLASS_NAMES))
        resnet = build_resnet50(len(CLASS_NAMES))
        ai_model = EnsemblePacked(hybrid, resnet).to(device)
        
        # Load weights (Map location cpu để tránh lỗi nếu server ko có GPU)
        state = torch.load(MODEL_PATH, map_location=device)
        ai_model.load_state_dict(state)
        ai_model.eval()
        print("✅ [AI] Model loaded successfully!")
    except Exception as e:
        print(f"❌ [AI] Load failed: {e}")

# ==========================================
# 3. HÀM HỖ TRỢ (XỬ LÝ ẢNH & GRAD-CAM)
# ==========================================
def get_gradcam_base64(input_tensor, rgb_img_float):
    try:
        # Nhắm vào layer cuối của cả 2 model
        target_h = [ai_model.hybrid.cnn.features[-1]]
        target_r = [ai_model.resnet.layer4[-1]]
        
        cam_h = GradCAM(model=ai_model.hybrid, target_layers=target_h)
        cam_r = GradCAM(model=ai_model.resnet, target_layers=target_r)

        gray_h = cam_h(input_tensor=input_tensor)[0]
        gray_r = cam_r(input_tensor=input_tensor)[0]

        # Fusion
        combined = (0.6 * gray_h) + (0.4 * gray_r)
        vis = show_cam_on_image(rgb_img_float, combined, use_rgb=True)
        
        # Convert to Base64
        img_pil = Image.fromarray(vis)
        buff = io.BytesIO()
        img_pil.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"GradCAM Error: {e}")
        return None

# ==========================================
# 4. API ENDPOINTS
# ==========================================

# --- API MỚI: Upload ảnh -> Chẩn đoán -> Lưu DB ---
@router.post("/detect", response_model=AnalysisOut) 
async def detect_and_save_analysis(
    id_patient: int = Form(...),
    id_doctor: int = Form(...),
    note: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    API này làm 3 việc:
    1. Upload ảnh & Lưu vào folder static/uploads
    2. Chạy AI để lấy kết quả (Prediction)
    3. Lưu thông tin vào Database PostgreSQL
    """
    if ai_model is None:
        raise HTTPException(status_code=503, detail="AI Model chưa sẵn sàng")

    # 1. Đọc và Lưu ảnh
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Tạo tên file unique để lưu đĩa (tránh trùng tên)
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)
    image.save(file_path)
    
    # Giả lập URL (Trong thực tế bạn sẽ up lên S3 và lấy link S3)
    image_url_db = f"/static/uploads/{filename}"

    # 2. Xử lý AI
    # Preprocess cho Model
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Preprocess cho GradCAM (Float 0-1)
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_float = np.float32(np.array(img_resized)) / 255.0
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        outputs = ai_model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.topk(probs, 1)

    ai_pred = CLASS_NAMES[top_idx.item()]
    ai_conf = float(top_prob.item()) * 100 # Đổi sang %

    # Tạo Heatmap (Optional: Bạn có thể trả về cho Client hiển thị luôn, nhưng DB ko lưu cái này)
    # heatmap_b64 = get_gradcam_base64(input_tensor, img_float)

    # 3. Lưu vào Database (PostgreSQL)
    q = text("""
        INSERT INTO skin_analysis
        (id_patient, id_doctor, image_url, ai_prediction, ai_confidence, doctor_diagnosis, note)
        VALUES
        (:id_patient, :id_doctor, :image_url, :ai_prediction, :ai_confidence, :doctor_diagnosis, :note)
        RETURNING id_analysis, id_patient, id_doctor, image_url, ai_prediction, ai_confidence, doctor_diagnosis, note, created_at;
    """)
    
    # Chuẩn bị dữ liệu insert
    params = {
        "id_patient": id_patient,
        "id_doctor": id_doctor,
        "image_url": image_url_db,     # Link ảnh vừa lưu
        "ai_prediction": ai_pred,      # Kết quả AI
        "ai_confidence": ai_conf,      # Độ tin cậy AI
        "doctor_diagnosis": None,      # Mới detect thì bác sĩ chưa khám
        "note": note
    }
    
    try:
        r = await db.execute(q, params)
        await db.commit()
        record = r.mappings().first()
        
        # Mẹo: Trả về record DB chuẩn
        return record
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


# --- API CŨ: Lấy tất cả (Giữ nguyên) ---
@router.get("", response_model=List[AnalysisOut])
async def get_all_analysis(db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_analysis, id_patient, id_doctor, image_url, ai_prediction, ai_confidence,
               doctor_diagnosis, note, created_at
        FROM skin_analysis
        ORDER BY id_analysis DESC;
    """)
    r = await db.execute(q)
    return r.mappings().all()

# --- API CŨ: Lấy theo ID Patient (Giữ nguyên) ---
@router.get("/patient/{id_patient}", response_model=List[AnalysisOut])
async def get_analysis_by_patient(id_patient: int, db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_analysis, id_patient, id_doctor, image_url, ai_prediction, ai_confidence,
               doctor_diagnosis, note, created_at
        FROM skin_analysis
        WHERE id_patient = :id_patient
        ORDER BY id_analysis DESC;
    """)
    r = await db.execute(q, {"id_patient": id_patient})
    return r.mappings().all()

# --- API CŨ: Lấy 1 cái (Giữ nguyên) ---
@router.get("/{id_analysis}", response_model=AnalysisOut)
async def get_one_analysis(id_analysis: int, db: AsyncSession = Depends(get_db)):
    q = text("""
        SELECT id_analysis, id_patient, id_doctor, image_url, ai_prediction, ai_confidence,
               doctor_diagnosis, note, created_at
        FROM skin_analysis
        WHERE id_analysis = :id_analysis
        LIMIT 1;
    """)
    r = await db.execute(q, {"id_analysis": id_analysis})
    row = r.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return row

# --- API MỚI: Bác sĩ cập nhật chẩn đoán (Confirm hoặc Sửa) ---
# --- API MỚI: Bác sĩ cập nhật chẩn đoán (Confirm hoặc Sửa) ---
@router.put("/{id_analysis}/diagnose", response_model=AnalysisOut)
async def doctor_confirm_analysis(
    id_analysis: int, 
    payload: DoctorDiagnosisUpdate, 
    db: AsyncSession = Depends(get_db)
):
    # 1. Tìm bản ghi
    q_select = text("SELECT * FROM skin_analysis WHERE id_analysis = :id_analysis")
    r = await db.execute(q_select, {"id_analysis": id_analysis})
    record = r.mappings().first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ này.")

    # 2. Xử lý Logic Yes/No
    final_diagnosis = ""
    
    if payload.confirm_ai:
        # Nếu đồng ý -> Lấy kết quả AI
        final_diagnosis = record['ai_prediction']
    else:
        # Nếu từ chối -> Lấy kết quả bác sĩ nhập
        if not payload.custom_diagnosis:
            raise HTTPException(status_code=400, detail="Vui lòng nhập chẩn đoán mới.")
        final_diagnosis = payload.custom_diagnosis

    # 3. Cập nhật vào Database
    q_update = text("""
        UPDATE skin_analysis
        SET doctor_diagnosis = :diagnosis,
            note = COALESCE(:note, note) 
        WHERE id_analysis = :id_analysis
        RETURNING *;
    """)
    
    try:
        r_update = await db.execute(q_update, {
            "diagnosis": final_diagnosis,
            "note": payload.note,          # <--- SỬA LẠI CHỖ NÀY: dùng payload.note
            "id_analysis": id_analysis
        })
        await db.commit()
        updated_record = r_update.mappings().first()
        return updated_record
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Lỗi Database: {str(e)}")