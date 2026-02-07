import os
import io
import json
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from typing import List, Optional, Tuple
import cv2
from PIL import Image, ImageOps
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torch.nn import functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAMPlusPlus
from cryptography.fernet import Fernet
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from .analysis_schema import AnalysisOut, DoctorDiagnosisUpdate
from Database.connection import get_db
import torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image, ImageOps
import io

load_dotenv()


# =========================================================
# 1. CONFIG
# =========================================================
CLS_MODEL_PATH = "Model/Best_efficientnet.pth"
SEG_MODEL_PATH = "Model/Best_unet_lung_seg.pth"

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

HEATMAP_IMAGE_WEIGHT = 0.0

HEATMAP_PREVIEW_SIZE = 224
HEATMAP_PREVIEW_BG = (0, 0, 0)
HEATMAP_DEFAULT_PREVIEW = True

CONFIDENCE_THRESHOLD = 0.70
MASK_THRESHOLD = 0.50


# Crypto key
key_str = os.getenv("SECRET_KEY")
if not key_str:
    raise ValueError("Missing SECRET_KEY in .env")

try:
    cipher_suite = Fernet(key_str.encode())
    print("Crypto key loaded.")
except Exception as e:
    raise ValueError(f"Invalid Fernet key in .env: {e}")


# =========================================================
# 2. MODELS
# =========================================================
def build_efficientnet(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(ch, feat))
            ch = feat

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat * 2, feat))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)


ai_cls_model: Optional[nn.Module] = None
ai_seg_model: Optional[nn.Module] = None


# =========================================================
# 3. PREPROCESS
# =========================================================
cls_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

seg_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =========================================================
# 4. UTILS
# =========================================================
def encrypt_image_from_bytes(image_bytes: bytes) -> bytes:
    return cipher_suite.encrypt(image_bytes)

def decrypt_image_to_bytes(encrypted_data: bytes) -> bytes:
    return cipher_suite.decrypt(encrypted_data)

def pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def letterbox_to_square(pil_img: Image.Image, size: int = 224, bg=(0, 0, 0)) -> Image.Image:
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = pil_img.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (size, size), bg)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), bg)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas

def safe_json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)

async def assert_foreign_keys_exist(db: AsyncSession, id_patient: int, id_doctor: int) -> None:
    q_patient = text("SELECT 1 FROM patient WHERE id_patient = :id_patient LIMIT 1;")
    q_doctor = text("SELECT 1 FROM login_doctor WHERE id_doctor = :id_doctor LIMIT 1;")

    rp = await db.execute(q_patient, {"id_patient": id_patient})
    rd = await db.execute(q_doctor, {"id_doctor": id_doctor})

    if rp.first() is None:
        raise HTTPException(status_code=400, detail=f"id_patient={id_patient} not found in patient table.")
    if rd.first() is None:
        raise HTTPException(status_code=400, detail=f"id_doctor={id_doctor} not found in doctor table.")


def compute_mask_guided_heatmap_bytes(
    input_tensor_cls: torch.Tensor,
    input_tensor_seg: torch.Tensor,
    original_pil: Image.Image,
    target_category: int,
    is_normal: bool = False 
) -> Optional[bytes]:
    try:
        if ai_cls_model is None or ai_seg_model is None:
            return None

        # 1. Chuẩn bị ảnh gốc (RGB, float 0-1)
        original_pil = ImageOps.exif_transpose(original_pil).convert("RGB")
        orig_w, orig_h = original_pil.size
        img_float = np.float32(np.array(original_pil)) / 255.0

        # 2. Luôn chạy Segmentation để lấy Mask phổi
        ai_seg_model.eval()
        with torch.no_grad():
            seg_logits = ai_seg_model(input_tensor_seg)
            seg_prob = torch.sigmoid(seg_logits)
            mask_224 = (seg_prob > MASK_THRESHOLD).float()
        
        # Resize mask về kích thước ảnh gốc
        mask_resized = F.interpolate(mask_224, size=(orig_h, orig_w), mode="nearest")
        mask_np = mask_resized.squeeze().cpu().numpy()
        
        # Tạo mask boolean cho vùng phổi
        lung_mask_bool = mask_np > 0.5

        # =========================================================
        # BƯỚC A: TẠO NỀN XANH DƯƠNG (Áp dụng cho cả Normal và Có bệnh)
        # =========================================================
        # Copy ảnh gốc để tô màu nền
        blue_tint_img = img_float.copy()

        # Công thức pha màu Xanh Dương (Cool Blue Tint)
        # R giảm 60%, G giảm 60%, B tăng (giữ 60% gốc + 40% xanh)
        blue_tint_img[lung_mask_bool, 0] *= 0.4  # Red
        blue_tint_img[lung_mask_bool, 1] *= 0.4  # Green
        blue_tint_img[lung_mask_bool, 2] = blue_tint_img[lung_mask_bool, 2] * 0.6 + 0.4 # Blue

        # Kẹp giá trị 0-1
        blue_tint_img = np.clip(blue_tint_img, 0, 1)

        # =========================================================
        # TRƯỜNG HỢP 1: NORMAL -> Chỉ trả về ảnh nền xanh
        # =========================================================
        if is_normal:
            final_heatmap_rgb = blue_tint_img
        
        # =========================================================
        # TRƯỜNG HỢP 2: CÓ BỆNH -> CHỒNG HEATMAP ĐỎ LÊN NỀN XANH
        # =========================================================
        else:
            # --- BƯỚC B: Tính toán GradCAM++ (Đỏ/Vàng) ---
            ai_cls_model.eval()
            target_layers = [ai_cls_model.features[-1]]
            cam_obj = GradCAMPlusPlus(model=ai_cls_model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(int(target_category))]

            with torch.enable_grad():
                cam_224 = cam_obj(input_tensor=input_tensor_cls, targets=targets)[0]

            # Resize CAM
            cam_tensor = torch.from_numpy(cam_224).float().unsqueeze(0).unsqueeze(0)
            cam_resized = F.interpolate(cam_tensor, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            cam_np = cam_resized.squeeze().cpu().numpy()

            # Tăng tương phản mạnh mẽ cho Heatmap
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() + 1e-8)
            cam_np = np.power(cam_np, 3) # Mũ 3 để cực kỳ đậm ở đỉnh
            cam_np[cam_np < 0.25] = 0     # Cắt nhiễu mạnh hơn

            # Cắt theo mask phổi
            cam_constrained = cam_np * mask_np
            val_max = cam_constrained.max()
            if val_max > 0:
                cam_constrained = cam_constrained / val_max

            # --- BƯỚC C: Tạo ảnh màu JET từ CAM (Chỉ màu, không nền) ---
            heatmap_uint8 = np.uint8(255 * cam_constrained)
            heatmap_color_jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color_jet = cv2.cvtColor(heatmap_color_jet, cv2.COLOR_BGR2RGB)
            heatmap_color_jet_float = np.float32(heatmap_color_jet) / 255.0

            # --- BƯỚC D: PHA TRỘN (Blending) ---
            # Chúng ta trộn Heatmap JET (C) lên trên nền Xanh Dương (A)
            
            # Mask cho vùng có tín hiệu heatmap mạnh
            heatmap_active_mask = (cam_constrained > 0.05)[..., np.newaxis]
            
            # Bắt đầu từ ảnh nền xanh
            final_heatmap_rgb = blue_tint_img.copy()
            
            # Tại những điểm có heatmap, ta trộn màu JET vào
            # Công thức trộn: 30% Nền Xanh + 70% Màu JET
            blended_color = 0.3 * blue_tint_img + 0.7 * heatmap_color_jet_float
            
            # Áp dụng trộn màu vào ảnh cuối cùng
            np.copyto(final_heatmap_rgb, blended_color, where=heatmap_active_mask)
            
            # Kẹp giá trị lần cuối
            final_heatmap_rgb = np.clip(final_heatmap_rgb, 0, 1)

        # =========================================================
        # XUẤT ẢNH KẾT QUẢ
        # =========================================================
        heatmap_pil = Image.fromarray(np.uint8(255 * final_heatmap_rgb))
        buf = io.BytesIO()
        heatmap_pil.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    except Exception as e:
        print(f"Heatmap Super-Blend Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =========================================================
# 5. ROUTER + LOAD MODELS
# =========================================================
router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.on_event("startup")
async def load_models():
    global ai_cls_model, ai_seg_model

    try:
        print("Loading EfficientNet classification model on", device)
        m_cls = build_efficientnet(num_classes=len(CLASS_NAMES)).to(device)
        state_cls = torch.load(CLS_MODEL_PATH, map_location=device)
        m_cls.load_state_dict(state_cls, strict=True)
        m_cls.eval()
        ai_cls_model = m_cls
        print("Classification model loaded.")
    except Exception as e:
        ai_cls_model = None
        print("Classification model load failed:", str(e))

    try:
        print("Loading U-Net segmentation model on", device)
        m_seg = UNet(in_channels=3, out_channels=1).to(device)
        state_seg = torch.load(SEG_MODEL_PATH, map_location=device)
        m_seg.load_state_dict(state_seg, strict=True)
        m_seg.eval()
        ai_seg_model = m_seg
        print("Segmentation model loaded.")
    except Exception as e:
        ai_seg_model = None
        print("Segmentation model load failed:", str(e))


# =========================================================
# 6. API: DETECT + SAVE DB
# =========================================================
@router.post("/detect", response_model=AnalysisOut)
async def detect_and_save_analysis(
    id_patient: int = Form(...),
    id_doctor: int = Form(...),
    note: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if ai_cls_model is None or ai_seg_model is None:
        raise HTTPException(status_code=503, detail="AI models are not ready.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    await assert_foreign_keys_exist(db, id_patient=id_patient, id_doctor=id_doctor)

    original_jpeg_bytes = pil_to_jpeg_bytes(image, quality=95)
    encrypted_original = encrypt_image_from_bytes(original_jpeg_bytes)

    input_tensor_cls = cls_transform(image).unsqueeze(0).to(device)
    input_tensor_seg = seg_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = ai_cls_model(input_tensor_cls)
        probs = F.softmax(outputs, dim=1)
        k = min(3, len(CLASS_NAMES))
        topk_prob, topk_idx = torch.topk(probs, k=k)

    top1_conf = float(topk_prob[0][0].item())
    top1_idx = int(topk_idx[0][0].item())
    ai_pred = CLASS_NAMES[top1_idx]

    if top1_conf < CONFIDENCE_THRESHOLD:
        warning = f"[AI Warning] Low confidence ({top1_conf * 100:.2f}%)."
        note = f"{note} | {warning}" if note else warning

    top3_list = []
    for i in range(k):
        top3_list.append({
            "disease": CLASS_NAMES[int(topk_idx[0][i].item())],
            "confidence": f"{float(topk_prob[0][i].item()) * 100:.2f}%"
        })
    top3_json_str = safe_json_dumps(top3_list)

    heatmap_bytes = compute_mask_guided_heatmap_bytes(
        input_tensor_cls=input_tensor_cls,
        input_tensor_seg=input_tensor_seg,
        original_pil=image,
        target_category=top1_idx,
        is_normal=(ai_pred == "Normal")  
    )
    encrypted_heatmap = encrypt_image_from_bytes(heatmap_bytes) if heatmap_bytes else None

    q = text("""
        INSERT INTO skin_analysis
        (id_patient, id_doctor, image_data, heatmap_data, ai_prediction, ai_confidence, top3_json, doctor_diagnosis, note)
        VALUES
        (:id_patient, :id_doctor, :image_data, :heatmap_data, :ai_prediction, :ai_confidence, :top3_json, :doctor_diagnosis, :note)
        RETURNING id_analysis, id_patient, id_doctor, ai_prediction, ai_confidence, top3_json, doctor_diagnosis, note, created_at;
    """)

    params = {
        "id_patient": id_patient,
        "id_doctor": id_doctor,
        "image_data": encrypted_original,
        "heatmap_data": encrypted_heatmap,
        "ai_prediction": ai_pred,
        "ai_confidence": top1_conf * 100.0,
        "top3_json": top3_json_str,
        "doctor_diagnosis": None,
        "note": note,
    }

    try:
        r = await db.execute(q, params)
        await db.commit()
        record = r.mappings().first()

        result_dict = dict(record)
        result_dict["image_url"] = f"/analysis/image/{record.id_analysis}/original"
        result_dict["heatmap_url"] = f"/analysis/image/{record.id_analysis}/heatmap"
        return result_dict

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


# =========================================================
# 7. API: GET IMAGE
# =========================================================
@router.get("/image/{id_analysis}/{img_type}")
async def get_secure_image(
    id_analysis: int,
    img_type: str,
    db: AsyncSession = Depends(get_db),
    preview: Optional[bool] = Query(default=None),
):
    if img_type not in ["original", "heatmap"]:
        raise HTTPException(status_code=400, detail="Invalid image type.")

    if preview is None:
        preview = (img_type == "heatmap" and HEATMAP_DEFAULT_PREVIEW)

    col_name = "image_data" if img_type == "original" else "heatmap_data"
    q = text(f"SELECT {col_name} FROM skin_analysis WHERE id_analysis = :id")

    result = await db.execute(q, {"id": id_analysis})
    record = result.mappings().first()

    if not record or not record[col_name]:
        raise HTTPException(status_code=404, detail="Image not found.")

    try:
        encrypted_bytes = record[col_name]
        if isinstance(encrypted_bytes, memoryview):
            encrypted_bytes = encrypted_bytes.tobytes()

        decrypted_bytes = decrypt_image_to_bytes(encrypted_bytes)

        if img_type == "heatmap" and preview:
            try:
                hm_pil = Image.open(io.BytesIO(decrypted_bytes))
                hm_sq = letterbox_to_square(hm_pil, size=HEATMAP_PREVIEW_SIZE, bg=HEATMAP_PREVIEW_BG)
                decrypted_bytes = pil_to_jpeg_bytes(hm_sq, quality=90)
            except Exception:
                pass

        return StreamingResponse(io.BytesIO(decrypted_bytes), media_type="image/jpeg")

    except Exception:
        raise HTTPException(status_code=500, detail="Decryption failed.")


# =========================================================
# 8. API: LIST / GET (NO BLOB)
# =========================================================
def get_select_query_without_blob() -> str:
    return """
        SELECT id_analysis, id_patient, id_doctor, ai_prediction, ai_confidence,
               top3_json, doctor_diagnosis, note, created_at
        FROM skin_analysis
    """

def process_records_with_urls(records):
    results = []
    for row in records:
        r = dict(row)
        r["image_url"] = f"/analysis/image/{r['id_analysis']}/original"
        r["heatmap_url"] = f"/analysis/image/{r['id_analysis']}/heatmap"
        results.append(r)
    return results

@router.get("", response_model=List[AnalysisOut])
async def get_all_analysis(db: AsyncSession = Depends(get_db)):
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
        raise HTTPException(status_code=404, detail="Analysis not found.")

    res = dict(row)
    res["image_url"] = f"/analysis/image/{res['id_analysis']}/original"
    res["heatmap_url"] = f"/analysis/image/{res['id_analysis']}/heatmap"
    return res


# =========================================================
# 9. API: DOCTOR CONFIRM
# =========================================================
@router.put("/{id_analysis}/diagnose", response_model=AnalysisOut)
async def doctor_confirm_analysis(
    id_analysis: int,
    payload: DoctorDiagnosisUpdate,
    db: AsyncSession = Depends(get_db),
):
    q_select = text("SELECT ai_prediction FROM skin_analysis WHERE id_analysis = :id_analysis")
    r = await db.execute(q_select, {"id_analysis": id_analysis})
    record = r.mappings().first()

    if not record:
        raise HTTPException(status_code=404, detail="Record not found.")

    final_diagnosis = record["ai_prediction"] if payload.confirm_ai else payload.custom_diagnosis
    if not payload.confirm_ai and not payload.custom_diagnosis:
        raise HTTPException(status_code=400, detail="custom_diagnosis is required when confirm_ai is false.")

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
        res = dict(updated_row)
        res["image_url"] = f"/analysis/image/{res['id_analysis']}/original"
        res["heatmap_url"] = f"/analysis/image/{res['id_analysis']}/heatmap"
        return res

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
