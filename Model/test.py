import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import timm
import torch.nn.functional as F
import numpy as np
import os

# ==========================================
# 1. CẤU HÌNH
# ==========================================
MODEL_PATH = "Ensemble_Hybrid_ResNet.pth"  # File model mới đã đóng gói
IMAGE_PATH = "test4.jpg"              # <--- ĐỔI TÊN ẢNH CỦA BẠN Ở ĐÂY
IMG_SIZE = 224

# Danh sách 9 Class (Đã bỏ Systemic Disease hay gì đó tùy theo folder train của bạn)
# LƯU Ý: Thứ tự này phải khớp với thứ tự folder lúc bạn train (Alphabet)
CLASS_NAMES = [
    'Acne and Rosacea Photos',
    'Atopic Dermatitis Photos',
    'Eczema Photos',
    'Melanoma Skin Cancer Nevi and Moles',
    'Nail Fungus and other Nail Disease',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Warts Molluscum and other Viral Infections'
]

NUM_CLASSES = len(CLASS_NAMES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. ĐỊNH NGHĨA LẠI KIẾN TRÚC (Phải giống hệt lúc đóng gói)
# ==========================================

# 2.1. Hybrid CNN + ViT Architecture
class HybridViTModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridViTModel, self).__init__()
        # EfficientNet-B0
        self.cnn = models.efficientnet_b0(weights=None)
        self.cnn.classifier = nn.Identity()
        
        # ViT Tiny
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        self.vit.head = nn.Identity()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x_cnn = self.cnn(x)
        x_vit = self.vit(x)
        x_cat = torch.cat((x_cnn, x_vit), dim=1)
        return self.classifier(x_cat)

# 2.2. ResNet Architecture Helper
def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# 2.3. Ensemble Packed Wrapper (QUAN TRỌNG: Phải có class này mới load được file pth)
class EnsemblePacked(nn.Module):
    def __init__(self, hybrid, resnet, weight_h=0.6, weight_r=0.4):
        super().__init__()
        self.hybrid = hybrid
        self.resnet = resnet
        self.weight_h = weight_h
        self.weight_r = weight_r

    def forward(self, x):
        logits_h = self.hybrid(x)
        logits_r = self.resnet(x)
        # Cộng gộp kết quả
        return self.weight_h * logits_h + self.weight_r * logits_r

# ==========================================
# 3. LOAD MODEL
# ==========================================
print(f"Using device: {device}")
print("⏳ Loading Ensemble model...")

try:
    # 1. Khởi tạo 2 model con rỗng (đúng số class là 9)
    hybrid_sub = HybridViTModel(num_classes=NUM_CLASSES)
    resnet_sub = build_resnet50(num_classes=NUM_CLASSES)
    
    # 2. Khởi tạo model tổng (Ensemble)
    model = EnsemblePacked(hybrid_sub, resnet_sub).to(device)
    
    # 3. Load trọng số vào model tổng
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Chế độ dự đoán
    print("✅ Model loaded successfully! (Ready to predict)")
    
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{MODEL_PATH}'")
    exit()
except Exception as e:
    print(f"❌ Lỗi cấu trúc model: {e}")
    exit()

# ==========================================
# 4. XỬ LÝ ẢNH & DỰ ĐOÁN
# ==========================================
def process_and_predict(image_path):
    # Transform: Phải giống hệt lúc train (Resize + Normalize ImageNet)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device) # [1, 3, 224, 224]
        
        with torch.no_grad():
            outputs = model(img_tensor) # Output này là tổng hợp (0.6*H + 0.4*R)
            probs = F.softmax(outputs, dim=1) # Chuyển thành %
            
        # Lấy Top 1
        top_prob, top_class_idx = torch.topk(probs, 1)
        confidence = top_prob.item()
        pred_label = CLASS_NAMES[top_class_idx.item()]
        
        return pred_label, confidence, probs.cpu().numpy()[0]
        
    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh: {e}")
        return None, None, None

# ==========================================
# 5. CHẠY THỰC TẾ
# ==========================================
if os.path.exists(IMAGE_PATH):
    print(f"\n📸 Đang dự đoán ảnh: {IMAGE_PATH}")
    label, conf, all_probs = process_and_predict(IMAGE_PATH)
    
    if label:
        print("\n" + "="*40)
        print(f"🏆 KẾT QUẢ DỰ ĐOÁN: {label}")
        print(f"🎯 ĐỘ TIN CẬY     : {conf * 100:.2f}%")
        print("="*40)
        
        print("\n📊 Chi tiết xác suất các bệnh:")
        # Sắp xếp từ cao xuống thấp
        sorted_indices = np.argsort(all_probs)[::-1]
        for idx in sorted_indices:
            print(f"{CLASS_NAMES[idx]:<55}: {all_probs[idx]*100:.2f}%")
else:
    print(f"⚠️ Không tìm thấy ảnh: {IMAGE_PATH}. Hãy kiểm tra lại đường dẫn.")