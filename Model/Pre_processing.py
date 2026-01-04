import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# =============================================
# CẤU HÌNH ĐƯỜNG DẪN (Giữ nguyên như cũ)
# =============================================
INPUT_FOLDER = 'D:/Hair Loss/hair-loss.v3i.multiclass/test'
OUTPUT_FOLDER = 'D:/Hair Loss/hair-loss.v3i.multiclass/processedtest'
TARGET_SIZE = 224 # Kích thước cuối cùng mong muốn
# =============================================

def read_image_safe(path):
    """Hàm đọc ảnh bất chấp đường dẫn có dấu cách"""
    try:
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        stream.close()
        return img
    except:
        return None

def zoom_and_crop(image_path, output_dir, target_size=224):
    img = read_image_safe(image_path)
    if img is None: return 0
    
    h, w, _ = img.shape
    base_name = os.path.basename(image_path).split('.')[0]
    count = 0

    # CHIẾN THUẬT: CẮT MẢNH NHỎ Ở GIỮA RỒI PHÓNG TO
    # Chúng ta giả định đầu người nằm ở giữa ảnh
    center_y, center_x = h // 2, w // 2
    
    # Định nghĩa kích thước "cửa sổ nhìn" (nhỏ hơn ảnh gốc nhiều để tạo hiệu ứng zoom)
    # Lấy khoảng 40% - 60% khu vực giữa ảnh (để loại bỏ tai, cổ, background)
    scan_size = int(min(h, w) * 0.5) 
    
    # Tạo danh sách các điểm cần soi (Giữa, và lệch trái phải một chút)
    offsets = [
        (0, 0),          # Chính giữa đỉnh đầu
        (-20, -20),      # Lệch lên trái xíu
        (-20, 20),       # Lệch lên phải xíu
        (20, 0),         # Lệch xuống dưới xíu
    ]

    for i, (dy, dx) in enumerate(offsets):
        # Tính tọa độ cắt
        cy = center_y + dy
        cx = center_x + dx
        
        y1 = max(0, cy - scan_size // 2)
        y2 = min(h, cy + scan_size // 2)
        x1 = max(0, cx - scan_size // 2)
        x2 = min(w, cx + scan_size // 2)
        
        # Cắt mảnh nhỏ ra (Đây là texture da đầu)
        patch = img[y1:y2, x1:x2]
        
        # Nếu cắt được mảnh có hình hài đàng hoàng
        if patch.size > 0 and patch.shape[0] > 50 and patch.shape[1] > 50:
            # PHÓNG TO LÊN 224x224 (Đây là bước quan trọng nhất!)
            patch_resized = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            # Lưu lại
            save_name = f"{base_name}_zoom_{i}.jpg"
            save_path = os.path.join(output_dir, save_name)
            
            # Lưu an toàn
            is_success, im_buf = cv2.imencode(".jpg", patch_resized)
            if is_success:
                im_buf.tofile(save_path)
                count += 1

    return count

# =================MAIN RUN=================
if os.path.exists(OUTPUT_FOLDER):
    # Xóa file cũ trong folder output đi cho sạch (nếu muốn)
    pass
else:
    os.makedirs(OUTPUT_FOLDER)

image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"🔍 Tìm thấy {len(image_files)} ảnh gốc.")
print("🚀 Đang thực hiện chế độ ZOOM & CROP (Lấy cận cảnh da đầu)...")

total_patches = 0
for img_name in tqdm(image_files):
    img_path = os.path.join(INPUT_FOLDER, img_name)
    total_patches += zoom_and_crop(img_path, OUTPUT_FOLDER, TARGET_SIZE)

print("="*50)
print(f"🎉 Xong! Đã tạo ra: {total_patches} ảnh cận cảnh.")
print(f"📁 Kiểm tra folder: {OUTPUT_FOLDER}")
print("Bác mở thử xem, giờ nó sẽ là ảnh phóng to vào da đầu đấy!")
print("="*50)