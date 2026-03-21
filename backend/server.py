import io
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch.nn.functional as F
import json
import os
import sys
import numpy as np
from skimage.metrics import structural_similarity as ssim
from rembg import remove

app = FastAPI()

# --- 設定 CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. 定義模型架構 (雙頭龍) - ResNet 分類器
# ==========================================
class MultiHeadResNet(nn.Module):
    def __init__(self, num_cats, num_cols):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_cat = nn.Linear(num_features, num_cats)
        self.fc_color = nn.Linear(num_features, num_cols)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_cat(features), self.fc_color(features)

# ==========================================
# 2. 資源管理 (延遲載入)
# ==========================================

# 全域變數 - ResNet 分類器
classifier = None
CLASS_NAMES = None
COLOR_NAMES = None

# 裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 圖片分類預處理 (ResNet)
transform_classify = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2.1 SSIM 相似度計算函數
# ==========================================
def image_similarity_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    用 SSIM 計算兩張圖片的相似度 (0-100%)
    無需 AI 模型，記憶體佔用 <1MB
    
    Args:
        img1: PIL Image 物件
        img2: PIL Image 物件
    
    Returns:
        相似度分數 (0-100)
    """
    try:
        # 調整成相同尺寸 (224x224，與分類模型一致)
        size = (224, 224)
        img1_resized = img1.resize(size)
        img2_resized = img2.resize(size)
        
        # 轉成灰階
        img1_gray = img1_resized.convert('L')
        img2_gray = img2_resized.convert('L')
        
        # 轉成 numpy array
        arr1 = np.array(img1_gray, dtype=np.float32)
        arr2 = np.array(img2_gray, dtype=np.float32)
        
        # 計算 SSIM (-1 到 1 之間，通常 0-1)
        score = ssim(arr1, arr2, data_range=255.0)
        
        # 轉成 0-100%
        similarity = max(0, min(100, score * 100))
        
        return similarity
    
    except Exception as e:
        print(f"❌ SSIM 計算錯誤: {e}")
        return 0.0

def load_class_mappings():
    """載入類別映射 JSON"""
    json_path = os.path.join(os.path.dirname(__file__), "class_mapping.json")
    if not os.path.exists(json_path):
        print(f"❌ 找不到 class_mapping.json")
        return {}, {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('cat_map', {}), data.get('color_map', {})

def get_classifier_model():
    """延遲載入 ResNet 分類模型"""
    global classifier, CLASS_NAMES, COLOR_NAMES
    
    if classifier is not None:
        return classifier, CLASS_NAMES, COLOR_NAMES

    print("⚡ 正在初始化分類模型 (ResNet)...")
    
    c_map, co_map = load_class_mappings()
    c_map = {int(k): v for k, v in c_map.items()}
    co_map = {int(k): v for k, v in co_map.items()}
    
    CLASS_NAMES = [c_map[i] for i in sorted(c_map.keys())] if c_map else []
    COLOR_NAMES = [co_map[i] for i in sorted(co_map.keys())] if co_map else []
    
    num_cats = len(CLASS_NAMES) if CLASS_NAMES else 1
    num_cols = len(COLOR_NAMES) if COLOR_NAMES else 1
    
    model = MultiHeadResNet(num_cats, num_cols)
    
    pth_name = "Model_Weights.pth"
    if not os.path.exists(pth_name):
        pth_name = "model_weights.pth"
    
    if os.path.exists(pth_name):
        try:
            state_dict = torch.load(pth_name, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            classifier = model
            print(f"✅ 分類模型載入成功 ({pth_name})")
        except Exception as e:
            print(f"❌ 權重檔載入失敗: {e}")
            classifier = None
    else:
        print("❌ 找不到 Model_Weights.pth")
        classifier = None

    return classifier, CLASS_NAMES, COLOR_NAMES

# ==========================================
# 3. API 接口
# ==========================================

@app.get("/")
def home():
    return {"message": "AI Backend is Running!"}

@app.post("/predict_type")
async def predict_type(file: UploadFile = File(...)):
    model, classes, colors = get_classifier_model()
    
    if model is None:
        return {"category": "unknown", "color": "unknown", "error": "Model failed to load"}

    try:
        image_data = await file.read()
        original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # --- 🚀 補回 AI 去背魔法 ---
        print("✨ 正在執行 AI 去背...")
        no_bg_image = remove(original_image)
        
        # 將去背後的圖片貼在純白背景上，模擬電商圖庫環境，增加顏色準確度
        white_bg = Image.new("RGB", no_bg_image.size, (255, 255, 255))
        if no_bg_image.mode == 'RGBA':
            white_bg.paste(no_bg_image, mask=no_bg_image.split()[3])
            final_image = white_bg
        else:
            final_image = no_bg_image.convert("RGB")
        # --------------------------
        
        # 使用去背後的 final_image 進行預測
        img_tensor = transform_classify(final_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cat_logits, col_logits = model(img_tensor)
            _, cat_idx = torch.max(cat_logits, 1)
            _, col_idx = torch.max(col_logits, 1)
            
            c_idx = cat_idx.item()
            co_idx = col_idx.item()
            
            pred_cat = classes[c_idx] if classes and c_idx < len(classes) else "unknown"
            pred_col = colors[co_idx] if colors and co_idx < len(colors) else "unknown"

        print(f"🎯 辨識結果: {pred_col} {pred_cat}")
        return {"category": pred_cat, "color": pred_col}
        
    except Exception as e:
        print(f"❌ 預測錯誤: {e}")
        return {"category": "error", "color": "error"}


@app.post("/compare_similarity")
async def compare_similarity(data: dict = Body(...)):
    print("\n" + "="*60)
    print("📸 收到 /compare_similarity 請求 (真實比對模式)")
    
    import base64
    import io
    import requests
    
    try: 
        source_b64 = data.get("source_image", "")
        closet_items = data.get("closet_items", []) 
        
        # 1. 解碼前端傳來的 Base64 圖片
        # 移除 "data:image/jpeg;base64," 這類的前綴
        if "," in source_b64:
            source_b64 = source_b64.split(",")[1]
            
        img_bytes = base64.b64decode(source_b64)
        source_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        print(f"📦 準備比對 {len(closet_items)} 件衣物")
        results = []

        # 2. 逐一下載衣櫃圖片並進行 SSIM 真實比對
        for item in closet_items:
            img_url = item.get("image_url")
            if not img_url:
                continue
            
            try:
                # 下載衣櫃裡的衣服圖片
                resp = requests.get(img_url, timeout=5)
                resp.raise_for_status()
                closet_img = Image.open(io.BytesIO(resp.content)).convert("RGB")

                # 🚀 呼叫你寫好的 SSIM 函數！
                sim_score = image_similarity_ssim(source_img, closet_img)
                
                # 你的函數回傳 0-100，我們轉成前端需要的 0.00-1.00 格式
                sim_value = round(sim_score / 100.0, 2)

                results.append({
                    "id": item.get("id"),
                    "similarity": sim_value, 
                    "title": item.get("title", "未命名衣物")
                })
                print(f"  - 和 [{item.get('title')}] 的真實相似度: {sim_score:.1f}%")

            except Exception as img_err:
                print(f"  - ⚠️ 跳過項目 {item.get('id')}: 圖片處理失敗 ({img_err})")
                continue

        # 3. 排序並回傳
        results.sort(key=lambda x: x["similarity"], reverse=True)
        print(f"✅ 成功處理 {len(results)} 筆真實比對")
        print("="*60 + "\n")
        
        return {"top_matches": results}

    except Exception as e:
        print(f"❌ 後端出錯: {e}")
        import traceback
        traceback.print_exc()
        return {"top_matches": [], "error": str(e)}