import io
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch.nn.functional as F
import json
import os
import sys
import numpy as np
from skimage.metrics import structural_similarity as ssim

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
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        img_tensor = transform_classify(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cat_logits, col_logits = model(img_tensor)
            _, cat_idx = torch.max(cat_logits, 1)
            _, col_idx = torch.max(col_logits, 1)
            
            c_idx = cat_idx.item()
            co_idx = col_idx.item()
            
            pred_cat = classes[c_idx] if classes and c_idx < len(classes) else "unknown"
            pred_col = colors[co_idx] if colors and co_idx < len(colors) else "unknown"

        return {"category": pred_cat, "color": pred_col}
    except Exception as e:
        print(f"預測錯誤: {e}")
        return {"category": "error", "color": "error"}

@app.post("/compare_url")
async def compare_url(file1: UploadFile = File(...), url2: str = Form(...)):
    print("\n" + "="*60)
    print("📸 收到 /compare_url 請求")
    print(f"🔗 url2: {url2[:80]}...")
    
    try:
        # 讀取上傳的第一張圖
        file1_data = await file1.read()
        print(f"✅ file1 大小: {len(file1_data)} bytes")
        img1 = Image.open(io.BytesIO(file1_data)).convert("RGB")
        print(f"✅ img1 尺寸: {img1.size}")
        
        # 下載第二張圖
        print(f"⬇️ 正在下載 url2...")
        r = requests.get(url2, timeout=12)
        r.raise_for_status()
        print(f"✅ url2 下載成功: {len(r.content)} bytes")
        img2 = Image.open(io.BytesIO(r.content)).convert("RGB")
        print(f"✅ img2 尺寸: {img2.size}")

        # 計算 SSIM 相似度（無需 CLIP 模型，記憶體佔用 <1MB）
        print("🤖 計算 SSIM 相似度...")
        score = image_similarity_ssim(img1, img2)
        print(f"🎯 SSIM 相似度: {score:.2f}%")
        print("="*60 + "\n")
        
        return {
            "similarity": round(score, 2),
            "message": "success"
        }

    except Exception as e:
        print(f"❌ 比對錯誤: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return {"similarity": 0, "message": str(e)}