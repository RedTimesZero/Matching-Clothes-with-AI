import io
import requests 
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import json
import os
import sys

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
# 1. 定義模型架構 (雙頭龍)
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
# 2. 資源管理 (延遲載入核心)
# ==========================================

# 全域變數
classifier = None
clip_model = None
clip_processor = None
cat_map = None
color_map = None
CLASS_NAMES = None
COLOR_NAMES = None
transform_classify = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 預處理 (靜態定義，不吃記憶體)
transform_classify = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_class_mappings():
    """載入 JSON 設定檔"""
    json_path = os.path.join(os.path.dirname(__file__), "class_mapping.json")
    if not os.path.exists(json_path):
        print(f"❌ 找不到 class_mapping.json")
        return {}, {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('cat_map', {}), data.get('color_map', {})

def get_classifier_model():
    """
    取得分類模型 (ResNet)。如果還沒載入，現在才載入。
    """
    global classifier, CLASS_NAMES, COLOR_NAMES
    
    if classifier is not None:
        return classifier, CLASS_NAMES, COLOR_NAMES

    print("⚡ 正在初始化分類模型 (ResNet)...")
    
    # 1. 載入類別
    c_map, co_map = load_class_mappings()
    # 轉換 key 為 int
    c_map = {int(k): v for k, v in c_map.items()}
    co_map = {int(k): v for k, v in co_map.items()}
    
    CLASS_NAMES = [c_map[i] for i in sorted(c_map.keys())]
    COLOR_NAMES = [co_map[i] for i in sorted(co_map.keys())]
    
    num_cats = len(CLASS_NAMES) if CLASS_NAMES else 1
    num_cols = len(COLOR_NAMES) if COLOR_NAMES else 1
    
    # 2. 載入模型架構
    model = MultiHeadResNet(num_cats, num_cols)
    
    # 3. 尋找權重檔 (容錯大小寫)
    pth_name = "Model_Weights.pth"
    if not os.path.exists(pth_name):
        pth_name = "model_weights.pth" # 試試看小寫
    
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
        print("❌ 找不到 Model_Weights.pth (請確認檔案已上傳)")
        classifier = None

    return classifier, CLASS_NAMES, COLOR_NAMES

def get_clip_model():
    """
    取得 CLIP 模型。如果還沒載入，現在才載入。
    """
    global clip_model, clip_processor
    
    if clip_model is not None:
        return clip_model, clip_processor
        
    print("⚡ 正在初始化 CLIP 模型...")
    try:
        CLIP_NAME = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(CLIP_NAME)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        print("✅ CLIP 模型載入成功")
    except Exception as e:
        print(f"❌ CLIP 載入失敗: {e}")
        
    return clip_model, clip_processor

# ==========================================
# 3. API 接口
# ==========================================

@app.get("/")
def home():
    return {"message": "AI Backend is Running!"}

@app.post("/predict_type")
async def predict_type(file: UploadFile = File(...)):
    # 這裡只呼叫 ResNet，不呼叫 CLIP
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
            
            # 安全存取
            pred_cat = classes[c_idx] if classes and c_idx < len(classes) else "unknown"
            pred_col = colors[co_idx] if colors and co_idx < len(colors) else "unknown"

        return {"category": pred_cat, "color": pred_col}
    except Exception as e:
        print(f"預測錯誤: {e}")
        return {"category": "error", "color": "error"}

@app.post("/compare_url")
async def compare_url(file1: UploadFile = File(...), url2: str = Form(...)):
    # ⚠️ 關鍵修改：這裡只呼叫 CLIP，絕對不要呼叫 ResNet (initialize_models)
    # 這樣可以避免記憶體爆掉
    c_model, c_processor = get_clip_model()
    
    if c_model is None:
        return {"similarity": 0, "message": "CLIP model not available"}

    try:
        # 1. 讀取使用者上傳
        img1_data = await file1.read()
        img1 = Image.open(io.BytesIO(img1_data)).convert("RGB")
        
        # 2. 下載連結圖片
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url2, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            return {"similarity": 0, "message": "Download failed"}
            
        img2 = Image.open(io.BytesIO(resp.content)).convert("RGB")
        
        # 3. 計算相似度
        inputs1 = c_processor(images=img1, return_tensors="pt")
        inputs2 = c_processor(images=img2, return_tensors="pt")
        
        with torch.no_grad():
            feat1 = c_model.get_image_features(**inputs1)
            feat2 = c_model.get_image_features(**inputs2)
            
        feat1 = feat1 / feat1.norm(p=2, dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(p=2, dim=-1, keepdim=True)
        
        score = F.cosine_similarity(feat1, feat2).item() * 100
        return {"similarity": score, "message": "success"}
        
    except Exception as e:
        print(f"比對錯誤: {e}")
        return {"similarity": 0, "message": str(e)}