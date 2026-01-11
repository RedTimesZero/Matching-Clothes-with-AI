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
# 2. 從 JSON 加載類別和顏色映射
# ==========================================
def load_class_mappings():
    """從 JSON 文件加載類別映射"""
    json_path = os.path.join(os.path.dirname(__file__), "class_mapping.json")
    
    if not os.path.exists(json_path):
        print(f"❌ class_mapping.json 不存在: {json_path}")
        print("請先運行 train_model.py 生成映射文件")
        sys.exit(1)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cat_map = {int(k): v for k, v in data['cat_map'].items()}
    color_map = {int(k): v for k, v in data['color_map'].items()}
    
    return cat_map, color_map

# 延遲初始化變量
classifier = None
clip_model = None
clip_processor = None
cat_map = None
color_map = None
CLASS_NAMES = None
COLOR_NAMES = None
NUM_CATS = None
NUM_COLORS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_initialized = False

# 預處理 (不依賴模型，可以提前定義)
transform_classify = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def initialize_models():
    """延遲初始化模型，只在第一次請求時加載"""
    global classifier, clip_model, clip_processor, cat_map, color_map
    global CLASS_NAMES, COLOR_NAMES, NUM_CATS, NUM_COLORS, models_initialized
    
    if models_initialized:
        return
    
    print("=" * 50)
    print("首次請求，正在加載模型...")
    print("=" * 50)
    
    try:
        # 加載類別映射
        print("正在加載類別映射...")
        cat_map, color_map = load_class_mappings()
        CLASS_NAMES = [cat_map[i] for i in sorted(cat_map.keys())]
        COLOR_NAMES = [color_map[i] for i in sorted(color_map.keys())]
        NUM_CATS = len(CLASS_NAMES)
        NUM_COLORS = len(COLOR_NAMES)
        print(f"✅ 已加載 {NUM_CATS} 種服裝類別和 {NUM_COLORS} 種顏色")
        
        # 載入分類模型
        print(f"正在載入分類模型... (類別: {NUM_CATS})")
        model = MultiHeadResNet(num_cats=NUM_CATS, num_cols=NUM_COLORS)
        state_dict = torch.load("model_weights.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        classifier = model
        print("✅ model_weights.pth 載入成功！")
        
        models_initialized = True
        print("=" * 50)
        print("✅ 分類模型初始化完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 模型初始化失敗: {e}")
        import traceback
        traceback.print_exc()

def initialize_clip_model():
    """延遲初始化 CLIP 模型，只在比對時加載"""
    global clip_model, clip_processor
    
    if clip_model is not None:
        return
    
    print("正在載入 CLIP 模型...")
    try:
        CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        print("✅ CLIP 模型載入成功！")
    except Exception as e:
        print(f"❌ CLIP 模型載入失敗: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 3. API 區域
# ==========================================

@app.get("/")
def home():
    return {"message": "AI Backend is Running!"}

# 功能一：辨識衣服種類與顏色
@app.post("/predict_type")
async def predict_type(file: UploadFile = File(...)):
    # 延迟初始化
    initialize_models()
    
    if classifier is None:
        return {"category": "unknown", "color": "unknown", "error": "Model not loaded"}
    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # 預處理
    img_tensor = transform_classify(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cat_logits, col_logits = classifier(img_tensor)
        
        _, cat_idx = torch.max(cat_logits, 1)
        _, col_idx = torch.max(col_logits, 1)
        
        # 防止 index out of range
        pred_cat = CLASS_NAMES[cat_idx.item()] if cat_idx.item() < len(CLASS_NAMES) else "unknown"
        pred_col = COLOR_NAMES[col_idx.item()] if col_idx.item() < len(COLOR_NAMES) else "unknown"

    return {"category": pred_cat, "color": pred_col}

# 功能二：直接接收網址進行比對
@app.post("/compare_url")
async def compare_url(file1: UploadFile = File(...), url2: str = Form(...)):
    # 延遲初始化分類模型
    initialize_models()
    
    # 延遲初始化 CLIP 模型 (只在這裡用到)
    initialize_clip_model()
    
    try:
        # 1. 讀取使用者上傳的圖
        img1_data = await file1.read()
        img1 = Image.open(io.BytesIO(img1_data)).convert("RGB")
        
        # 2. 讓後端下載衣櫃的圖
        # print(f"正在下載圖片: {url2}") # 除錯用
        headers = {'User-Agent': 'Mozilla/5.0'} # 偽裝成瀏覽器，避免被擋
        resp = requests.get(url2, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            print(f"下載失敗: {resp.status_code} - {url2}")
            return {"similarity": 0, "message": "Download failed"}
            
        img2 = Image.open(io.BytesIO(resp.content)).convert("RGB")
        
        # 3. CLIP 比對
        inputs1 = clip_processor(images=img1, return_tensors="pt")
        inputs2 = clip_processor(images=img2, return_tensors="pt")
        
        with torch.no_grad():
            feat1 = clip_model.get_image_features(**inputs1)
            feat2 = clip_model.get_image_features(**inputs2)
            
        feat1 = feat1 / feat1.norm(p=2, dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(p=2, dim=-1, keepdim=True)
        
        score = F.cosine_similarity(feat1, feat2).item() * 100
        return {"similarity": score, "message": "success"}
        
    except Exception as e:
        print(f"比對錯誤: {e}")
        return {"similarity": 0, "message": str(e)}