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

# Hugging Face API 設定
HF_API_URL = os.getenv(
    "HF_API_URL",
    "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-base-patch32"
)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

def pil_to_bytes(img):
    """Convert PIL image to bytes for HTTP upload."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def hf_image_embedding(image_bytes):
    """Call Hugging Face Inference API to get image embedding."""
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set")
    headers = {"Content-Type": "application/octet-stream", **HF_HEADERS}
    resp = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"HF API error {resp.status_code}: {resp.text}")
    data = resp.json()
    emb = torch.tensor(data[0], dtype=torch.float32)
    if emb.ndim > 1:
        emb = emb.view(emb.shape[0], -1)
    emb = emb.squeeze(0)
    return emb

def cosine_score(a, b):
    """Cosine similarity (%) between two 1-D tensors."""
    if a.ndim != 1:
        a = a.view(-1)
    if b.ndim != 1:
        b = b.view(-1)
    a = a / (a.norm(p=2) + 1e-8)
    b = b / (b.norm(p=2) + 1e-8)
    return float(torch.dot(a, b).item() * 100)

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
    print("\n" + "="*60)
    print("📸 收到 /compare_url 請求")
    print(f"🔗 url2: {url2[:80]}...")
    
    if not HF_API_TOKEN:
        print("❌ HF_API_TOKEN 未設定")
        return {"similarity": 0, "message": "HF_API_TOKEN not set"}

    try:
        # 讀取上傳圖片
        file1_data = await file1.read()
        print(f"✅ file1 大小: {len(file1_data)} bytes")
        img1 = Image.open(io.BytesIO(file1_data)).convert("RGB")
        print(f"✅ img1 尺寸: {img1.size}")
        
        # 下載衣櫃圖片
        print(f"⬇️  正在下載 url2...")
        r = requests.get(url2, timeout=10)
        r.raise_for_status()
        print(f"✅ url2 下載成功: {len(r.content)} bytes, status={r.status_code}")
        img2 = Image.open(io.BytesIO(r.content)).convert("RGB")
        print(f"✅ img2 尺寸: {img2.size}")

        # 呼叫 HF API 取得 embedding
        print("🤖 呼叫 HF API 取得 embedding...")
        emb1 = hf_image_embedding(pil_to_bytes(img1))
        print(f"✅ emb1 shape: {emb1.shape}, norm: {emb1.norm().item():.4f}")
        
        emb2 = hf_image_embedding(pil_to_bytes(img2))
        print(f"✅ emb2 shape: {emb2.shape}, norm: {emb2.norm().item():.4f}")

        # 計算相似度
        score = cosine_score(emb1, emb2)
        print(f"🎯 相似度分數: {score:.2f}%")
        print("="*60 + "\n")
        
        return {"similarity": score, "message": "success"}

    except Exception as e:
        print(f"❌ 比對錯誤: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return {"similarity": 0, "message": str(e)}