import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import torch

app = FastAPI()
# --- 加入這段在 app = FastAPI() 之後 ---
@app.get("/")
def home():
    return {"message": "Hello! AI Closet Backend is running!"}

# --- 1. 設定 CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 修改重點：全域變數設為 None (一開始不載入) ---
model = None
processor = None
MODEL_NAME = "openai/clip-vit-base-patch32"

def get_model():
    """
    這就是「延遲載入」的核心函式。
    只有被呼叫時，才會檢查模型在不在。
    如果不在，現在才開始載入。
    """
    global model, processor
    
    if model is None:
        print("⚠️ 第一次請求，正在載入 AI 模型 (這可能會花幾秒鐘)...")
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("✅ 模型載入完成！")
    
    return model, processor

def calculate_similarity(image1_bytes, image2_bytes):
    try:
        # 重點：要用的時候才呼叫 get_model() 拿模型
        current_model, current_processor = get_model()

        img1 = Image.open(io.BytesIO(image1_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB")
        
        inputs1 = current_processor(images=img1, return_tensors="pt")
        inputs2 = current_processor(images=img2, return_tensors="pt")
        
        with torch.no_grad():
            feat1 = current_model.get_image_features(**inputs1)
            feat2 = current_model.get_image_features(**inputs2)
            
        feat1 = feat1 / feat1.norm(p=2, dim=-1, keepdim=True)
        feat2 = feat2 / feat2.norm(p=2, dim=-1, keepdim=True)
        
        score = F.cosine_similarity(feat1, feat2).item() * 100
        return score
    except Exception as e:
        print(f"Error: {e}")
        return 0

# --- 3. 建立 API 接口 ---
@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1_data = await file1.read()
    img2_data = await file2.read()
    
    score = calculate_similarity(img1_data, img2_data)
    
    return {"similarity": score, "message": "success"}

# uvicorn server:app --reload