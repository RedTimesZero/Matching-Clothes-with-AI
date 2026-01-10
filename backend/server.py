import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import torch

app = FastAPI()

# --- 1. 設定 CORS (這一步超重要) ---
# 因為你的 React 在 localhost:5173，Python 在 localhost:8000
# 如果不加這個，瀏覽器會因為安全理由擋住連線
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源 (開發階段方便)，上線可改成 ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 載入模型 (伺服器啟動時只載入一次) ---
print("正在載入 AI 模型...請稍候")
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("模型載入完成！")

def calculate_similarity(image1_bytes, image2_bytes):
    try:
        # 直接從記憶體讀取圖片，不需要存成檔案
        img1 = Image.open(io.BytesIO(image1_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB")
        
        inputs1 = processor(images=img1, return_tensors="pt")
        inputs2 = processor(images=img2, return_tensors="pt")
        
        with torch.no_grad():
            feat1 = model.get_image_features(**inputs1)
            feat2 = model.get_image_features(**inputs2)
            
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
    # 讀取上傳的檔案內容
    img1_data = await file1.read()
    img2_data = await file2.read()
    
    score = calculate_similarity(img1_data, img2_data)
    
    return {"similarity": score, "message": "success"}

# --- 啟動指令 (寫在註解裡備忘) ---
# uvicorn server:app --reload