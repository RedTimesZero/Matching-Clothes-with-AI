import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------
# 1. 設定與準備
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'bestmodel.pth' # 請確認路徑正確

# 定義類別名稱 (請依照您訓練時的順序修改這裡！)
# 假設是單一標籤分類 (例如種類)：
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 

# ----------------------
# 2. 定義模型架構 (關鍵！)
# ----------------------
# 注意：這裡必須跟您訓練時的架構完全一樣
# 舉例：如果您是用 ResNet18 Transfer Learning
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names)) 

# ----------------------
# 3. 載入權重
# ----------------------
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("模型權重載入成功！")
except Exception as e:
    print(f"載入失敗，請檢查模型架構是否匹配。錯誤訊息: {e}")

model = model.to(device)
model.eval() # 設定為推論模式

# ----------------------
# 4. 圖片預處理
# ----------------------
# 這些數值通常是 ImageNet 的標準，若您訓練時有特殊設定請修改
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------
# 5. 預測函數
# ----------------------
def predict_image(image_path):
    
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    img_tensor = data_transforms(image).unsqueeze(0).to(device) # 增加 batch 維度
    
    with torch.no_grad():
        outputs = model(img_tensor)
        # 如果是多標籤 (Multi-label, 同時辨識顏色和種類)，這裡處理方式會不同
        # 這裡是假設單一分類 (Single-label)
        _, preds = torch.max(outputs, 1)
        
    print(f"預測結果: {class_names[preds[0]]}")

# ----------------------
# 6. 執行測試
image_path = r"C:\Users\USER\Desktop\Matching-Clothes-with-AI\IP\example_closet\white_tshirt.jpg"
predict_image(image_path)