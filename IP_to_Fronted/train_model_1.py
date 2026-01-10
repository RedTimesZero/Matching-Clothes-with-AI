"""
服裝分類模型訓練腳本 (CPU/GPU 版本)
支持 GPU 運算，若無 GPU 自動降級到 CPU
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
import sys

# ============================================================
# 1. 初始化環境與下載資料
# ============================================================

def setup_kaggle_and_download():
    """設定 Kaggle API 並下載資料"""
    KAGGLE_TOKEN = "KGAT_3270ac64ff39696c1ec0b890d8d5cdca"
    
    # 設定環境變數
    os.environ['KAGGLE_USERNAME'] = "RedTimesZero"
    os.environ['KAGGLE_KEY'] = KAGGLE_TOKEN

    # 圖片資料夾不存在則下載資料集
    if not os.path.exists('images'):
        print("正在下載資料集...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
            subprocess.run(["kaggle", "datasets", "download", "-d", "paramaggarwal/fashion-product-images-small"], check=True)
            
            # Windows 用 PowerShell 解壓 ZIP
            import zipfile
            with zipfile.ZipFile('fashion-product-images-small.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            print("✅ 下載完成！")
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
            print("請手動從 Kaggle 下載資料集並解壓: https://www.kaggle.com/paramaggarwal/fashion-product-images-small")
            print("確保解壓後有 images/ 資料夾")
    else:
        print("✅ 資料集已存在，跳過下載")


def load_and_prepare_data():
    """讀取與篩選資料"""
    print("正在處理資料...")
    if not os.path.exists('styles.csv'):
        print("❌ styles.csv 不存在，請先執行 remove_tops.py")
        sys.exit(1)
    df = pd.read_csv('styles.csv', on_bad_lines='skip')

    # 清理欄位
    df.columns = [c.strip() for c in df.columns]
    df['gender'] = df['gender'].astype(str).str.strip()
    df['articleType'] = df['articleType'].astype(str).str.strip()
    df['baseColour'] = df['baseColour'].astype(str).str.strip()

    # 篩選 Women 的服飾 (用 articleType 作為細分類)
    filtered_df = df[
        (df['gender'] == 'Women') &
        (df['articleType'].notna())
    ].copy()

    # 確保 ID 正確
    filtered_df['id'] = pd.to_numeric(filtered_df['id'], errors='coerce').fillna(0).astype(int)

    # 建立標籤映射 (用 articleType 而不是 subCategory)
    filtered_df['cat_label'], cat_uniques = pd.factorize(filtered_df['articleType'])
    filtered_df['color_label'], color_uniques = pd.factorize(filtered_df['baseColour'])

    cat_map = dict(enumerate(cat_uniques))
    color_map = dict(enumerate(color_uniques))

    print(f"資料準備完成！樣本數: {len(filtered_df)}")
    print(f"服裝類型: {list(cat_uniques)}")
    return filtered_df, cat_map, color_map


# ============================================================
# 2. 定義 Dataset 類別
# ============================================================

class ClothingDataset(Dataset):
    """服裝圖片資料集"""
    def __init__(self, dataframe, transform=None, image_folder="images"):
        self.df = dataframe
        self.transform = transform
        self.image_folder = image_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row['id'])
        img_path = os.path.join(self.image_folder, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 若圖片不存在，返回零張量
            return torch.zeros(3, 224, 224), 0, 0

        if self.transform:
            image = self.transform(image)
        
        return image, int(row['cat_label']), int(row['color_label'])


# ============================================================
# 3. 定義模型
# ============================================================

class MultiHeadResNet(nn.Module):
    """多頭 ResNet 模型 (分類別和顏色)"""
    def __init__(self, num_cats, num_cols):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fc_cat = nn.Linear(num_features, num_cats)
        self.fc_color = nn.Linear(num_features, num_cols)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc_cat(features), self.fc_color(features)


# ============================================================
# 4. 訓練函數
# ============================================================

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=1):
    """訓練模型並驗證"""
    train_losses = []
    val_losses = []
    
    print("開始訓練...")
    
    for epoch in range(num_epochs):
        # ========== 訓練階段 ==========
        model.train()
        epoch_train_loss = 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for i, (imgs, cats, cols) in enumerate(train_loader):
            imgs = imgs.to(device)
            cats = cats.to(device).long()
            cols = cols.to(device).long()

            optimizer.zero_grad()
            out_cat, out_col = model(imgs)
            loss = criterion(out_cat, cats) + criterion(out_col, cols)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_losses.append(loss.item())

            if i % 50 == 0:
                print(f"Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # ========== 驗證階段 ==========
        model.eval()
        epoch_val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for imgs, cats, cols in val_loader:
                imgs = imgs.to(device)
                cats = cats.to(device).long()
                cols = cols.to(device).long()

                out_cat, out_col = model(imgs)
                loss = criterion(out_cat, cats) + criterion(out_col, cols)
                
                epoch_val_loss += loss.item()
                val_losses.append(loss.item())
                val_count += 1

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / val_count
        
        print(f"平均訓練 Loss: {avg_train_loss:.4f}")
        print(f"平均驗證 Loss: {avg_val_loss:.4f}")

    print("🎉 訓練完成！")
    return train_losses, val_losses


# ============================================================
# 5. 可視化損失曲線
# ============================================================

def plot_loss_curve(train_losses, val_losses):
    """繪製訓練和驗證損失曲線"""
    plt.figure(figsize=(12, 5))
    
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    plt.title('Training vs Validation Loss Curve')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png', dpi=150)
    print("Loss 曲線已保存為 loss_curve.png")
    plt.show()


# ============================================================
# 6. 預測函數
# ============================================================

def predict_image(model, image_path, transform, cat_map, color_map, device):
    """預測單張圖片的分類和顏色"""
    model.eval()
    
    try:
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            p_cat, p_col = model(input_tensor)
            cat_res = cat_map[torch.argmax(p_cat).item()]
            col_res = color_map[torch.argmax(p_col).item()]

        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{col_res} {cat_res}")
        plt.show()
        
        print(f"預測結果: {col_res} {cat_res}")
    except FileNotFoundError:
        print(f"圖片不存在: {image_path}")


# ============================================================
# 7. 主程序
# ============================================================

def main():
    """主函數"""
    # 檢查設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用裝置: {device}")
    if device.type == "cuda":
        print(f"   GPU 型號: {torch.cuda.get_device_name(0)}")
    
    # 下載資料
    setup_kaggle_and_download()
    
    # 準備資料
    filtered_df, cat_map, color_map = load_and_prepare_data()
    
    # 定義轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 建立模型
    model = MultiHeadResNet(len(cat_map), len(color_map)).to(device)
    
    # 檢查是否有已保存的模型
    if os.path.exists('model_weights.pth'):
        print("✅ 找到已訓練的模型，直接載入...")
        model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    else:
        print("🔄 未找到模型，開始訓練...")
        
        # 準備 DataLoader
        full_dataset = ClothingDataset(filtered_df, transform=transform)
        
        # 分割訓練集和驗證集 (80:20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"訓練集: {train_size}, 驗證集: {val_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 訓練設定
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 訓練模型
        train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=1)
        
        # 可視化損失
        plot_loss_curve(train_losses, val_losses)
        
        # 保存模型
        torch.save(model.state_dict(), 'model_weights.pth')
        print("模型已保存為 model_weights.pth")
    
    # 選擇性: 測試預測 (需要提供圖片路徑)
    predict_image(model, "input/red_sweater.jpg", transform, cat_map, color_map, device)


if __name__ == "__main__":
    main()
