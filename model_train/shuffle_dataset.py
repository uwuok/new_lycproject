import os
import shutil
import random

os.chdir(os.path.abspath(os.path.dirname(__file__)))

def split_dataset(dataset_path, output_path, max_per_class=60, train_ratio=0.8):
    # 確保輸出資料夾存在
    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    
    # 遍歷每個類別資料夾
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue
        
        # 讀取類別內所有檔案，並隨機排序
        files = os.listdir(category_path)
        random.shuffle(files)
        
        # 僅取前 max_per_class 的檔案
        files = files[:max_per_class]
        
        # 計算訓練與驗證集數量
        train_count = int(len(files) * train_ratio)
        
        # 分別建立類別資料夾
        train_category_path = os.path.join(train_path, category)
        val_category_path = os.path.join(val_path, category)
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(val_category_path, exist_ok=True)
        
        # 複製檔案到訓練集和驗證集
        for i, file in enumerate(files):
            src = os.path.join(category_path, file)
            if i < train_count:
                dst = os.path.join(train_category_path, file)
            else:
                dst = os.path.join(val_category_path, file)
            shutil.copy(src, dst)
    
    print(f"Dataset split complete. Train and val sets saved in: {output_path}")

# 使用範例
dataset_path = r'C:\Users\user\Desktop\cnn\0114\dataset'  # 原始資料集路徑
output_path = r'C:\Users\user\Desktop\cnn\0114\dataset_new'  # 輸出資料集路徑
split_dataset(dataset_path, output_path, max_per_class=40, train_ratio=0.8)
