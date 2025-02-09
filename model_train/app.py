import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import fontManager
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import os
import cv2
import random
import pandas as pd

# 設定資料集路徑
train_dataset_path = r'C:\Users\user\Desktop\cnn\0114\results_0121_all_82_b100\dataset_new\train'
validation_data_path = r'C:\Users\user\Desktop\cnn\0114\results_0121_all_91_b100\dataset_new\val'

# train_dataset_path = r'C:\Users\user\Desktop\cnn\needbackup\qq\train_data'
# validation_data_path = r'C:\Users\user\Desktop\cnn\needbackup\qq\validation_data'

# train_dataset_path = r'C:\Users\user\Desktop\cnn\needbackup\enhance_denoise_resize_\train_data'
# validation_data_path = r'C:\Users\user\Desktop\cnn\needbackup\enhance_denoise_resize_\validation_data'



# 圖片保存路徑
output_dir = r'C:\Users\user\Desktop\cnn\0114\results_0121_all_82_b100'
os.makedirs(output_dir, exist_ok=True)  # 確保保存目錄存在

# 指定目錄路徑
directory = r'C:\Users\user\Desktop\cnn\0114\results_0121_all_82_b100\dataset\1'
# 獲取目錄中所有圖片的檔案名稱
all_images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
# 隨機選擇一張圖片
random_image = random.choice(all_images)
# 建立圖片的完整路徑
image_path = os.path.join(directory, random_image)
# 讀取圖片
test_img = cv2.imread(image_path)
# 顯示隨機選取的圖片名稱
print(f"隨機選取的圖片是：{random_image}")


# 讀取圖片資料集，保持原始大小
batch_size = 100
img_height = test_img.shape[0] // 5  # 根據需要調整圖片高度
img_width = test_img.shape[1] // 5  # 根據需要調整圖片寬度
num_classes = 5

# epochs
epochs = 10000

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dataset_path,
    # seed=12173,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    validation_data_path,
    # seed=12173,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
)


# 讀取訓練集的類別標籤
train_labels = []
for _, labels in train_ds:
    train_labels.extend(labels.numpy())

# 計算類別權重
class_weights = compute_class_weight(
    'balanced', classes=np.unique(train_labels), y=train_labels
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

print(class_weight_dict)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_dir, "best_model.h5"),
    monitor='val_loss',
    save_best_only=True
)


# 數據增強方法
data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # 隨機翻轉
    # tf.keras.layers.RandomRotation(0.2),                   # 隨機旋轉
    tf.keras.layers.RandomZoom(0.1),                       # 隨機縮放
    tf.keras.layers.RandomTranslation(0.1, 0.1),           # 隨機平移
    # tf.keras.layers.RandomContrast(0.1),                   # 隨機對比度調整
    # tf.keras.layers.RandomBrightness(0.1)                  # 隨機亮度調整
])

# 應用數據增強於訓練集
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# 正規化圖片
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))



# 根據需要調整模型結構
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 1)),  # Define input shape (height, width, channels)
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='elu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='elu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='elu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='elu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='elu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='elu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(640, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(300, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), # 12
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()
# for idx, layer in enumerate(model.layers):
#     config = layer.get_config()  # 包含层的详细配置信息
#     input_shape = layer.input_shape
#     output_shape = layer.output_shape
#     print(f"Layer {idx}: {layer.name}")
#     print(f"  Type: {layer.__class__.__name__}")
#     print(f"  Input shape: {input_shape}")
#     print(f"  Output shape: {output_shape}")
#     print()


# 編譯模型
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# 學習率調整回調
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)


# 自定义回调类：记录损失和准确率，并保存到 Excel
class MetricsHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_path):
        super(MetricsHistoryCallback, self).__init__()
        self.output_path = output_path
        self.history = {
            "epoch": [],
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": []
        }

    def on_epoch_end(self, epoch, logs=None):
        # 保存当前 epoch 的日志
        self.history["epoch"].append(epoch + 1)
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))
        self.history["accuracy"].append(logs.get("accuracy"))
        self.history["val_accuracy"].append(logs.get("val_accuracy"))

        # 将数据写入 Excel 文件
        df = pd.DataFrame(self.history)
        df.to_excel(self.output_path, index=False)
        print(f"Epoch {epoch + 1}: 数据已保存到 {self.output_path}")


# 保存 best_model 的信息到 Excel
class SaveBestModelInfoCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_path, monitor="val_accuracy"):
        super(SaveBestModelInfoCallback, self).__init__()
        self.output_path = output_path
        self.monitor = monitor
        self.best_value = -float("inf")  # 初始化为负无穷
        self.best_epoch = None
        self.best_logs = None

    def on_epoch_end(self, epoch, logs=None):
        # 检查当前 epoch 是否是最佳
        current_value = logs.get(self.monitor)
        if current_value is not None and current_value > self.best_value:
            self.best_value = current_value
            self.best_epoch = epoch + 1
            self.best_logs = logs.copy()  # 复制最佳 epoch 的日志

            # 保存到 Excel
            best_model_info = {
                "Epoch": [self.best_epoch],
                "Best Accuracy": [self.best_logs.get("accuracy")],
                "Best Validation Accuracy": [self.best_logs.get("val_accuracy")]
            }
            df = pd.DataFrame(best_model_info)
            df.to_excel(self.output_path, index=False)
            print(f"Best model updated at Epoch {self.best_epoch}, 信息已保存到 {self.output_path}")

# 路径设置
loss_history_path = os.path.join(output_dir, "loss_history.xlsx")
best_model_info_path = os.path.join(output_dir, "best_model_info.xlsx")

# 回调实例化
loss_history_callback = MetricsHistoryCallback(loss_history_path)
save_best_model_callback = SaveBestModelInfoCallback(best_model_info_path)


class ErrorTrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, output_dir):
        super(ErrorTrackingCallback, self).__init__()
        self.val_ds = val_ds
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        for batch_idx, (images, labels) in enumerate(self.val_ds):
            predictions = np.argmax(self.model.predict(images), axis=1)
            for i in range(len(labels)):
                if predictions[i] != labels[i]:  # 錯誤分類
                    image_save_path = os.path.join(
                        self.output_dir, 
                        f"epoch_{epoch+1}_batch_{batch_idx}_img_{i}_true_{labels[i].numpy()+1}_pred_{predictions[i]+1}.png"
                    )
                    plt.imsave(image_save_path, images[i].numpy().squeeze(), cmap='gray')


# 自定義回調以計算每個類別的準確率
class ClassAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, num_classes):
        super(ClassAccuracyCallback, self).__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes
        self.class_accuracy = {i: [] for i in range(num_classes)}

    def on_epoch_end(self, epoch, logs=None):
        total_correct = np.zeros(self.num_classes)
        total_samples = np.zeros(self.num_classes)

        for images, labels in self.val_ds:
            predictions = np.argmax(self.model.predict(images), axis=1)
            for i in range(len(labels)):
                total_samples[labels[i]] += 1
                if predictions[i] == labels[i]:
                    total_correct[labels[i]] += 1

        for i in range(self.num_classes):
            if total_samples[i] > 0:
                self.class_accuracy[i].append(total_correct[i] / total_samples[i])
            else:
                self.class_accuracy[i].append(0)

class ValAccuracyStoppingCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy):
        super(ValAccuracyStoppingCallback, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get("val_accuracy")
        if val_accuracy is not None and val_accuracy >= self.target_accuracy:
            print(f"\n驗證準確率達到目標值 {self.target_accuracy * 100:.2f}%，停止訓練。")
            self.model.stop_training = True

# 添加回調並設置目標驗證準確率
target_accuracy = 0.98 # 假設目標是 95% 的驗證準確率


# 實例化回調
class_accuracy_callback = ClassAccuracyCallback(val_ds, num_classes)

# error_tracking_callback = ErrorTrackingCallback(val_ds, os.path.join(output_dir, "error_images"))

val_accuracy_stopping_callback = ValAccuracyStoppingCallback(target_accuracy)



# 訓練模型
# history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[val_accuracy_stopping_callback, class_accuracy_callback, error_tracking_callback])
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[class_accuracy_callback, checkpoint_callback, loss_history_callback, save_best_model_callback], class_weight=class_weight_dict)

from matplotlib import font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 添加自定義字型
font_path = r'C:\Users\user\Desktop\cnn\1127_test_test_test\ChineseFont.ttf' 
fm.fontManager.addfont(font_path)
mpl.rc('font', family='ChineseFont')  # 設定為自定義字型

# 混淆矩陣計算並保存
def plot_and_save_confusion_matrix(val_ds, model, output_path):
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        predictions = np.argmax(model.predict(images), axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predictions)
    
    y_true = [label + 1 for label in y_true]
    y_pred = [pred + 1 for pred in y_pred]
    
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存圖片
    plt.close()

plot_and_save_confusion_matrix(val_ds, model, os.path.join(output_dir, 'confusion_matrix.png'))

# 損失圖保存
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='訓練損失值')
plt.plot(history.history['val_loss'], label='驗證損失值')
plt.title('損失隨著迭代次數的變化')
plt.xlabel('迭代次數')
plt.ylabel('損失')
plt.xticks(np.arange(0, len(history.history['loss']), step=10))
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 準確率圖保存
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='訓練準確率')
plt.plot(history.history['val_accuracy'], label='驗證準確率')
plt.title('準確率隨著迭代次數的變化')
plt.xlabel('迭代次數')
plt.ylabel('準確率')
plt.xticks(np.arange(0, len(history.history['accuracy']), step=10))
plt.legend()
plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 各類別準確率保存
plt.figure(figsize=(12, 8))
for i in range(num_classes):
    plt.plot(class_accuracy_callback.class_accuracy[i], label=f'類別 {i + 1} 準確率')
plt.title('各類別準確率隨著迭代次數的變化')
plt.xlabel('迭代次數')
plt.ylabel('準確率')
plt.xticks(np.arange(1, len(class_accuracy_callback.class_accuracy[0]) + 1, step=10))
plt.legend()
plt.savefig(os.path.join(output_dir, 'class_accuracy_curve.png'), bbox_inches='tight')
plt.close()

# 儲存模型
model.save(os.path.join(output_dir,'0120.h5'))