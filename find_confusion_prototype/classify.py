from datetime import datetime
import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
import pandas as pd

cnt = 0
stop_flag = False  # 全局停止旗標
model = None

# 儲存預測結果的列表
prediction_data = []
confusion_data = []  # 儲存混淆樣本的列表
prediction_error_data = []  # 儲存預測錯誤樣本的列表
actual_vs_predicted = []

def load_model(model_path):
    global model
    if model is None:
        model = tf.keras.models.load_model(model_path)

# 建立目錄，如果不存在
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(image, predicted_class, original_class, filename, predictions, is_confusion, is_predict_error):
    # 根據不同情況選擇目錄
    if is_confusion:
        base_dir = os.path.join(os.getcwd(), "confusion", str(predicted_class))
    elif is_predict_error:
        base_dir = os.path.join(os.getcwd(), "predict_error", str(predicted_class))
    else:
        base_dir = os.path.join(os.getcwd(), str(predicted_class))

    create_dir(base_dir)
    save_path = os.path.join(base_dir, filename)  # 使用原始檔名
    cv2.imwrite(save_path, image)
    print(f'圖片已保存到: {save_path}')

    # 將預測結果儲存至列表
    result = {
        'Filename': filename,
        'Original Class': original_class,
        'Predicted Class': predicted_class,
        'Predicted Probability': np.max(predictions),
        'Prediction_1': predictions[0][0],
        'Prediction_2': predictions[0][1],
        'Prediction_3': predictions[0][2],
        'Prediction_4': predictions[0][3],
        'Prediction_5': predictions[0][4],
    }
    prediction_data.append(result)

    # 根據判斷分類數據
    if is_confusion:
        confusion_data.append(result)
    if is_predict_error:
        prediction_error_data.append(result)
        
    actual_vs_predicted.append((original_class, predicted_class))  # 儲存實際與預測標籤

# 儲存為 Excel 文件的函式
def export_logs_to_excel():
    # 將預測結果列表儲存到 Excel
    with pd.ExcelWriter("prediction_logs.xlsx") as writer:
        pd.DataFrame(prediction_data).to_excel(writer, sheet_name="Prediction Log", index=False)
        pd.DataFrame(confusion_data).to_excel(writer, sheet_name="Confusion", index=False)
        pd.DataFrame(prediction_error_data).to_excel(writer, sheet_name="Prediction Error", index=False)

    print("日誌已保存至 prediction_logs.xlsx")
    
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix():
    # 提取實際和預測標籤
    y_true = [pair[0] for pair in actual_vs_predicted]
    y_pred = [pair[1] for pair in actual_vs_predicted]

    # 計算混淆矩陣
    labels = list(range(1, 6))  # 假設有 5 個類別
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 可視化混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')

    # 儲存圖像
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


# 預測圖片類別
def predict(img, original_class, tag=''):
    model_path = r'0116.h5'
    load_model(model_path)

    # 圖片格式轉換
    img = image.img_to_array(img.convert('L'))
    img = np.expand_dims(img, axis=-1)
    img_array = np.expand_dims(img, axis=0)
    img_array /= 255.0

    # 預測類別
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0] + 1
    prediction_confidence = np.max(predictions)

    print(f'{tag} Predicted probability: {predictions[0]}')
    print(f'{tag} Predicted class: {predicted_class}')

    return predicted_class, predictions, prediction_confidence

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_path)
    for original_class in range(1, 6):  # 遍歷原始資料夾
        source_dir = os.path.join(current_path, str(original_class))
        for filename in os.listdir(source_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(source_dir, filename)
                img = image.load_img(img_path)  # 調整大小符合模型輸入

                predicted_class, predictions, confidence = predict(img, original_class, tag=filename)

                # 判斷是否為混淆樣本或預測錯誤
                is_confusion = confidence < 0.8
                is_predict_error = predicted_class != original_class

                # 保存圖片
                save_image(cv2.imread(img_path), predicted_class, original_class, filename, predictions, is_confusion, is_predict_error)
    # 匯出日誌至 Excel
    export_logs_to_excel()
    plot_confusion_matrix()  # 新增這行呼叫混淆矩陣繪製函式
    