import cv2
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

current_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(current_path)

model_path = r'cnn_real_p85.h5'
model = tf.keras.models.load_model(model_path)
input_dir = r'D:\4000x3000_CNN\prototype_AI\PPT_MODEL_1_P85\set1'

ps2 = np.array([[252, 1156], [284, 1276], [316, 1364], [364, 1468], 
                [404, 1524], [452, 1596], [492, 1644], [540, 1692], 
                [580, 1740], [620, 1772], [668, 1820], [708, 1852], 
                [740, 1876], [804, 1924], [860, 1956], [908, 1988], 
                [900, 2020], [1052, 2844], [468, 2996], [420, 2996]])

# 以最小外接矩陣所找到的 ROI
def get_roi(image, ps):
    # 最小邊界矩形並擷取 ROI
    # x1, y1, w1, h1 = cv2.boundingRect(pts1)
    # roi1 = image[y1:y1 + h1, x1:x1 + w1]
    x2, y2, w2, h2 = cv2.boundingRect(ps)
    roi = image[y2:y2 + h2, x2:x2 + w2]
    print(f'ps = ({x2}, {y2}), ({x2}, {y2 + h2}), ({x2 + w2}, {y2 + h2}), ({x2 + w2}, {y2})')
    
    # 儲存 R2 圖
    r2_dir = os.path.join(os.getcwd(), 'r2')
    if not os.path.exists(r2_dir):
        os.makedirs(r2_dir)
    r2_path = os.path.join(r2_dir, 'roi2.png')
    cv2.imwrite(r2_path, roi)
    
    return roi

def mask_roi(img, ps):
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [ps], 255)
    roi = cv2.bitwise_and(img, img, mask=mask)
    # if len(roi.shape) > 2:
        # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return roi

def pre_proc(img):
    # ROI
    roi2 = mask_roi(img, ps2)
    roi2 = get_roi(roi2, ps2)
    # 轉灰階
    g2 = roi2
    if len(g2.shape) == 3:
        g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))    
    enhanced2 = clahe.apply(g2)
    return enhanced2

def predict(img):
    # 圖片格式轉換
    img = cv2.resize(img, (149, 348))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 標準化處理
    
    # 預測類別
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predictions = predictions[0]
    predicted_class = predicted_class[0] + 1
    print(f'Predicted probability: {predictions}')
    print(f'Predicted class: {predicted_class}')
    return predicted_class, predictions

def save_predict(image, filename, predicted_class, predictions):
    # 建立目錄，如果不存在
    def create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 保存到類別目錄
    base_dir = os.path.join(os.getcwd(), str(predicted_class))
    create_dir(base_dir)
    class_filename = os.path.join(base_dir, f"{os.path.splitext(filename)[0]}.jpg")
    cv2.imwrite(class_filename, image)
    print(f'圖片已保存到類別目錄: {class_filename}')
    
    # 如果是 confusion 狀態，另外保存到 confusion 目錄
    if np.max(predictions) < 0.8:
        confusion_dir = os.path.join(os.getcwd(), "confusion", str(predicted_class))
        create_dir(confusion_dir)
        confusion_filename = os.path.join(confusion_dir, f"{os.path.splitext(filename)[0]}.jpg")
        cv2.imwrite(confusion_filename, image)
        print(f'圖片也保存到 confusion 目錄: {confusion_filename}')
    

def load_pic(input_dir):
    # 讀取目錄中所有圖片
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', 'jpeg')):
            file_path = os.path.join(input_dir, filename)
            # 讀取讀片
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Can't read the file: {file_path}")
                continue
            # roi2 預處理
            r2 = pre_proc(img)
            
            # 預測
            predicted_class, predictions = predict(r2)
            
            # 保存預測結果
            save_predict(img, filename, predicted_class, predictions)


if __name__ == '__main__':
    import time
    # while True:
    #     predict()
    #     time.sleep(60)
    load_pic(input_dir)