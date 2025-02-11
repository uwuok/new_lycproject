import cv2
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import mysql.connector


current_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(current_path)

model_path = r'C:\Users\user\Desktop\cnc_gui_master\method_AI\best_model.h5'
model = tf.keras.models.load_model(model_path)
input_dir = r"C:\Users\user\Desktop\cnc_gui_master\method_AI\origin"


ps1 = np.array([[152, 196], [648, 244], [548, 300], [496, 344],
                [428, 408], [380, 472], [336, 544], [304, 616], 
                [268, 692], [248, 800], [204, 844]])

ps2 = np.array([[252, 1156], [284, 1276], [316, 1364], [364, 1468],
                [404, 1524], [452, 1596], [492, 1644], [540, 1692],
                [580, 1740], [620, 1772], [668, 1820], [708, 1852],
                [740, 1876], [804, 1924], [860, 1956], [908, 1988],
                [900, 2020], [1052, 2844], [468, 2996], [420, 2996]])

# 以最小外接矩陣所找到的 ROI


def update_sql_r2(flusher_level_result_value):
    # 連接到 MySQL 資料庫
    db_connection = mysql.connector.connect(
        host="localhost",  # MySQL 伺服器地址
        user="root",       # 使用者名稱
        password="ncut2024",  # 密碼
        database="cnc_db",  # 資料庫名稱
        auth_plugin="mysql_native_password"
    )

    # 創建一個游標對象
    cursor = db_connection.cursor()

    # 要插入的整數值
    # flusher_level_result_value = 10  # 假設要插入 10

    # 插入資料到 `Level_result` 表格中的 `flusher_level_result`
    query = """
    UPDATE level_result
    SET flusher_level_result = %s
    """

    # 插入資料的參數 (可以根據需求修改 excluder_level_result)
    values = (flusher_level_result_value, )  

    # 執行插入操作
    cursor.execute(query, values)

    # 提交交易
    db_connection.commit()

    # 顯示操作結果
    print(f"update record id 1 with flusher_level_result_value: {flusher_level_result_value}")

    # 關閉游標和資料庫連接
    cursor.close()
    db_connection.close()


import os
import cv2

def get_roi(image, ps, n):
    # 計算最小邊界矩形並擷取 ROI
    x, y, w, h = cv2.boundingRect(ps)
    roi = image[y:y + h, x:x + w]
    print(f'ps = ({x}, {y}), ({x}, {y + h}), ({x + w}, {y + h}), ({x + w}, {y})')

    # 根據 n 的值設定目錄和檔案名稱
    if n == 1:
        dir_name = 'r1'
        file_name = 'roi1.png'
    elif n == 2:
        dir_name = 'r2'
        file_name = 'roi2.png'
    else:
        raise ValueError("n 必須是 1 或 2")

    # 建立目錄（如果不存在）
    dir_path = os.path.join(os.getcwd(), dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 儲存 ROI 圖片
    file_path = os.path.join(dir_path, file_name)
    cv2.imwrite(file_path, roi)

    # 如果 n 為 2，另存調整大小的 ROI 圖片
    if n == 1:
        resized_dir = os.path.join(os.getcwd(), "resized_r1")
        if not os.path.exists(resized_dir):
            os.makedirs(resized_dir)
        resized_file_path = os.path.join(resized_dir, 'resized_r1.jpg')
        cv2.imwrite(resized_file_path, roi)
    if n == 2:
        resized_dir = os.path.join(os.getcwd(), "resized_r2")
        if not os.path.exists(resized_dir):
            os.makedirs(resized_dir)
        resized_file_path = os.path.join(resized_dir, 'resized_r2.jpg')
        cv2.imwrite(resized_file_path, roi)

    return roi



def mask_roi(img, ps):
    # 找到多邊形點的最小外接矩形
    x, y, w, h = cv2.boundingRect(ps)
    
    # 創建遮罩
    mask_shape = img.shape[:2]
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [ps], 255)
    
    # 裁剪出遮罩範圍的圖像
    roi = cv2.bitwise_and(img, img, mask=mask)
    
    # 直接裁剪到最小矩形範圍
    roi = roi[y:y+h, x:x+w]
    
    return roi


def pre_proc(img):
    
    # roi1
    roi1 = get_roi(img, ps1, 1)
    roi1 = mask_roi(img, ps1)
    
    
    # ROI2 (mainly)
    roi2 = get_roi(img, ps2, 2)
    roi2 = mask_roi(img, ps2)
    
    
    # 轉灰階
    g1 = roi1
    g2 = roi2
    if len(g2.shape) == 3:
        g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    # enhanced2 = clahe.apply(g2)
    # return enhanced2
    return g1, g2


def predict(img):
    # 圖片格式轉換
    img = cv2.resize(img, (160, 368))
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

    base_dir = os.path.join(os.getcwd(), str(predicted_class))
    if np.max(predictions) < 0.8:
        base_dir = os.path.join(os.getcwd(), "confusion")
        base_dir = os.path.join(base_dir, str(predicted_class))

    create_dir(base_dir)  # 確保目錄存在
    filename = os.path.join(base_dir, f"{os.path.splitext(filename)[0]}.jpg")

    cv2.imwrite(filename, image)
    print(f'圖片已保存到: {filename}')


def load_pic(input_dir):
    # 讀取目錄中所有圖片
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', 'jpeg')):
            file_path = os.path.join(input_dir, filename)
            # 讀取讀片
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Can't read the file: {file_path}")
                continue
            # roi2 預處理
            r1, r2 = pre_proc(img)
            
            cv2.imwrite('r2.png', r2)

            # 預測
            predicted_class, predictions = predict(r2)

            # print(predictions)
            print(predicted_class)

            # 綁存預測結果(Json)
            # update_json('Flusher_level_bar_R1', int(-87))
            update_sql_r2(int(predicted_class))
            # 保存預測結果(圖片)
            save_predict(img, filename, predicted_class, predictions)


if __name__ == '__main__':
    import time
    # while True:
    #     predict()
    #     time.sleep(60)
    load_pic(input_dir)