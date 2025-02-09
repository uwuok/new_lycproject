import time

import cv2
from datetime import datetime
import os

# 初始化相機
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
cnt = 0
stop_flag = False  # 全局停止旗標

def take_picture():
    if not cap.isOpened() or stop_flag:
        print("停止拍攝或無法抓取鏡頭")
        return

    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{current_dir}\photo_{timestamp}.jpg"

        global cnt
        cnt += 1
        cv2.imwrite(filename, frame)
        print(f"已拍攝照片並保存為： {filename} [{cnt}]")
    else:
        print("無法讀取鏡頭 frame")


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        take_picture()
        time.sleep(10)
