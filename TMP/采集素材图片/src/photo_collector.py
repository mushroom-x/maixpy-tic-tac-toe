'''
OpenCV从USB摄像头读入图片,通过神经网络模型实时的对当前的棋子进行分类.
'''
import cv2
import numpy as np
import keras
from keras.models import load_model

# 初始化Capture
cap = cv2.VideoCapture(1)
cv2.namedWindow('image_win',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow('binary',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

img_count = 0
while(True):
    ## 逐帧获取画面
    # 如果画面读取成功 ret=True，frame是读取到的图片对象(numpy的ndarray格式)
    ret, frame = cap.read()

    if not ret:
        # 如果图片没有读取成功
        print("图像获取失败，请按照说明进行问题排查")
        break
    
    ## 如果c键按下，则进行图片保存
    # binary = cv2.inRange(frame, 0, 150)
    # img_resize = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_LINEAR)
    # img_resize = binary
    img_resize = 255 - frame
    # 更新窗口“image_win”中的图片
    cv2.imshow('image_win', frame)
    cv2.imshow('binary', img_resize)
    # 等待按键事件发生 等待1ms
    key = cv2.waitKey(1)
    if key == ord('q'):
        # 如果按键为q 代表quit 退出程序
        print("程序正常退出...Bye 不要想我哦")
        break
    elif key == ord('c'):
        
        # 写入图片 并命名图片为 图片序号.png
        cv2.imwrite("test_{}.png".format(img_count), img_resize)
        print("截图，并保存为  {}.png".format(img_count))
        # 图片编号计数自增1
        img_count += 1

# 释放VideoCapture
cap.release()
# 销毁所有的窗口
cv2.destroyAllWindows()
