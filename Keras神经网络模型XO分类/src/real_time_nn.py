'''
OpenCV从USB摄像头读入图片,通过神经网络模型实时的对当前的棋子进行分类.
'''
import cv2
import numpy as np
import keras
from keras.models import load_model

# 初始化Capture
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cv2.namedWindow('image_win',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow('image_resize',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

# 载入keras模型
nn_model = load_model('../../common/nn_model.h5')
# label与字母之间的映射
label_letter_map = ['O', 'X']

def img_bin2vect(img_bin):
    '''28x28的二值化图像转换为784的向量'''
    tmp_img = img_bin.T
    vect = tmp_img.reshape(784).astype('float32')/255.0
    print('Shape : vect.shape: {}'.format(vect.shape))
    return vect

while(True):
    ## 逐帧获取画面
    # 如果画面读取成功 ret=True，frame是读取到的图片对象(numpy的ndarray格式)
    ret, frame = cap.read()

    if not ret:
        # 如果图片没有读取成功
        print("图像获取失败，请按照说明进行问题排查")
        break


    # 画布
    canvas = np.copy(frame)
    # 将BGR彩图变换为灰度图
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 图像缩放为28x28
    frame_resize = cv2.resize(frame, (28, 28))
    # 图像二值化
    frame_binary = cv2.inRange(frame_resize, 0, 125)
    # 转换为向量
    result = nn_model.predict(np.array([img_bin2vect(frame_binary)]))[0]
    # 判断所属类别
    label = np.argmax(result)
    letter = label_letter_map[label]
    print('识别结果: {}'.format(letter))
    # 画布上绘制字符
    cv2.putText(canvas, text="Letter: {}".format(letter), org=(50, 100), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=3, 
        thickness=2,
        lineType=cv2.LINE_AA, 
        color=(0, 0, 255))
    
    # 更新窗口“image_win”中的图片
    cv2.imshow('image_win', canvas)
    cv2.imshow('image_resize', frame_binary)
    # 等待按键事件发生 等待1ms
    key = cv2.waitKey(1)
    if key == ord('q'):
        # 如果按键为q 代表quit 退出程序
        print("程序正常退出...Bye 不要想我哦")
        break
    elif key == ord('c'):
        ## 如果c键按下，则进行图片保存
        # 写入图片 并命名图片为 图片序号.png
        cv2.imwrite("{}.png".format(img_count), frame)
        print("截图，并保存为  {}.png".format(img_count))
        # 图片编号计数自增1
        img_count += 1

# 释放VideoCapture
cap.release()
# 销毁所有的窗口
cv2.destroyAllWindows()