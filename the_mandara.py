import cv2
import face_recognition
import pickle
import math
import dlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    cap = cv2.VideoCapture(1)  # 1はカメラのデバイス番号
    while True:
        ret, frame = cap.read()

        # 顔認識
        detector = dlib.get_frontal_face_detector()
        rects = detector(frame, 1)

        # 切り出し
        if rects == None:
            continue

        # 画像の保存
        for x, rect in enumerate(rects):
            dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
            print(dst)
            print(rect.top())
            cv2.imwrite('./target_face/face{}.jpg'.format(x), dst)
            plt.imshow(dst)



if __name__ == '__main__':
    main()
