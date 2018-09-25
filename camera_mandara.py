
import dlib

import face_recognition
import json
import sys
import glob
import pickle
import numpy as np
import math
from PIL import Image
import cv2
import time


# OpenCVをインポート
import cv2
# カメラデバイスを取得
c = cv2.VideoCapture(0)
# readで画像をキャプチャ、imgにRGBのデータが入ってくる
r, img = c.read()
# 保存
cv2.imwrite('./target_face/photo.jpg', img)

def get_distance(a, b):
    distance = 0
    for i in range(len(a)):
        distance += (a[i] - b[i]) ** 2
    return math.sqrt(distance)


pic_image_path = "./target_face/jobjob.jpeg"

with open('data.pickle', mode='rb') as f:
    datas = pickle.load(f)

pic_image = face_recognition.load_image_file(pic_image_path)
pic_image_encoded = face_recognition.face_encodings(pic_image)[0]

similar_vec = []
similar_path = ""
similar_distance = 0

print(datas)

i = 0
for k in datas:
    distance = get_distance(datas[k], list(pic_image_encoded))
    if i == 0:
        similar_distance = distance
        similar_path = k
        similar_vec = datas[k]
    else:
        if similar_distance > distance:
            similar_distance = distance
            similar_path = k
            similar_vec = datas[k]

    print("{0}:{1}".format(k, distance))
    i += 1


print("似ているのは{}！！！".format(similar_path))
im1 = Image.open("./database/{}".format(similar_path))
im2 = Image.open("./target_face/{}".format(pic_image_path.split("/")[-1]))

im1.show()
im2.show()