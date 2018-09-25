
import dlib


import pickle
import numpy as np
import math
from PIL import Image
import cv2


PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


# カメラデバイスを取得
c = cv2.VideoCapture(0)
# readで画像をキャプチャ、imgにRGBのデータが入ってくる
r, img = c.read()


def get_distance(a, b):
    distance = 0
    for i in range(len(a)):
        distance += (a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2
    return math.sqrt(distance)

pic_image_path = "./target_face/photo.jpeg"

with open('data.pickle', mode='rb') as f:
    datas = pickle.load(f)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dets, scores, idx = detector.run(img, 0)

rects = detector(img_rgb, 1)

landmark = []
cap_landmark = []
for rect in rects:
    cap_landmark = np.matrix([[p.x, p.y] for p in PREDICTOR(img_rgb, rect).parts()])
cap_landmark = list(cap_landmark)

with open('data.pickle', mode='rb') as f:
    datas = pickle.load(f)

similar_vec = []
similar_path = ""
similar_distance = 0

print(datas)

i = 0
for k in datas:
    distance = get_distance(datas[k], list(cap_landmark))
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