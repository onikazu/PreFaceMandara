import dlib

import face_recognition
import json
import sys
import glob
import pickle
import numpy as np

pic_image = "./target_face/jobs2.jpg"

with open('sample.pickle', mode='rb') as f:
    datas = pickle.load(f)

pic_image = face_recognition.load_image_file(pic_image)
pic_image_encoded = face_recognition.face_encodings(pic_image)[0]

similar_vec = []
similar_num = 0
similar_distance = 0

for i in range(len(datas)):
    # datas[i] = np.array(datas[i])
    distance = face_recognition.face_distance(datas[i], list(pic_image_encoded))
    if i == 0:
        similar_distance = distance
        similar_num = i
        similar_vec = datas[i]
    else:
        if similar_distance > distance:
            similar_distance = distance
            similar_num = i
            similar_vec = datas[i]

print("似ているのは{}番の写真！！！".format(similar_num))


