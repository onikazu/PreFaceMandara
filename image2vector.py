import dlib

import face_recognition
import json
import sys
import glob
import pickle

trimmed_image_path = "./database/img_align_celeba"
trimmed_images = glob.glob(trimmed_image_path + "/*.jpg")

vector_images = {}

# 辞書型のデータを作る
for image_file in trimmed_images:
    image = face_recognition.load_image_file(image_file)
    # 顔認識
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 1)
    print(rects)
    print(rects[0])
    print(rects.__bool__)
    # 顔認識してい無いとき

    if not rects:
        continue

    print(len(face_recognition.face_encodings(image)[0]))
    face_encoding = face_recognition.face_encodings(image)[0]
    vector_images[image_file.split("/")[-1]] = face_encoding.tolist()

with open('data.pickle', mode='wb') as f:
    pickle.dump(vector_images, f)

print("finished")
