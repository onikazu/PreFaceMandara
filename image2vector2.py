import dlib

import face_recognition
import json
import sys
import glob
import pickle
import cv2
import numpy as np


# for add in
PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

database_path = "./database"
database_images = glob.glob(database_path + "/*")
detector = dlib.get_frontal_face_detector()

print(database_images)

vector_images = {}

for image_paths in database_images:
    image = cv2.imread(image_paths, cv2.IMREAD_COLOR)
    rects = detector(image, 1)
    for rect in rects:
        landmarks = np.matrix(
            [[p.x, p.y] for p in PREDICTOR(image, rect).parts()]
        )
        vector_images[image_paths.split("/")[-1]] = landmarks

with open('data.pickle', mode='wb') as f:
    pickle.dump(vector_images, f)
