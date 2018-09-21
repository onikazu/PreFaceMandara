import dlib

import face_recognition
import json
import sys
import glob
import pickle

trimmed_image_path = "./trimmed_face"
trimmed_images = glob.glob(trimmed_image_path + "/*")

vector_images = []

for image in trimmed_images:
    image = face_recognition.load_image_file(image)
    face_encoding = face_recognition.face_encodings(image)[0]
    vector_images.append(json.dumps(face_encoding.tolist()))

with open('sample.pickle', mode='wb') as f:
    pickle.dump(vector_images, f)

