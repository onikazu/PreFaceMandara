import dlib

import face_recognition
import json
import sys
import glob
import pickle

trimmed_image_path = "./database/img_align_celeba"
trimmed_images = glob.glob(trimmed_image_path + "/*.jpg")
print(trimmed_images)


vector_images = {}

for image_file in trimmed_images:
    image = face_recognition.load_image_file(image_file)

    face_encoding = face_recognition.face_encodings(image)[0]
    vector_images[image_file.split("/")[-1]] = face_encoding.tolist()

with open('data.pickle', mode='wb') as f:
    pickle.dump(vector_images, f)

print("finished")
