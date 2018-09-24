"""
face_trimming.py
Trim face from images
"""

import glob
import cv2

# for cv2
cascade_path = "./haarcascade_frontalface_alt.xml"

# paths for the images
origin_image_path = "./origin_face"
origin_images = glob.glob(origin_image_path + "/*")
dir_path = "./trimmed_face"

cascade = cv2.CascadeClassifier(cascade_path)

for image in origin_images:
    image_name = image.split('/')[-1]
    image = cv2.imread(image, 0)
    facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))
    if len(facerect) > 0:
        for rect in facerect:
            # 顔だけ切り出して保存
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            dst = image[y:y + height, x:x + width]
            save_path = dir_path + '/' + 'trimmed(' + image_name + ').jpg'
            # 認識結果の保存
            cv2.imwrite(save_path, dst)
            print("save!")

print("finish")