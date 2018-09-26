import cv2
import face_recognition
import pickle
import math
import dlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob


def get_distance(a, b):
    """
    画像間の類似度を測定する
    :param a: list
    :param b: list
    :return: float
    """
    distance = 0
    for i in range(len(a)):
        distance += (a[i] - b[i]) ** 2
    return math.sqrt(distance)

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

        # 顔写真の読み込み
        target_image_paths = glob.glob("./target_face/face*.jpg")

        # databaseの読み込み
        with open('data.pickle', mode='rb') as f:
            datas = pickle.load(f)

        # 判定
        for target_image_path in target_image_paths:
            target_image = face_recognition.load_image_file(target_image_path)
            target_image_encoded = face_recognition.face_encodings(target_image)[0]

            similar_vec = []
            similar_path = []
            similar_distance = []

            i = 0
            for k in datas:
                distance = get_distance(datas[k], list(target_image_encoded))
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
            im2 = Image.open("./target_face/{}".format(target_image_paths.split("/")[-1]))

            im1.show()
            im2.show()




if __name__ == '__main__':
    main()
