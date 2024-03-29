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


def concat_tile(im_list_2d):
    """
    イメージをタイル状に敷き詰める
    :param im_list_2d: list(2d)
    :return:
    """
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def main():
    while True:
        cap = cv2.VideoCapture(0)  # 1はカメラのデバイス番号
        ret, frame = cap.read()
        # 顔認識
        detector = dlib.get_frontal_face_detector()
        rects = detector(frame, 1)

        # cv2.imshow("camera", frame)

        # 顔認識できなかったら
        if not rects:
            print("cant recognize faces")
            continue

        # 画像の保存
        dst = 0
        for x, rect in enumerate(rects):
            dst = frame[rect.top():rect.bottom(), rect.left():rect.right()]
            print(dst)
            cv2.imwrite('./target_face/face{}.jpg'.format(x), dst)
            # plt.imshow(dst)

        # 顔写真の読み込み
        target_image_paths = glob.glob("./target_face/face*.jpg")

        # databaseの読み込み
        with open('mini_data.pickle', mode='rb') as f:
            datas = pickle.load(f)

        # 判定
        print(target_image_paths)
        print("start")
        target_num = 0
        for target_image_path in target_image_paths:
            try:
                target_image = face_recognition.load_image_file(target_image_path)
                target_image_encoded = face_recognition.face_encodings(target_image)[0]
            except IndexError or OSError:
                print("error happened")
                break
            # 9個似ている顔を判定してやる
            similar_vecs = []
            similar_paths = []
            similar_distances = []
            i = 0
            for k in datas:
                distance = get_distance(datas[k], list(target_image_encoded))
                # 最初
                if i == 0:
                    similar_distances.append(distance)
                    similar_paths.append(k)
                    similar_vecs.append(datas[k])
                    i += 1
                for j in range(len(similar_distances)):
                    # 10個以上
                    if len(similar_distances) >= 10:
                        # より近い
                        if similar_distances[j] > distance:
                            similar_distances.insert(j, distance)
                            similar_paths.insert(j, k)
                            similar_vecs.insert(j, datas[k])
                            del similar_distances[-1]
                            del similar_paths[-1]
                            del similar_vecs[-1]
                            break
                    # 10個以下
                    else:
                        if similar_distances[j] > distance:
                            similar_distances.insert(j, distance)
                            similar_paths.insert(j, k)
                            similar_vecs.insert(j, datas[k])
                            break
                        if j == len(similar_distances)-1:
                            similar_distances.append(distance)
                            similar_paths.append(k)
                            similar_vecs.append(datas[k])

                print("{0}:{1}".format(k, distance))
                print("number{} is end".format(i))
                i += 1
            print("finish about one face")

            # im0 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[0]))
            # im1 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[1]))
            # im2 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[2]))
            # im3 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[3]))
            # im4 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[4]))
            # im5 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[5]))
            # im6 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[6]))
            # im7 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[7]))
            # im8 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[8]))
            # im9 = cv2.imread("./database/img_align_celeba/{}".format(similar_paths[9]))
            print(similar_paths)

            im0 = cv2.imread("./database/{}".format(similar_paths[0]))
            im1 = cv2.imread("./database/{}".format(similar_paths[1]))
            im2 = cv2.imread("./database/{}".format(similar_paths[2]))
            im3 = cv2.imread("./database/{}".format(similar_paths[3]))
            im4 = cv2.imread("./database/{}".format(similar_paths[4]))
            im5 = cv2.imread("./database/{}".format(similar_paths[5]))
            im6 = cv2.imread("./database/{}".format(similar_paths[6]))
            im7 = cv2.imread("./database/{}".format(similar_paths[7]))
            im8 = cv2.imread("./database/{}".format(similar_paths[8]))
            im9 = cv2.imread("./database/{}".format(similar_paths[9]))
            im_target = cv2.imread("./target_face/{}".format(target_image_paths[target_num].split("/")[-1]))
            im_target = cv2.resize(im_target, (178, 218))

            # 最も似ている画像について
            print("似ているのは{}！！！".format(similar_paths[0]))
            # cv2.imshow("most similar", im0)
            # cv2.imshow("target", im_target)

            im0_s = cv2.resize(im0, dsize=(0, 0), fx=0.5, fy=0.5)
            im1_s = cv2.resize(im1, dsize=(0, 0), fx=0.5, fy=0.5)
            im2_s = cv2.resize(im2, dsize=(0, 0), fx=0.5, fy=0.5)
            im3_s = cv2.resize(im3, dsize=(0, 0), fx=0.5, fy=0.5)
            im4_s = cv2.resize(im4, dsize=(0, 0), fx=0.5, fy=0.5)
            im5_s = cv2.resize(im5, dsize=(0, 0), fx=0.5, fy=0.5)
            im6_s = cv2.resize(im6, dsize=(0, 0), fx=0.5, fy=0.5)
            im7_s = cv2.resize(im7, dsize=(0, 0), fx=0.5, fy=0.5)
            im8_s = cv2.resize(im8, dsize=(0, 0), fx=0.5, fy=0.5)
            # im9_s = cv2.resize(im9, dsize=(0, 0), fx=0.5, fy=0.5)
            im_target_s = cv2.resize(im_target, dsize=(0, 0), fx=0.5, fy=0.5)
            im_tile = concat_tile([[im0_s, im1_s, im2_s],
                                   [im7_s, im_target_s, im3_s],
                                   [im6_s, im5_s, im4_s]])
            # cv2.imshow("tile", im_tile)
            target_num += 1
            cv2.imwrite('./opencv_concat_tile.jpg', im_tile)
            tile_image = Image.open('./opencv_concat_tile.jpg')
            tile_image.show()
            tile_image.close()
            cap.release()
            break


if __name__ == '__main__':
    main()
