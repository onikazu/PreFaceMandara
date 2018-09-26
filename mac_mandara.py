import cv2
import face_recognition
import pickle
import math
import dlib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def capture_camera1(mirror=True, size=None):
    """
    ウェブカメラで顔を認識したら自動でシャッターを切り、保存する
    :param mirror: boolean
    :param size: tuple
    :return: None
    """
    rects = 0
    frame = 0
    cap = cv2.VideoCapture(1)  # 1はカメラのデバイス番号
    while not rects:
        ret, frame = cap.read()

        # 画像を左右反転させる
        if mirror is True:
            frame = frame[:, ::-1]

        # 顔が写っているかどうかのチェック
        detector = dlib.get_frontal_face_detector()
        rects = detector(frame, 1)

        # フレームをリサイズ
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

    # フレームを表示する
    cv2.imshow('camera capture', frame)
    # save image
    cv2.imwrite('./target_face/camera.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()


def capture_camera2(mirror=True, size=None):
    """
    escボタンでシャッターを切り、画像を保存する
    :param mirror: boolean
    :param size: tuple
    :return: None
    """
    cap = cv2.VideoCapture(1)  # 1はカメラのデバイス番号
    while True:
        ret, frame = cap.read()

        # 鏡のように映るか否か
        if mirror is True:
            frame = frame[:, ::-1]
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(1)
        if k == 27:
            # save image
            cv2.imwrite('./target_face/camera.jpg', frame)
            cap.release()
            cv2.destroyAllWindows()
            break


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


def exe_mandara():
    pic_image_path = "./target_face/camera.jpg"

    with open('data.pickle', mode='rb') as f:
        datas = pickle.load(f)

    pic_image = face_recognition.load_image_file(pic_image_path)
    pic_image_encoded = face_recognition.face_encodings(pic_image)[0]

    similar_vec = []
    similar_path = ""
    similar_distance = 0

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
    animation(similar_path, pic_image_path)


def animation(similar_path, pic_image_path):
    """
    結果をアニメーションにして表示する（開発途中）
    :return:
    """
    # 結果表示
    picList = ["./database/{}".format(similar_path), "./target_face/{}".format(pic_image_path.split("/")[-1])]
    fig = plt.figure()
    ims = []
    for i in range(len(picList)):
        tmp = Image.open(picList[i])
        print(tmp)
        ims.append([plt.imshow(tmp)])

    # アニメーション作成
    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)
    ani.save("./test.gif")


if __name__ == "__main__":
    capture_camera2()
    exe_mandara()