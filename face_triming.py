"""
写真から顔のトリミングを行うプログラム
"""

import glob
import cv2

cascade_path = "./haarcascade_frontalface_alt.xml"


origin_image_path = "./origin_face"
origin_images = glob.glob(origin_image_path + "/*")
dir_path = "./trimmed_face"


i = 0

print(origin_images)



cascade = cv2.CascadeClassifier(cascade_path)

for image in origin_images:
    image = cv2.imread(image, 0)
    print(image)
    facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))
    if len(facerect) > 0:

        for rect in facerect:
            # 顔だけ切り出して保存
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            dst = image[y:y + height, x:x + width]
            save_path = dir_path + '/' + 'image(' + str(i) + ')' + '.jpg'
            # 認識結果の保存
            cv2.imwrite(save_path, dst)
            i += 1
            print("save!")

print("finish")


"""
for line in open('テキストファイルのパス','r'):
    line = line.rstrip()
    print(line)
    image = cv2.imread(origin_image_path+line,0)
    if image is None:
        print('Not open : ',line)
        quit()

    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))

    if len(facerect) > 0:
        for rect in facerect:
            # 顔だけ切り出して保存
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            dst = image[y:y + height, x:x + width]
            save_path = dir_path + '/' + 'image(' + str(i) + ')' + '.jpg'
            #認識結果の保存
            cv2.imwrite(save_path, dst)
            print("save!")
            i += 1
print("Finish")
"""