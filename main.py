import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(3, width)
cap.set(4, height)
segmentor = SelfiSegmentation()  # default parameter is 1
cap.set(cv2.CAP_PROP_FPS, 60)
fpsReader = cvzone.FPS()


listImg = os.listdir("Images")

imgList = []
for imgPath in listImg:
    img = cv2.imread(f"Images/{imgPath}")
    imgList.append(img)


indexImg = 5
imgList[indexImg] = cv2.resize(imgList[indexImg], (width, height))

while True:
    success, img = cap.read()

    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)  # they recommended and that is default 0.1
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    fps, imgStacked = fpsReader.update(imgStacked)
    cv2.imshow("Background Change", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('c'):
        indexImg = (indexImg+1) % len(imgList)
        imgList[indexImg] = cv2.resize(imgList[indexImg], (width, height))

    elif key == ord('q'):
        break

