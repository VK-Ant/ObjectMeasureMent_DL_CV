import os
import numpy as np
import pandas as pd
import cv2
import utills

print(cv2.__version__)


webcam = False
path = '1.jpg'

cap = cv2.VideoCapture(0)
# width,height, brightness
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)

scale = 3
wP = 210 * scale
hP = 297 * scale


while True:

    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    # get contours
    imgcontour, conts = utills.getContours(
        img, minArea=50000, filter=4)

    if len(conts) != 0:
        biggest = conts[0][2]
        # print(biggest)
        imgWarp = utills.warpImg(img, biggest, wP, hP)
        # cv2.imshow('ImageWarp', imgWarp)
        imgcontour2, conts2 = utills.getContours(
            imgWarp, minArea=2000, filter=4,
            cThr=[50, 50], draw=True)
        # cv2.imshow('ImageWarp1', imgWarp)

        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgcontour2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = utills.reorder(obj[2])
                nW = round((utills.findDis(
                    nPoints[0][0]//scale, nPoints[1][0]//scale)/10), 1)

                nH = round((utills.findDis(
                    nPoints[0][0]//scale, nPoints[2][0]//scale)/10), 1)

                cv2.arrowedLine(imgcontour2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgcontour2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgcontour2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgcontour2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow('FinalMeasurement output', imgcontour2)

    img = cv2.resize(img, (0, 0), None, 0.2, 0.2)

    cv2.imshow('Original', img)

    # press q quit the streaming live video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destoryAllWindows()
