import cv2
from cvzone.PoseModule import PoseDetector
import socket
import math
import numpy as np
import cvzone

#Parameters
width, height = 1280, 720
#Webcam
cap = cv2.VideoCapture('brazos.mp4') #0 or 1 for webcam
cap.set(3, width)
cap.set(4, height)


#Pose Detector
detector = PoseDetector()
posList = []

#Communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5053)
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)
    data = []
    angles = []
    if bboxInfo:
        lmString = ''
        for lm in lmList:
            data.extend([lm[1], height - lm[2], lm[3]])
        sock.sendto(str.encode(str(data)), serverAddressPort)
    if lmList:
        x1, y1, z1 = lmList[13][1:]
        x2, y2, z2 = lmList[11][1:]
        x3, y3, z3 = lmList[23][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360
        print(angle)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

