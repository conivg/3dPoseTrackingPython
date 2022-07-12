import cv2
from cvzone.PoseModule import PoseDetector
import socket

#Parameters
width, height = 1280, 720
#Webcam
cap = cv2.VideoCapture('dance.mp4') #0 or 1 for webcam
cap.set(3, width)
cap.set(4, height)


#Pose Detector
detector = PoseDetector()
posList = []

#Communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)
    data = []
    if bboxInfo:
        lmString = ''
        for lm in lmList:
            data.extend([lm[1], height - lm[2], lm[3]])
        print(data)
        sock.sendto(str.encode(str(data)), serverAddressPort)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


