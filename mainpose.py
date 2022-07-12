import cv2
from cvzone.PoseModule import PoseDetector
import socket

#Parameters
width, height = 1280, 720
#Webcam
cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

#Pose Detector
detector = PoseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
