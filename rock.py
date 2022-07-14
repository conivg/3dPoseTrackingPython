import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
#Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

while True:
    imgBG = cv2.imread("Resources/BG.png")
    success, img = cap.read()

    imgScaled = cv2.resize(img, (0,0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]
    #cv2.imshow("Image", img)
    #Find hands
    hands, img = detector.findHands(imgScaled)
    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        if fingers == [0, 0, 0, 0, 0]:
            print('ROCK')
        if fingers == [1, 1, 1, 1, 1]:
            print('PAPER')
        if fingers == [0, 1, 1, 0, 0]:
            print('SCISSORS')
    imgBG[234:654, 795:1195] = imgScaled
    cv2.imshow("BG", imgBG)
    cv2.waitKey(1)
