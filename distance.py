import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import math
import numpy as np
import cvzone
import datetime


class distanceExample:
    x1, y1, z1 = 0, 0, 0

    def calculatedis(self, x2, y2, z2):
        distance = int(
            math.sqrt((y2 - distanceExample.y1) ** 2 + (x2 - distanceExample.x1) ** 2 + (z2 - distanceExample.z1) ** 2))
        # A, B, C = coff
        # distanceCM = A * distance ** 2 + B * distance + C
        # distanceM = distanceCM / 100
        distanceExample.x1, distanceExample.y1, distanceExample.z1 = x2, y2, z2
        return distance
def main():
    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    # initialTime = 0
    initialDistance = 0
    # changeInTime = 0
    # changeInDistance = 0

    listPos = []
    listSpeed = []
    listAcc = []
    listJerk = []
    # Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    acc = 0
    # Find Function
    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2)
    # y = Ax^2 + Bx + C
    initialTime = time.time()
    listPos.append(initialDistance)
    listJerk.append(0)
    listSpeed.append(0)
    listAcc.append(0)

    # loop
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            lmList = hands[0]['lmList']
            x, y, w, h = hands[0]['bbox']
            x2, y2, z2 = lmList[8]
            xP, yP, zP = lmList[5]
            # xP, yP, zP = lmList[5]
            traj = distanceExample.calculatedis(self=img, x2=x2, y2=y2, z2=z2)
            listPos.append(traj)

            if (listPos.__len__() >= 2):
                speed = abs((listPos[-1] - listPos[-2]) / (1 / 30))  # (px/frame)
                listSpeed.append(speed)
                acc = ((listSpeed[-1] - listSpeed[-2]) / (1 / 30)) / 100  # (px/frame^2)
                listAcc.append(acc)
                jerk = ((listAcc[-1] - listAcc[-2]) / (1 / 30)) / 10000  # (px/frame^3)
                listJerk.append(jerk)
                ci = (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)) / (math.sqrt(xP ** 2 + yP ** 2 + zP ** 2))
                weight = speed**2
                # print("time: ",datetime.datetime.now(),"speed(tf):",listSpeed[-1],"speed(ti):",listSpeed[-2],"acc:",acc, "jerk:",jerk)
                print("weight:", weight)
                cvzone.putTextRect(img, "speed:" f'{round(listSpeed[-1], 2)} p/f', (x + 400, y - 50))
                cvzone.putTextRect(img, "acc:" f'{round(acc, 2)} p/f^2', (x + 400, y))
                cvzone.putTextRect(img, "jerk:" f'{round(jerk, 3)} p/f^3', (x + 400, y + 50))
                cvzone.putTextRect(img, "CI:" f'{round(ci, 3)}', (x + 400, y + 100))

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
