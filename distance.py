import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import math
import numpy as np
import cvzone
import sympy as sym

class distanceExample:

    def calculatedis(x1, x2, y1, y2, z1, z2, coff):
        distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2 + (z2 - z1) ** 2))
        A, B, C = coff
        distanceCM = A * distance ** 2 + B * distance + C
        distanceM = distanceCM / 100
        return distance
def main():
    #Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    #initialTime = 0
    initialDistance = 0
    #changeInTime = 0
    #changeInDistance = 0

    listPos = []
    listSpeed =[]
    listAcc = []
    listJerk =[]
    #Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    acc = 0
    #Find Function
    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2)
    # y = Ax^2 + Bx + C
    initialTime = time.time()
    listPos.append(initialDistance)
    #loop
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            lmList = hands[0]['lmList']
            x,y,w,h = hands[0]['bbox']
            x1, y1, z1 = lmList[5]
            x2, y2, z2 = lmList[17]
            distanceM = distanceExample.calculatedis(x1, x2, y1, y2, z1, z2, coff)
            listPos.append(distanceM)

            if(listPos.__len__() >= 2 ):
                speed = abs((listPos[-1] - listPos[-2]) / (1/30))  # abs((distanceM - initialDistance)/(2*(1/60)*(time.time() - initialTime)))
                listSpeed.append(speed)
                if(listSpeed.__len__() >= 2):
                    acc = ((listSpeed[-1] - listSpeed[-2]) / (1/30))/100
                    listAcc.append(acc)
                    #acc = ((listSpeed[-1] - listSpeed[-2]) / (1/60))/100
                    print("speed(tf):",listSpeed[-1],"speed(ti):",listSpeed[-2],"acc:",acc)
                    cvzone.putTextRect(img, f'{round(listSpeed[-1], 2)} p/f', (x + 5, y - 10))
                    if(listAcc.__len__() >= 2 ):
                        jerk = ((listAcc[-1] - listAcc[-2]) / (2/30))
                        print(jerk)

        cv2.imshow("image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()