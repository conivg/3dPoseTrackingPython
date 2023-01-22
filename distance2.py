import cv2
import cvzone
import math
from cvzone.HandTrackingModule import HandDetector
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)
##p1=[0,0,0]
x1 = 0
y1 = 0
z1 = 0
#Find Functions
#raw distance
x = [232,151,131,125,105,90,83,75,68]
#Value at centimiters
y = [20,25,30,35,40,45,50,55,60]
coff = np.polyfit(x,y,2)
listPosX = []
listPosY = []
listPosZ = []
listVel = []
listvx = []
listvy = []
listvz=[]
listAcc=[]
listJerk=[]
listPosX.append(0)
listPosY.append(0)
listPosZ.append(0)
listVel.append(0)
listvx.append(0)
listvy.append(0)
listvz.append(0)
listAcc.append(0)
listJerk.append(0)
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]['lmList']
        x, y, w, h = hands[0]['bbox']
        #x1,y1,z1 = lmList[5]
        xP, yP, zP = lmList[5]
        x2, y2, z2 = lmList[8]
        listPosX.append(x2)
        listPosY.append(y2)
        listPosZ.append(z2)
        #displacement
        difX = listPosX[-1]-listPosX[-2]
        difY = listPosY[-1]-listPosY[-2]
        difZ = listPosZ[-1]-listPosZ[-2]
        dif = math.sqrt((difX)**2 + (difY)**2 + (difZ)**2)
        ##Velocity
        velX = (difX) / (1 / 30)
        velY = (difY) / (1 / 30)
        velZ = (difZ) / (1 / 30)
        vel = math.sqrt((velX) ** 2 + (velY) ** 2 + (velZ) ** 2)
        listvx.append(velX)
        listvy.append(velY)
        listvz.append(velZ)
        listVel.append(vel)
        #Acceleration
        acc = ((listVel[-1] - listVel[-2]) / (1 / 30))
        listAcc.append(acc)
        jerk = ((listAcc[-1] - listAcc[-2]) / (1 / 30))  # (px/frame^3)
        #listJerk.append(jerk)
        ci = (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)) / (math.sqrt(xP ** 2 + yP ** 2 + zP ** 2))
        weight = vel ** 2
        print("speed(tf):", listVel[-1], "speed(ti):", listVel[-2],"acc:", acc, "jerk:", jerk, "weight: ", weight)

        cvzone.putTextRect(img, "speed:" f'{round(vel,2)} p/f', (x + 400, y - 50))
        cvzone.putTextRect(img, "acc:" f'{int(acc)} p/f^2', (x + 400, y))
        cvzone.putTextRect(img, "jerk:" f'{round(jerk, 2)} p/f^3', (x + 400, y + 50))
        cvzone.putTextRect(img, "CI:" f'{round(ci, 2)}', (x + 400, y + 100))
        cvzone.putTextRect(img, "weight:" f'{int(weight)}', (x + 400, y + 150))

        #A,B,C =coff
        #difCM = A*dif**2 + B*dif + C
        #x1, y1, z1 = x2, y2, z2
    cv2.imshow("Image",img)
    cv2.waitKey(1)