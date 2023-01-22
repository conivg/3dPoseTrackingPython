import cv2
import cvzone
import math
import time
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
x = [300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
#Value at centimiters
y = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff = np.polyfit(x,y,2)
listPosX = []
listPosY = []
listPosZ = []
listDespCM = []
listVel = []
listvx = []
listvy = []
listvz=[]
listAcc=[]
listJerk=[]
listDespCM.append(0)
listPosX.append(0)
listPosY.append(0)
listPosZ.append(0)
listVel.append(0)
listvx.append(0)
listvy.append(0)
listvz.append(0)
listAcc.append(0)
listJerk.append(0)
start_time = time.time()
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
        A,B,C = coff
        distanceCM = A*dif**2 + B*dif + C
        listDespCM.append(distanceCM)
        desp = listDespCM[-1] - listDespCM[-2]
        actualtime = time.time()
        timer = actualtime-start_time
        print("time:", timer)
        #Velocity
        #velX = (difX) / (1/30)
        #velY = (difY) / (1/30)
        #velZ = (difZ) / (1/30)
        vel = abs(desp/timer)
        #listvx.append(velX)
        #listvy.append(velY)
        #listvz.append(velZ)
        listVel.append(vel)
        #Acceleration
        acc = round(((listVel[-1] - listVel[-2]) / timer)/100, 3)
        listAcc.append(acc)

        jerk = round(((listAcc[-1] - listAcc[-2]) / timer)/10000, 2)  # (px/frame^3)
        #listJerk.append(jerk)
        ci = (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)) / (math.sqrt(xP ** 2 + yP ** 2 + zP ** 2))
        weight = vel ** 2
        print("speed(tf):", round(listVel[-1], 3), "speed(ti):", round(listVel[-2],3),"acc:", acc, "jerk:", jerk, "weight: ", weight)

        cvzone.putTextRect(img, "speed:" f'{int(vel)} cm/s', (x + 400, y - 50))
        cvzone.putTextRect(img, "acc:" f'{int(acc)} cm/s^2', (x + 400, y))
        cvzone.putTextRect(img, "jerk:" f'{round(jerk, 2)} cm/s^3', (x + 400, y + 50))
        cvzone.putTextRect(img, "CI:" f'{round(ci, 2)}', (x + 400, y + 100))
        cvzone.putTextRect(img, "weight:" f'{round(weight,2)}', (x + 400, y + 150))
        start_time = actualtime
        #A,B,C =coff
        #difCM = A*dif**2 + B*dif + C
        #x1, y1, z1 = x2, y2, z2
    cv2.imshow("Image",img)
    cv2.waitKey(1)