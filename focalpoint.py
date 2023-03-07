import math
import time
import cv2
from cvzone.PoseModule import PoseDetector
import mediapipe as mp
import numpy as np
import cvzone

#Webcam
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
frame_count = cap.get(cv2.CAP_PROP_FPS)
#Pose detector
detector = PoseDetector()
armPosX = []
armPosX.append(0)
armPosY = []
armPosY.append(0)
armPosZ = []
armPosZ.append(0)
listDespCM = []
listDespCM.append(0)
start_time = time.time()
actualtime = time.time()
#Loop
while True:
        timer = time.time() - start_time
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)
        lmListReal = detector.findPositionReal(img)
        if lmList:
            #x1, y1, z1 = lmList[12][1:]
            xreal,yreal,zreal = lmListReal[12][1:]
            #x2, y2, z2 = lmList[14][1:]
            #x_pivot, y_pivot, z_pivot = lmList[24][1:]

            x2real, y2real, z2real = lmListReal[16][1:]
            armPosX.append(x2real)
            armPosY.append(y2real)
            armPosZ.append(z2real)

            difX = armPosX[-1] - armPosX[-2]
            difY = armPosY[-1] - armPosY[-2]
            difZ = armPosZ[-1] - armPosZ[-2]

            dif = math.sqrt((difX) ** 2 + (difY) ** 2 + (difZ) ** 2)

            listDespCM.append(dif)
            # Calculate difference in displacement
            desp = listDespCM[-1] - listDespCM[-2]

            if(timer > 0):
                vel = abs(desp / timer)

                print(vel)
                start_time = time.time()
            #distance = int(math.sqrt((y2real-yreal)**2 + (x2real-xreal)**2 + (z2real-zreal)**2))
            #proportion =
            #ydifcentDPI = ((y2-y1)*2.54)/96
            #ydifcentReal = (armPos[-1] - armPos[0])
            #x1cent = (y1 * 2.54)/96

                #print(abs(vel))
        #img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("Image",img)
        cv2.waitKey(1)

