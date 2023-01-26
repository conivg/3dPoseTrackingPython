import cv2
import cvzone
import math
import time
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, savgol_filter
import atexit




def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 640,1280
    cap.set(4, 480)  # 480,720
    detector = HandDetector(detectionCon=0.8, maxHands=1)
    ##p1=[0,0,0]
    x1 = 0
    y1 = 0
    z1 = 0
    # Find Functions
    # raw distance
    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    # Value at centimiters
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    coff = np.polyfit(x, y, 2)
    listPosX = []
    listPosY = []
    listPosZ = []
    listDespCM = []
    listVel = []
    listvx = []
    listvy = []
    listvz = []
    listAcc = []
    listJerk = []
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
    velprint = 0
    accprint = 0
    printjerk = 0
    running_time = time.time()
    start_time = time.time()
    velFilter = []
    velFilter.append(0)
    accFilter = []
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        if hands:
            actualtime = time.time()
            timer = actualtime - start_time

            lmList = hands[0]['lmList']
            x, y, w, h = hands[0]['bbox']
            #x1,y1,z1 = lmList[5]
            xP, yP, zP = lmList[5]
            x2, y2, z2 = lmList[8]
            listPosX.append(x2)
            listPosY.append(y2)
            listPosZ.append(z2)

            #displacement

            #Get displacement in each plane
            difX = listPosX[-1]-listPosX[-2]
            difY = listPosY[-1]-listPosY[-2]
            difZ = listPosZ[-1]-listPosZ[-2]
            #Calculate total displacement
            dif = math.sqrt((difX)**2 + (difY)**2 + (difZ)**2)
            #Calculate coefficients to transform px to cm
            A,B,C = coff
            #Calculate real distance (cm)
            distanceCM = A*dif**2 + B*dif + C
            listDespCM.append(distanceCM)
            #Calculate difference in displacement
            desp = listDespCM[-1] - listDespCM[-2]


            #Calculate velocity in time (sec)
            vel = abs(desp/timer)
            listVel.append(vel)
            n = 15
            b = [0.1/ n] * n
            a = 1
            velFilter = lfilter(b, a, listVel)

            print("value:", listVel[-1])
            print("filtervalue", velFilter[-1])

            # Velocity
            # velX = (difX) / (1/30)
            # velY = (difY) / (1/30)
            # velZ = (difZ) / (1/30)
            #listvx.append(velX)
            #listvy.append(velY)
            #listvz.append(velZ)

            #Calculate acceleration from difference in velocity
            acc = round(((velFilter[-1] - velFilter[-2]) / timer), 3)
            listAcc.append(acc)
            accFilter = lfilter(b, a, listAcc) #savgol_filter(listAcc, 5, 2, mode='nearest')#

            # Calculate jerkiness from difference in acceleration
            jerk = ((accFilter[-1] - accFilter[-2]) / timer)
            listJerk.append(jerk)
            jerkFilter = lfilter(b,a,listJerk)
            #Calculate contraction index:
            # ratio between 2 joints that surround another (middle) joint
            ci = (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)) / (math.sqrt(xP ** 2 + yP ** 2 + zP ** 2))

            #Calculate the weight of the movement (Kinetic Energy)
            weight = vel ** 2


            #print("speed(tf):", round(listVel[-1], 3), "speed(ti):", round(listVel[-2],3),"acc:", acc, "jerk:", jerk, "weight: ", weight)


            if time.time() - running_time >= 0.05:
                velprint = velFilter[-1] #vel
                accprint = accFilter[-1]
                printjerk = jerkFilter[-1]
                weightprint = weight
                running_time = time.time()

            cvzone.putTextRect(img, "speed:" f'{int(velprint)} cm/s', (x + 150, y - 50)) #100,400
            cvzone.putTextRect(img, "acc:" f'{accprint} cm/s^2', (x + 150, y))
            cvzone.putTextRect(img, "jerk:" f'{round(abs(printjerk),2)} cm/s^3', (x + 150, y + 50))
            cvzone.putTextRect(img, "CI:" f'{round(ci, 2)}', (x + 150, y + 100))
            cvzone.putTextRect(img, "weight:" f'{round(weightprint/1000,2)}', (x + 150, y + 150))
            start_time = actualtime


            #atexit.register(plots, velFilter)
            #A,B,C =coff
            #difCM = A*dif**2 + B*dif + C
            #x1, y1, z1 = x2, y2, z2
        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()