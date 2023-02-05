import cv2
from cvzone.PoseModule import PoseDetector
import socket
import math
import numpy as np
import cvzone
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, savgol_filter, filtfilt,butter

#Parameters
width, height = 1280, 720
#Webcam
cap = cv2.VideoCapture('brazos.mp4') #0 or 1 for webcam
cap.set(3, width)
cap.set(4, height)


#Pose Detector
detector = PoseDetector()
posList = []

class Classificator:
    def __init__(self):
        self.x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
        # Value at centimiters
        self.y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        self.coff = np.polyfit(self.x, self.y, 2)
        self.listPosX = []
        self.listPosY = []
        self.listPosZ = []
        self.listDespCM = []
        self.listVel = []
        self.listvx = []
        self.listvy = []
        self.listvz = []
        self.listAcc = []
        self.listJerk = []
        self.listDespCM.append(0)
        self.listPosX.append(0)
        self.listPosY.append(0)
        self.listPosZ.append(0)
        self.listVel.append(0)
        self.listvx.append(0)
        self.listvy.append(0)
        self.listvz.append(0)
        self.listAcc.append(0)
        self.listJerk.append(0)
        self.velprint = 0
        self.accprint = 0
        self.printjerk = 0
        self.running_time = time.time()
        self.start_time = time.time()
        self.velFilter = []
        self.velFilter.append(0)

def main():
    #Communication
    #sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #serverAddressPort = ("127.0.0.1", 5053)
    classificator = Classificator()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)
        data = []
        angles = []
        if bboxInfo:
            lmString = ''
            for lm in lmList:
                data.extend([lm[1], lm[2], lm[3]])
                #sock.sendto(str.encode(str(data)), serverAddressPort)
        if lmList:
            actualtime = time.time()
            timer = actualtime - classificator.start_time
            x1, y1, z1 = lmList[15][1:]
            xP, yP, zP = lmList[11][1:]
            #x3, y3, z3 = lmList[23][1:]
            # Calculate the Angle
            #angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                 #math.atan2(y1 - y2, x1 - x2))
            #if angle < 0:
                #angle += 360
            #print(angle)
            classificator.listPosX.append(x1)
            classificator.listPosY.append(y1*(-1))
            classificator.listPosZ.append(z1)

            if(classificator.listPosY[-1] - classificator.listPosY[-2] > 1):
                cvzone.putTextRect(img, "Ascending arms", (width + 150, height - 150))  # 100,400
            if (classificator.listPosY[-1] - classificator.listPosY[-2] < -1):
                cvzone.putTextRect(img, "Descending arms", (width + 150, height - 150))  # 100,400
            # Get displacement in each plane
            difX = classificator.listPosX[-1] - classificator.listPosX[-2]
            difY = classificator.listPosY[-1] - classificator.listPosY[-2]
            difZ = classificator.listPosZ[-1] - classificator.listPosZ[-2]
            # Calculate total displacement
            dif = math.sqrt((difX) ** 2 + (difY) ** 2 + (difZ) ** 2)
            # Calculate coefficients to transform px to cm
            A, B, C = classificator.coff
            # Calculate real distance (cm)
            distanceCM = A * dif ** 2 + B * dif + C
            classificator.listDespCM.append(distanceCM)
            # Calculate difference in displacement
            desp = classificator.listDespCM[-1] - classificator.listDespCM[-2]

            #print(dif)
            # Calculate velocity in time (sec)
            vel = abs(desp / timer)
            classificator.listVel.append(vel)
            # Calculate acceleration from difference in velocity
            acc = round(((classificator.listVel[-1] - classificator.listVel[-2]) / timer), 2)
            classificator.listAcc.append(acc)
            classificator.listAcc[:] = [x / 10 for x in  classificator.listAcc]
            # accFilter = lowpass_filter(listAcc, 0.1, 1, order=2)
            print("speed(tf):", round( classificator.listVel[-1], 3), "speed(ti):", round( classificator.listVel[-2], 3), "acc:", classificator.listAcc[-1])

            # Calculate jerkiness from difference in acceleration
            jerk = ((classificator.listAcc[-1] - classificator.listAcc[-2]) / timer)
            classificator.listJerk.append(jerk)
            # jerkFilter = lowpass_filter(listJerk, 0.1, 1, order=4)#lfilter(b,a,listJerk)
            # Calculate contraction index:
            # ratio between 2 joints that surround another (middle) joint
            ci = (math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)) / (math.sqrt(xP ** 2 + yP ** 2 + zP ** 2))

            # Calculate the weight of the movement (Kinetic Energy)
            weight = vel ** 2
            print("jerk:", round(jerk, 3), "ci:", round(ci, 3), "weight:", weight)

            if time.time() - classificator.running_time >= 0.1:
                velprint = classificator.listVel[-1]  # vel
                accprint = classificator.listAcc[-1]
                printjerk = classificator.listJerk[-1]
                weightprint = weight
                classificator.running_time = time.time()

            cvzone.putTextRect(img, "speed:" f'{int(velprint)} cm/s', (width + 150, height - 50))  # 100,400
            cvzone.putTextRect(img, "acc:" f'{round(accprint, 1)} cm/s^2', (width + 150, height))
            cvzone.putTextRect(img, "jerk:" f'{round(abs(printjerk), 2)} cm/s^3', (width + 150,height + 50))
            cvzone.putTextRect(img, "CI:" f'{round(ci, 2)}', (width + 150, height + 100))
            cvzone.putTextRect(img, "weight:" f'{round(weightprint / 1000, 2)}', (width + 150, height + 150))
            classificator.start_time = actualtime

        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

