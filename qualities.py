import cv2
from cvzone.PoseModule import PoseDetector
import socket
import datetime
import math
import numpy as np
import cvzone
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, savgol_filter, filtfilt, butter

# Parameters
width, height = 1280, 720
# Webcam
cap = cv2.VideoCapture('completeposes.mp4')  # 0 or 1 for webcam
cap.set(3, width)
cap.set(4, height)

# Pose Detector
detector = PoseDetector()
posList = []

class Classificator:
    def __init__(self):
        self.x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57,56,55,22]
        # Value at centimiters
        self.y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,105,110,150]
        self.coff = np.polyfit(self.x, self.y, 2)
        self.listPosXLower = []
        self.listPosYLower = []
        self.listPosZLower = []
        self.listDespCM = []
        self.listVel = []
        self.listAcc = []
        self.listJerk = []
        self.listDespCM.append(0)
        self.listPosXLower.append(0)
        self.listPosYLower.append(0)
        self.listPosZLower.append(0)
        self.listVel.append(0)
        self.listAcc.append(0)
        self.listJerk.append(0)
        self.velprintUL = 0
        self.velprintUR = 0
        self.velprintKL = 0
        self.velprintKR = 0
        self.velprintHL = 0
        self.velprintHR = 0
        self.accprint = 0
        self.printjerk = 0
        self.running_time = time.time()
        self.start_time = time.time()
        self.velFilter = []
        self.velFilter.append(0)
        self.jointsInitialization()

    def jointsInitialization(self):
            self.shoulder_left = []
            self.shoulder_right = []
            self.elbow_left = []
            self.elbow_right = []
            self.wrist_left = []
            self.wrist_right = []
            self.pynky_left = []
            self.pynky_right = []
            self.index_left = []
            self.index_right = []
            self.thumb_left = []
            self.thumb_right = []
            self.hip_left = []
            self.hip_right = []
            self.knee_left = []
            self.knee_right = []
            self.ankle_left = []
            self.ankle_right = []
            self.heel_left = []
            self.heel_right = []
            self.foot_left = []
            self.foot_right = []

    def jointsAppending(self, lmlist,lmlistReal):
        count = 1
        if(count == 1):
            lmlist[11].extend(lmlistReal[11][1:4])
            self.shoulder_left.append(lmlist[11][1:])
            lmlist[12].extend(lmlistReal[12][1:4])
            self.shoulder_right.append(lmlist[12][1:])
            lmlist[13].extend(lmlistReal[13][1:4])
            self.elbow_left.append(lmlist[13][1:])
            lmlist[14].extend(lmlistReal[14][1:4])
            self.elbow_right.append(lmlist[14][1:])
            lmlist[15].extend(lmlistReal[15][1:4])
            self.wrist_left.append(lmlist[15][1:])
            lmlist[16].extend(lmlistReal[16][1:4])
            self.wrist_right.append(lmlist[16][1:])
            lmlist[17].extend(lmlistReal[17][1:4])
            self.pynky_left.append(lmlist[17][1:])
            lmlist[18].extend(lmlistReal[18][1:4])
            self.pynky_right.append(lmlist[18][1:])
            lmlist[19].extend(lmlistReal[19][1:4])
            self.index_left.append(lmlist[19][1:])
            lmlist[20].extend(lmlistReal[20][1:4])
            self.index_right.append(lmlist[20][1:])
            lmlist[21].extend(lmlistReal[21][1:4])
            self.thumb_left.append(lmlist[21][1:])
            lmlist[22].extend(lmlistReal[22][1:4])
            self.thumb_right.append(lmlist[22][1:])
            lmlist[23].extend(lmlistReal[23][1:4])
            self.hip_left.append(lmlist[23][1:])
            lmlist[24].extend(lmlistReal[24][1:4])
            self.hip_right.append(lmlist[24][1:])
            lmlist[25].extend(lmlistReal[25][1:4])
            self.knee_left.append(lmlist[25][1:])
            lmlist[26].extend(lmlistReal[26][1:4])
            self.knee_right.append(lmlist[26][1:])
            lmlist[27].extend(lmlistReal[27][1:4])
            self.ankle_left.append(lmlist[27][1:])
            lmlist[28].extend(lmlistReal[28][1:4])
            self.ankle_right.append(lmlist[28][1:])
            lmlist[29].extend(lmlistReal[29][1:4])
            self.heel_left.append(lmlist[29][1:])
            lmlist[30].extend(lmlistReal[30][1:4])
            self.heel_right.append(lmlist[30][1:])
            lmlist[31].extend(lmlistReal[31][1:4])
            self.foot_left.append(lmlist[31][1:])
            lmlist[32].extend(lmlistReal[32][1:4])
            self.foot_right.append(lmlist[32][1:])
            count = count+1

    def vel(self):
        #Elbow left
        self.difULX = self.elbow_left[-1][3] - self.elbow_left[-2][3]
        self.difULY = (self.elbow_left[-1][4] - self.elbow_left[-2][4])
        self.difULZ = self.elbow_left[-1][5] - self.elbow_left[-2][5]
        # Elbow right
        self.difURX = self.elbow_right[-1][3] - self.elbow_right[-2][3]
        self.difURY = (self.elbow_right[-1][4] - self.elbow_right[-2][4])
        self.difURZ = self.elbow_right[-1][5] - self.elbow_right[-2][5]
        # Knee left
        self.difKLX = self.knee_left[-1][3] - self.knee_left[-2][3]
        self.difKLY = (self.knee_left[-1][4] - self.knee_left[-2][4])
        self.difKLZ = self.knee_left[-1][5] - self.knee_left[-2][5]
        # Knee right
        self.difKRX = self.knee_right[-1][3] - self.knee_right[-2][3]
        self.difKRY = (self.knee_right[-1][4] - self.knee_right[-2][4])
        self.difKRZ = self.knee_right[-1][5] - self.knee_right[-2][5]

        # heel left
        self.difHLX = self.heel_left[-1][3] - self.heel_left[-2][3]
        self.difHLY = (self.heel_left[-1][4] - self.heel_left[-2][4])
        self.difHLZ = self.heel_left[-1][5] - self.heel_left[-2][5]
        # heel right
        self.difHRX = self.heel_right[-1][3] - self.heel_right[-2][3]
        self.difHRY = (self.heel_right[-1][4] - self.heel_right[-2][4])
        self.difHRZ = self.heel_right[-1][5] - self.heel_right[-2][5]

        ## Calculate total displacement
        self.leftElbowdisp = math.sqrt((self.difULX) ** 2 + (self.difULY) ** 2 + (self.difULZ) ** 2)
        self.rightElbowdisp = math.sqrt((self.difURX) ** 2 + (self.difURY) ** 2 + (self.difURZ) ** 2)

        self.leftKneedisp = math.sqrt((self.difKLX) ** 2 + (self.difKLY) ** 2 + (self.difKLZ) ** 2)
        self.rightKneedisp = math.sqrt((self.difKRX) ** 2 + (self.difKRY) ** 2 + (self.difKRZ) ** 2)

        self.leftHeeldisp = math.sqrt((self.difHLX) ** 2 + (self.difHLY) ** 2 + (self.difHLZ) ** 2)
        self.rightHeeldisp = math.sqrt((self.difHRX) ** 2 + (self.difHRY) ** 2 + (self.difHRZ) ** 2)
        aux = []
        # Calculate velocity in time (sec)
        velUL = abs(self.leftElbowdisp / (2 / 30))
        aux.append(velUL)
        velUR = abs(self.rightElbowdisp / (2 / 30))
        aux.append(velUR)
        velKL = abs(self.leftKneedisp / (2 / 30))
        aux.append(velKL)
        velKR = abs(self.rightKneedisp / (2 / 30))
        aux.append(velKR)
        velHL = abs(self.leftHeeldisp / (2 / 30))
        aux.append(velHL)
        velKR = abs(self.rightHeeldisp / (2 / 30))
        aux.append(velKR)
        self.listVel.append(aux)

    def acc(self):
        self.difX = self.knee_right[-1][3] - (2*self.knee_right[-2][3]) + self.knee_right[-3][3]
        self.difY = (self.knee_right[-1][4] - (2*self.knee_right[-2][4]) + self.knee_right[-3][4])
        self.difZ = self.knee_right[-1][5] - (2*self.knee_right[-2][5]) + self.knee_right[-3][5]
        self.RightKneeDisp = math.sqrt((self.difX) ** 2 + (self.difY) ** 2 + (self.difZ) ** 2)
        acc = (self.RightKneeDisp / (1/30)**2)
        self.listAcc.append(acc/10)
    def flow(self):
        self.difX = self.knee_right[-1][3] - (2 * self.knee_right[-2][3]) + (2*self.knee_right[-4][3]) - self.knee_right[-5][3]
        self.difY = (self.knee_right[-1][4] - (2 * self.knee_right[-2][4]) + (2*self.knee_right[-4][4]) - self.knee_right[-5][4])
        self.difZ = self.knee_right[-1][5] - (2 * self.knee_right[-2][5]) + (2*self.knee_right[-4][5]) - self.knee_right[-5][5]
        self.RightKneeDisp = math.sqrt((self.difX) ** 2 + (self.difY) ** 2 + (self.difZ) ** 2)
        flow = (self.RightKneeDisp / (2 * ((1 / 30) ** 3)))
        self.listJerk.append(flow/1000)
    #def weight(self):
        #self.weightVar = self.listVel[-1] ** 2
    def contractionIndex(self):
        leftShoulder = math.sqrt(self.shoulder_left[-1][3] ** 2 + self.shoulder_left[-1][4] ** 2 + self.shoulder_left[-1][5] ** 2)
        rightShoulder = (math.sqrt(self.shoulder_right[-1][3] ** 2 + self.shoulder_right[-1][4] ** 2 + self.shoulder_right[-1][5] ** 2))
        cont = leftShoulder / rightShoulder
        self.ci = cont



def main():
    classificator = Classificator()
    f = open("test.csv", "a")
    headers = "movement, velocity, acc, jerk, ci, weight,evaluatedjoint, pivot\n"
    f.write(headers)
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)
        lmListReal = detector.findPositionReal(img)

        if lmList:
            actualtime = time.time()
            timer = actualtime - classificator.start_time

            classificator.jointsAppending(lmList, lmListReal)

            if (classificator.knee_right.__len__() > 5):
                ## Calculate acceleration from difference in velocity
                classificator.vel()
                classificator.acc()
                #classificator.weight()
                classificator.flow()
                classificator.contractionIndex()
                #((classificator.listVel[-1] - classificator.listVel[-2]) / timer)
                #classificator.listAcc.append(acc)
                #
                # # Calculate jerkiness from difference in acceleration
                # jerk = ((classificator.listAcc[-1] - classificator.listAcc[-2]) / timer)
                # classificator.listJerk.append(jerk)
                # # Calculate contraction index:
                # # ratio between 2 joints that surround another (middle) joint
                #
                #
                # # Calculate the weight of the movement (Kinetic Energy)
                # weight = vel ** 2
                # # print("jerk:", round(jerk, 3), "ci:", round(ci, 3), "weight:", weight)
                #

                if time.time() - classificator.running_time >= 0.1:
                    classificator.velprintUL = classificator.listVel[-1][0]
                    classificator.velprintUR = classificator.listVel[-1][1]
                    classificator.velprintKL = classificator.listVel[-1][2]
                    classificator.velprintKR = classificator.listVel[-1][3]
                    classificator.velprintHL = classificator.listVel[-1][4]
                    classificator.velprintHR = classificator.listVel[-1][5]
                    classificator.accprint = classificator.listAcc[-1]
                    printjerk = classificator.listJerk[-1]
                    #weightprint = classificator.weightVar
                    classificator.running_time = time.time()
                #
                cvzone.putTextRect(img, "Speed ALeft:" f'{int(classificator.velprintUL)} cm/s', (800, 350))  # 100,400
                cvzone.putTextRect(img, "Speed ARight:" f'{int(classificator.velprintUR)} cm/s', (800, 400))
                cvzone.putTextRect(img, "Speed KLeft:" f'{int(classificator.velprintKL)} cm/s', (800, 450))
                cvzone.putTextRect(img, "Speed KRight:" f'{int(classificator.velprintKR)} cm/s', (800, 500))
                cvzone.putTextRect(img, "Speed HLeft:" f'{int(classificator.velprintHL)} cm/s', (800, 550))
                cvzone.putTextRect(img, "Speed HRight:" f'{int(classificator.velprintHR)} cm/s', (800, 600))
                #cvzone.putTextRect(img, "acc:" f'{round(classificator.accprint, 1)} cm/s^2', (800, 400))
                #cvzone.putTextRect(img, "jerk:" f'{round(abs(printjerk), 2)} cm/s^3', (800, 450))
                #cvzone.putTextRect(img, "CI:" f'{round(classificator.ci, 2)}', (800, 500))
                #cvzone.putTextRect(img, "weight:" f'{round(weightprint, 2)}', (800, 550))
                # cvzone.putTextRect(img, "joint:" f'{evaluatedJoint}', (800, 600))
                # cvzone.putTextRect(img, "pivot:" f'{pivot}', (800, 650))
                classificator.start_time = actualtime
                # writetext = f'{datetime.datetime.now()}' + "," + lowerbodyMov + "," + f'{velprint}' + "," + f'{accprint}' + "," + f'{abs(printjerk)}' + "," + f'{ci}' + "," + f'{weightprint}' + "," + f'{evaluatedJoint}' + "," + f'{pivot}' + "\n"
                # f.write(writetext)

            img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
