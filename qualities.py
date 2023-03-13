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
cap = cv2.VideoCapture('videos/vid1.mp4')  # 0 or 1 for webcam
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
        self.listWeight = []
        self.listCI= []
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
        self.stateUp = 0
        self.stateLow = 0
        self.lowerbodyMov = ""
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

    def vel(self):
        self.stateUp = 0
        self.stateLow =0
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
        velUL = int(abs(self.leftElbowdisp / (2 / 30)))
        aux.append(velUL)
        velUR = int(abs(self.rightElbowdisp / (2 / 30)))
        aux.append(velUR)
        velKL = int(abs(self.leftKneedisp / (2 / 30)))
        aux.append(velKL)
        velKR = int(abs(self.rightKneedisp / (2 / 30)))
        aux.append(velKR)
        velHL = int(abs(self.leftHeeldisp / (2 / 30)))
        aux.append(velHL)
        velKR = int(abs(self.rightHeeldisp / (2 / 30)))
        aux.append(velKR)

        if velUL >= 10 or velUR >= 10:
            self.stateUp = 1
        if velKL >= 10 or velKR >= 10:
            self.stateLow = 1

        self.listVel.append(aux)

    def acc(self):
        # Elbow Left
        self.difULX = self.elbow_left[-1][3] - (2 * self.elbow_left[-2][3]) + self.elbow_left[-3][3]
        self.difULY = (self.elbow_left[-1][4] - (2 * self.elbow_left[-2][4]) + self.elbow_left[-3][4])
        self.difULZ = self.elbow_left[-1][5] - (2 * self.elbow_left[-2][5]) + self.elbow_left[-3][5]
        # Elbow Right
        self.difURX = self.elbow_right[-1][3] - (2 * self.elbow_right[-2][3]) + self.elbow_right[-3][3]
        self.difURY = (self.elbow_right[-1][4] - (2 * self.elbow_right[-2][4]) + self.elbow_right[-3][4])
        self.difURZ = self.elbow_right[-1][5] - (2 * self.elbow_right[-2][5]) + self.elbow_right[-3][5]
        # Knee Left
        self.difKLX = self.knee_left[-1][3] - (2 * self.knee_left[-2][3]) + self.knee_left[-3][3]
        self.difKLY = (self.knee_left[-1][4] - (2 * self.knee_left[-2][4]) + self.knee_left[-3][4])
        self.difKLZ = self.knee_left[-1][5] - (2 * self.knee_left[-2][5]) + self.knee_left[-3][5]
        # Right Knee
        self.difKRX = self.knee_right[-1][3] - (2*self.knee_right[-2][3]) + self.knee_right[-3][3]
        self.difKRY = (self.knee_right[-1][4] - (2*self.knee_right[-2][4]) + self.knee_right[-3][4])
        self.difKRZ = self.knee_right[-1][5] - (2*self.knee_right[-2][5]) + self.knee_right[-3][5]
        # Knee Left
        self.difHLX = self.heel_left[-1][3] - (2 * self.heel_left[-2][3]) + self.heel_left[-3][3]
        self.difHLY = (self.heel_left[-1][4] - (2 * self.heel_left[-2][4]) + self.heel_left[-3][4])
        self.difHLZ = self.heel_left[-1][5] - (2 * self.heel_left[-2][5]) + self.heel_left[-3][5]
        # Right Knee
        self.difHRX = self.heel_right[-1][3] - (2 * self.heel_right[-2][3]) + self.heel_right[-3][3]
        self.difHRY = (self.heel_right[-1][4] - (2 * self.heel_right[-2][4]) + self.heel_right[-3][4])
        self.difHRZ = self.heel_right[-1][5] - (2 * self.heel_right[-2][5]) + self.heel_right[-3][5]



        self.leftElbowdisp = math.sqrt((self.difULX) ** 2 + (self.difULY) ** 2 + (self.difULZ) ** 2)
        self.rightElbowdisp = math.sqrt((self.difURX) ** 2 + (self.difURY) ** 2 + (self.difURZ) ** 2)

        self.leftKneedisp = math.sqrt((self.difKLX) ** 2 + (self.difKLY) ** 2 + (self.difKLZ) ** 2)
        self.RightKneeDisp = math.sqrt((self.difKRX) ** 2 + (self.difKRY) ** 2 + (self.difKRZ) ** 2)

        self.leftHeeldisp = math.sqrt((self.difHLX) ** 2 + (self.difHLY) ** 2 + (self.difHLZ) ** 2)
        self.RightHeelDisp = math.sqrt((self.difHRX) ** 2 + (self.difHRY) ** 2 + (self.difHRZ) ** 2)

        aux = []
        accUL = (self.leftElbowdisp / (1 / 30) ** 2) / 10
        aux.append(round(accUL,4))
        accUR = (self.rightElbowdisp / (1 / 30) ** 2) /10
        aux.append(round(accUR,4))
        accKL = (self.leftKneedisp / (1 / 30) ** 2) /10
        aux.append(round(accKL,4))
        accKR = (self.RightKneeDisp / (1/30)**2) /10
        aux.append(round(accKR,4))
        accHL = (self.leftHeeldisp / (1 / 30) ** 2) /10
        aux.append(round(accHL,4))
        accHR = (self.RightHeelDisp / (1 / 30) ** 2) / 10
        aux.append(round(accHR,4))
        self.listAcc.append(aux)
        #self.listAcc.append(acc/10)

    def flow(self):
        # Elbow Left
        self.difULX = self.elbow_left[-1][3] - (2 * self.elbow_left[-2][3]) + (2 * self.elbow_left[-4][3]) - self.elbow_left[-5][3]
        self.difULY = (self.elbow_left[-1][4] - (2 * self.elbow_left[-2][4]) + (2 * self.elbow_left[-4][4]) - self.elbow_left[-5][4])
        self.difULZ = self.elbow_left[-1][5] - (2 * self.elbow_left[-2][5]) + (2 * self.elbow_left[-4][5]) - self.elbow_left[-5][5]
        # Elbow Right
        self.difURX = self.elbow_right[-1][3] - (2 * self.elbow_right[-2][3]) + (2 * self.elbow_right[-4][3]) - self.elbow_right[-5][3]
        self.difURY = (self.elbow_right[-1][4] - (2 * self.elbow_right[-2][4]) + (2 * self.elbow_right[-4][4]) - self.elbow_right[-5][4])
        self.difURZ = self.elbow_right[-1][5] - (2 * self.elbow_right[-2][5]) + (2 * self.elbow_right[-4][5]) - self.elbow_right[-5][5]
        # Left Knee
        self.difKLX = self.knee_left[-1][3] - (2 * self.knee_left[-2][3]) + (2 * self.knee_left[-4][3]) - self.knee_left[-5][3]
        self.difKLY = (self.knee_left[-1][4] - (2 * self.knee_left[-2][4]) + (2 * self.knee_left[-4][4]) - self.knee_left[-5][4])
        self.difKLZ = self.knee_left[-1][5] - (2 * self.knee_left[-2][5]) + (2 * self.knee_left[-4][5]) - self.knee_left[-5][5]
        # Right Knee
        self.difKRX = self.knee_right[-1][3] - (2 * self.knee_right[-2][3]) + (2*self.knee_right[-4][3]) - self.knee_right[-5][3]
        self.difKRY = (self.knee_right[-1][4] - (2 * self.knee_right[-2][4]) + (2*self.knee_right[-4][4]) - self.knee_right[-5][4])
        self.difKRZ = self.knee_right[-1][5] - (2 * self.knee_right[-2][5]) + (2*self.knee_right[-4][5]) - self.knee_right[-5][5]
        # Left Heel
        self.difHLX = self.heel_left[-1][3] - (2 * self.heel_left[-2][3]) + (2 * self.heel_left[-4][3]) - self.heel_left[-5][3]
        self.difHLY = (self.heel_left[-1][4] - (2 * self.heel_left[-2][4]) + (2 * self.heel_left[-4][4]) - self.heel_left[-5][4])
        self.difHLZ = self.heel_left[-1][5] - (2 * self.heel_left[-2][5]) + (2 * self.heel_left[-4][5]) - self.heel_left[-5][5]
        # Right Knee
        self.difHRX = self.heel_right[-1][3] - (2 * self.heel_right[-2][3]) + (2 * self.heel_right[-4][3]) - self.heel_right[-5][3]
        self.difHRY = (self.heel_right[-1][4] - (2 * self.heel_right[-2][4]) + (2 * self.heel_right[-4][4]) - self.heel_right[-5][4])
        self.difHRZ = self.heel_right[-1][5] - (2 * self.heel_right[-2][5]) + (2 * self.heel_right[-4][5]) -  self.heel_right[-5][5]

        self.leftElbowdisp = math.sqrt((self.difULX) ** 2 + (self.difULY) ** 2 + (self.difULZ) ** 2) / 100
        self.rightElbowdisp = math.sqrt((self.difURX) ** 2 + (self.difURY) ** 2 + (self.difURZ) ** 2) / 100
        self.leftKneedisp = math.sqrt((self.difKLX) ** 2 + (self.difKLY) ** 2 + (self.difKLZ) ** 2) / 100
        self.RightKneeDisp = math.sqrt((self.difKRX) ** 2 + (self.difKRY) ** 2 + (self.difKRZ) ** 2) / 100
        self.leftHeeldisp = math.sqrt((self.difHLX) ** 2 + (self.difHLY) ** 2 + (self.difHLZ) ** 2) / 100
        self.rightHeeldisp = math.sqrt((self.difHRX) ** 2 + (self.difHRY) ** 2 + (self.difHRZ) ** 2) / 100
        aux = []
        flowUL = (self.leftElbowdisp / (2 * ((1 / 30) ** 3)))
        aux.append(round(flowUL, 4))
        flowUR = (self.rightElbowdisp / (2 * ((1 / 30) ** 3)))
        aux.append(round(flowUR, 4))
        flowKL = (self.leftKneedisp / (2 * ((1 / 30) ** 3)))
        aux.append(round(flowKL, 4))
        flowKR = (self.RightKneeDisp / (2 * ((1 / 30) ** 3)))
        aux.append(round(flowKR, 4))
        flowHL = (self.leftHeeldisp / (2 * ((1 / 30) ** 3)))
        aux.append(round(flowHL, 4))
        flowHR = (self.rightHeeldisp / (2 * ((1 / 30) ** 3)))
        aux.append(round(flowHR, 4))
        self.listJerk.append(aux)

    def weight(self):
        weightUL = self.listVel[-1][0] ** 2
        weightUR = self.listVel[-1][1] ** 2
        weightKL = self.listVel[-1][2] ** 2
        weightKR = self.listVel[-1][3] ** 2
        weightHL = self.listVel[-1][4] ** 2
        weightHR = self.listVel[-1][5] ** 2
        aux = []
        aux.append(weightUL)
        aux.append(weightUR)
        aux.append(weightKL)
        aux.append(weightKR)
        aux.append(weightHL)
        aux.append(weightHR)
        self.listWeight.append(aux)

    def contractionIndex(self):
        leftShoulder = math.sqrt(self.shoulder_left[-1][3] ** 2 + self.shoulder_left[-1][4] ** 2 + self.shoulder_left[-1][5] ** 2)
        rightShoulder = (math.sqrt(self.shoulder_right[-1][3] ** 2 + self.shoulder_right[-1][4] ** 2 + self.shoulder_right[-1][5] ** 2))
        cont = leftShoulder / rightShoulder
        self.ci = cont

    def movType(self):
        if(self.stateLow == 1):
            #profile values
            if ((self.knee_left[-1][2] - self.knee_left[-2][2] < -0.1) &
                    (self.knee_right[-1][2] - self.knee_right[-2][2] > 0.1) &
                    (self.hip_left[-1][1] - self.hip_left[-2][1] > 0.1) &
                    (self.hip_right[-1][1] - self.hip_right[-2][1] > 0.1)):
                     self.lowerbodyMov = "Plie"
            if ((self.knee_left[-1][2] - self.knee_left[-2][2] > 0.1) &
                    (self.knee_right[-1][2] - self.knee_right[-2][2] < -0.1) &
                    (self.hip_left[-1][1] - self.hip_left[-2][1] < -0.1) &
                    (self.hip_right[-1][1] - self.hip_right[-2][1] < -0.1)):

                     self.lowerbodyMov = "Back from Plie"
        #Releve -> Negative displacement of hips.y; Negative displacement of heels.y; Not foot displacement in y and x
            # if ((self.hip_left[-1][1] - self.hip_left[-2][1] < -0.3) &
            #         (self.hip_right[-1][1] - self.hip_right[-2][1] < -0.3) &
            #         (self.heel_left[-1][1] - self.heel_left[-2][1] < -0.3) &
            #         (self.heel_right[-1][1] - self.heel_right[-2][1] < -0.3) &
            #         (self.foot_right[-1][1] - self.foot_right[-2][1] < -0.3) &
            #         (self.foot_left[-1][1] - self.foot_left[-2][1] < -0.3)):
            #         self.lowerbodyMov = "Releve"
            # if ((self.hip_left[-1][1] - self.hip_left[-2][1] > 0.3) &
            #         (self.hip_right[-1][1] - self.hip_right[-2][1] > 0.3) &
            #         (self.heel_left[-1][1] - self.heel_left[-2][1] > 0.3) &
            #         (self.heel_right[-1][1] - self.heel_right[-2][1] > 0.3) &
            #         (self.foot_right[-1][1] - self.foot_right[-2][1] > 0.3) &
            #         (self.foot_left[-1][1] - self.foot_left[-2][1] > 0.3)):
            #     self.lowerbodyMov = "Back from Releve"
        # Etendre Right -> Not displacement in left hip.y; Negative displacement in right foot.x;  Not displacement in left foot.x;
        #     if ((abs(self.foot_right[-1][0] - self.foot_right[-2][0]) > 0.2) &
        #         (abs(self.heel_right[-1][0] - self.heel_right[-2][0]) > 0.2) &
        #         (abs(self.hip_right[-1][1] - self.hip_right[-2][1]) < 0.1) &
        #         (abs(self.hip_left[-1][1] - self.hip_left[-2][1]) < 0.1) &
        #         (self.heel_right[-1][1] - self.heel_right[-2][1] < -0.3)):
        #         lowerbodyMov = "Etendre right"
        #     if ((self.foot_right[-1][0] - self.foot_right[-2][0] > 1) &
        #         (self.heel_right[-1][0] - self.heel_right[-2][0] > 1) &
        #         (abs(self.hip_right[-1][1] - self.hip_right[-2][1]) < 0.1) &
        #         (abs(self.hip_left[-1][1] - self.hip_left[-2][1]) < 0.1) &
        #         (self.heel_right[-1][1] - self.heel_right[-2][1] > 1.5)):
        #         self.lowerbodyMov = "Back from Etendre right"
        # Etendre Left -> Not displacement in right hip.y; Positive displacement in left foot.x or z;  Not displacement in right foot.x; Backward/Forward
            if ((abs(self.foot_left[-1][0] - self.foot_left[-2][0]) > 0.3) &
                 (abs(self.heel_left[-1][0] - self.heel_left[-2][0]) > 0.3) &
                 (abs(self.foot_right[-1][0] - self.foot_right[-2][0]) < 0.1)):
                self.lowerbodyMov = "Etendre left"
            # if ((self.foot_left[-1][0] - self.foot_left[-2][0] < -0.3) &
            #     (self.heel_left[-1][0] - self.heel_left[-2][0] < -0.3) &
            #     (abs(self.hip_right[-1][1] - self.hip_right[-2][1]) < 0.1) &
            #     (abs(self.hip_left[-1][1] - self.hip_left[-2][1]) < 0.1) &
            #     (self.heel_left[-1][1] - self.heel_left[-2][1] > 0.3)):
            #     self.lowerbodyMov = "Back from Etendre left"
        #Tourner ->Not hips displacement in y; hips.x displacement any side; hips.z displacement any side
            if ((abs(self.hip_left[-1][0] - self.hip_left[-2][0]) > 0.5) &
                    (abs(self.hip_right[-1][0] - self.hip_right[-2][0]) > 0.5) &
                    (abs(self.hip_left[-1][2] - self.hip_left[-2][2]) > 0.5) &
                    (abs(self.hip_right[-1][2] - self.hip_right[-2][2]) > 0.5) &
                    (abs(self.foot_right[-1][0] - self.foot_right[-2][0]) > 0.5) &
                    (abs(self.foot_left[-1][0] - self.foot_left[-2][0]) > 0.5) &
                    (abs(self.hip_left[-1][1] - self.hip_left[-2][1]) < 0.05)
            ):
                self.lowerbodyMov = "Tourner"
        #Sauter -> Jumping; Negative in foot.y; Negative displacement in hips.y; Not displacement in hips.x any side; Negative displacement in heel.y
        #     if ((self.foot_left[-1][1] - self.foot_left[-2][1] < -1) &
        #         (self.foot_right[-1][1] - self.foot_right[-2][1] < -1) &
        #         (self.hip_left[-1][1] - self.hip_left[-2][1] < -0.8) &
        #         (self.hip_right[-1][1] - self.hip_right[-2][1] < -0.8) &
        #         (self.heel_left[-1][1] - self.heel_left[-2][1] < -1) &
        #         (self.heel_right[-1][1] - self.heel_right[-2][1] < -1) &
        #         (abs(self.hip_right[-1][0] - self.hip_right[-2][0]) < 0.05)):
        #         self.lowerbodyMov = "Jumping"
        # # Elancer -> Jumping with direction; Positive Negative in foot.y; Negative displacement in hips.y; Displacement in hips.x any side
        #     if ((self.foot_left[-1][1] - self.foot_left[-2][1] < -0.8) &
        #         (self.foot_right[-1][1] - self.foot_right[-2][1] < -0.8) &
        #         (self.hip_left[-1][1] - self.hip_left[-2][1] < -0.8) &
        #         (self.hip_right[-1][1] - self.hip_right[-2][1] < -0.8) &
        #         (self.heel_left[-1][1] - self.heel_left[-2][1] < -0.5) &
        #         (self.heel_right[-1][1] - self.heel_right[-2][1] < -0.5) &
        #         (abs(self.hip_right[-1][0] - self.hip_right[-2][0]) > 0.5)):
        #         self.lowerbodyMov = "Elancer"
        else:
            self.lowerbodyMov = ""



def main():
    classificator = Classificator()
    f = open("files/p4-states.csv", "a")
    headers = "timestamp, stateUp, stateLow, lowmovtype, velUL, velUR, velKL, velKR, velHL, velHR," \
              "accUL, accUR, accKL, accKR, accHL, accHR," \
              "jerUL, jerUR, jerKL, jerKR, jerHL, jerHR," \
              "weiUL, weiUR, weiKL, weiKR, weiHL, weiHR," \
              "CI,\n"
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
                classificator.movType()
                classificator.acc()
                classificator.weight()
                classificator.flow()
                classificator.contractionIndex()


                if time.time() - classificator.running_time >= 0.1:
                    classificator.velprintUL = classificator.listVel[-1][0]
                    #classificator.velprintUR = classificator.listVel[-1][1]
                    #classificator.velprintKL = classificator.listVel[-1][2]
                    #classificator.velprintKR = classificator.listVel[-1][3]
                    #classificator.velprintHL = classificator.listVel[-1][4]
                    #classificator.velprintHR = classificator.listVel[-1][5]
                    #classificator.accprint = classificator.listAcc[-1]
                    #printjerk = classificator.listJerk[-1]
                    #weightprint = classificator.weightVar
                    classificator.running_time = time.time()
                #
                cvzone.putTextRect(img, "Speed ALeft:" f'{classificator.velprintUL} m/s', (800, 350))
                cvzone.putTextRect(img, "Move" f'{classificator.stateLow}', (800, 400))
                cvzone.putTextRect(img, "Move" f'{classificator.lowerbodyMov} m/s', (800, 450))
                # 100,400
                # cvzone.putTextRect(img, "Speed ARight:" f'{classificator.velprintUR} cm/s', (800, 400))
                # cvzone.putTextRect(img, "Speed KLeft:" f'{classificator.velprintKL} cm/s', (800, 450))
                # cvzone.putTextRect(img, "Speed KRight:" f'{classificator.velprintKR} cm/s', (800, 500))
                # cvzone.putTextRect(img, "Speed HLeft:" f'{classificator.velprintHL} cm/s', (800, 550))
                # cvzone.putTextRect(img, "Speed HRight:" f'{classificator.velprintHR} cm/s', (800, 600))
                #cvzone.putTextRect(img, "acc:" f'{round(classificator.accprint, 1)} cm/s^2', (800, 400))
                #cvzone.putTextRect(img, "jerk:" f'{round(abs(printjerk), 2)} cm/s^3', (800, 450))
                #cvzone.putTextRect(img, "CI:" f'{round(classificator.ci, 2)}', (800, 500))
                #cvzone.putTextRect(img, "weight:" f'{round(weightprint, 2)}', (800, 550))
                # cvzone.putTextRect(img, "joint:" f'{evaluatedJoint}', (800, 600))
                # cvzone.putTextRect(img, "pivot:" f'{pivot}', (800, 650))
                classificator.start_time = actualtime
                #Writing
                writetext = f'{datetime.datetime.now()}' + "," + f'{classificator.stateUp}'+"," + f'{classificator.stateLow}'+","\
                            + f'{classificator.lowerbodyMov}'+","+f'{classificator.listVel[-1][0]}' + "," \
                            + f'{classificator.listVel[-1][1]}' + "," + f'{classificator.listVel[-1][2]}' + "," + \
                            f'{classificator.listVel[-1][3]}' + "," + f'{classificator.listVel[-1][4]}' + "," +\
                            f'{classificator.listVel[-1][5]}' + "," + f'{classificator.listAcc[-1][0]}' + "," +\
                            f'{classificator.listAcc[-1][1]}' + "," + f'{classificator.listAcc[-1][2]}' + "," + \
                            f'{classificator.listAcc[-1][3]}' + "," + f'{classificator.listAcc[-1][4]}' + "," + \
                            f'{classificator.listAcc[-1][5]}' + "," + f'{classificator.listJerk[-1][0]}' + "," + \
                            f'{classificator.listJerk[-1][1]}' + "," + f'{classificator.listJerk[-1][2]}' + "," + \
                            f'{classificator.listJerk[-1][3]}' + "," + f'{classificator.listJerk[-1][4]}' + "," + \
                            f'{classificator.listJerk[-1][5]}' + "," + f'{classificator.listWeight[-1][0]}' + "," +\
                            f'{classificator.listWeight[-1][1]}' + "," + f'{classificator.listWeight[-1][2]}' + "," + \
                            f'{classificator.listWeight[-1][3]}' + "," + f'{classificator.listWeight[-1][4]}' + "," + \
                            f'{classificator.listWeight[-1][5]}' + "," + f'{round(classificator.ci, 2)}' + "\n"
                f.write(writetext)

            img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()
