import cv2
from cvzone.PoseModule import PoseDetector
import socket
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
cap = cv2.VideoCapture('trackingtest.mp4')  # 0 or 1 for webcam
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
        self.velprint = 0
        self.accprint = 0
        self.printjerk = 0
        self.running_time = time.time()
        self.start_time = time.time()
        self.velFilter = []
        self.velFilter.append(0)
        self.jointsInitialization()

    def jointsInitialization(self):
            self.shoulder_left =[]
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

    def jointsAppending(self, lmlist):
        count = 1
        if(count == 1):
            self.shoulder_left.append(lmlist[11][1:])
            self.shoulder_right.append(lmlist[12][1:])
            self.elbow_left.append(lmlist[13][1:])
            self.elbow_right.append(lmlist[14][1:])
            self.wrist_left.append(lmlist[15][1:])
            self.wrist_right.append(lmlist[16][1:])
            self.pynky_left.append(lmlist[17][1:])
            self.pynky_right.append(lmlist[18][1:])
            self.index_left.append(lmlist[19][1:])
            self.index_right.append(lmlist[20][1:])
            self.thumb_left.append(lmlist[21][1:])
            self.thumb_right.append(lmlist[22][1:])
            self.hip_left.append(lmlist[23][1:])
            self.hip_right.append(lmlist[24][1:])
            self.knee_left.append(lmlist[25][1:])
            self.knee_right.append(lmlist[26][1:])
            self.ankle_left.append(lmlist[27][1:])
            self.ankle_right.append(lmlist[28][1:])
            self.heel_left.append(lmlist[29][1:])
            self.heel_right.append(lmlist[30][1:])
            self.foot_left.append(lmlist[31][1:])
            self.foot_right.append(lmlist[32][1:])
            count = count+1
def main():
    # Communication
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # serverAddressPort = ("127.0.0.1", 5053)
    classificator = Classificator()
    lowerbodyMov = ""
    evaluatedJoint = ""
    pivot = ""
    f = open("demofile2.csv", "a")
    headers = "movement, velocity, acc, jerk, ci, weight,evaluatedjoint, pivot\n"
    f.write(headers)
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)
        #classificator.jointsAppending(lmList)

        if lmList:
            actualtime = time.time()
            timer = actualtime - classificator.start_time

            classificator.jointsAppending(lmList)
            #Lowerbody movements
            #Plier -> Positive displacement of LeftKnee.x; Negative displacement of RightKnee.x; Negative (Positive because of plane) displacement of hips.y

            if(classificator.knee_right.__len__() > 2):
                if ((classificator.knee_left[-1][0] - classificator.knee_left[-2][0] > 0.1) &
                        (classificator.knee_right[-1][0] - classificator.knee_right[-2][0] < -0.1) &
                        (classificator.hip_left[-1][1] - classificator.hip_left[-2][1] > 0.1) &
                        (classificator.hip_right[-1][1] - classificator.hip_right[-2][1] > 0.1)):
                            classificator.listPosXLower.clear()
                            classificator.listPosYLower.clear()
                            classificator.listPosZLower.clear()
                            for item in classificator.hip_right:
                               classificator.listPosXLower.append(item[0])
                               classificator.listPosYLower.append(item[1])
                               classificator.listPosZLower.append(item[2])
                            xP, yP, zP = classificator.knee_right[-1][0:]
                            evaluatedJoint = "hip right"
                            pivot = "knee right"
                            lowerbodyMov = "Plie"
                if ((classificator.knee_left[-1][0] - classificator.knee_left[-2][0] < -0.3) &
                        (classificator.knee_right[-1][0] - classificator.knee_right[-2][0] > 0.3) &
                        (classificator.hip_left[-1][1] - classificator.hip_left[-2][1] < -0.3) &
                        (classificator.hip_right[-1][1] - classificator.hip_right[-2][1] < -0.3)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.hip_right:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.knee_right[-1][0:]
                    evaluatedJoint = "hip right"
                    pivot = "knee right"
                    lowerbodyMov = "Back from Plie"
            #Releve -> Negative displacement of hips.y; Negative displacement of heels.y; Not foot displacement in y and x
                if ((classificator.hip_left[-1][1] - classificator.hip_left[-2][1] < -0.3) &
                        (classificator.hip_right[-1][1] - classificator.hip_right[-2][1] < -0.3) &
                        (classificator.heel_left[-1][1] - classificator.heel_left[-2][1] < -0.3) &
                        (classificator.heel_right[-1][1] - classificator.heel_right[-2][1] < -0.3) &
                        (classificator.foot_right[-1][1] - classificator.foot_right[-2][1] < -0.3) &
                        (classificator.foot_left[-1][1] - classificator.foot_left[-2][1] < -0.3)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.hip_right:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.knee_right[-1][0:]
                    evaluatedJoint = "hip right"
                    pivot = "knee right"
                    lowerbodyMov = "Releve"
                if ((classificator.hip_left[-1][1] - classificator.hip_left[-2][1] > 0.3) &
                        (classificator.hip_right[-1][1] - classificator.hip_right[-2][1] > 0.3) &
                        (classificator.heel_left[-1][1] - classificator.heel_left[-2][1] > 0.3) &
                        (classificator.heel_right[-1][1] - classificator.heel_right[-2][1] > 0.3) &
                        (classificator.foot_right[-1][1] - classificator.foot_right[-2][1] > 0.3) &
                        (classificator.foot_left[-1][1] - classificator.foot_left[-2][1] > 0.3)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.hip_right:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.knee_right[-1][0:]
                    evaluatedJoint = "hip right"
                    pivot = "knee right"
                    lowerbodyMov = "Back from Releve"
            # Etendre Right -> Not displacement in left hip.y; Negative displacement in right foot.x;  Not displacement in left foot.x;
                if ((classificator.foot_right[-1][0] - classificator.foot_right[-2][0] < -1) &
                    (classificator.heel_right[-1][0] - classificator.heel_right[-2][0] < -1) &
                    (abs(classificator.hip_right[-1][1] - classificator.hip_right[-2][1]) < 0.1) &
                    (abs(classificator.hip_left[-1][1] - classificator.hip_left[-2][1]) < 0.1) &
                    (classificator.heel_right[-1][1] - classificator.heel_right[-2][1] < -0.3)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.foot_right:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.ankle_right[-1][0:]
                    evaluatedJoint = "foot right"
                    pivot = "ankle right"
                    lowerbodyMov = "Etendre right"
                if ((classificator.foot_right[-1][0] - classificator.foot_right[-2][0] > 1) &
                    (classificator.heel_right[-1][0] - classificator.heel_right[-2][0] > 1) &
                    (abs(classificator.hip_right[-1][1] - classificator.hip_right[-2][1]) < 0.1) &
                    (abs(classificator.hip_left[-1][1] - classificator.hip_left[-2][1]) < 0.1) &
                    (classificator.heel_right[-1][1] - classificator.heel_right[-2][1] > 1.5)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.foot_right:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.ankle_right[-1][0:]
                    evaluatedJoint = "foot right"
                    pivot = "ankle right"
                    lowerbodyMov = "Back from Etendre right"
            # Etendre Left -> Not displacement in right hip.y; Positive displacement in left foot.x or z;  Not displacement in right foot.x;
                if ((classificator.foot_left[-1][0] - classificator.foot_left[-2][0] > 1) &
                    (classificator.heel_left[-1][0] - classificator.heel_left[-2][0] > 1) &
                    (abs(classificator.hip_right[-1][1] - classificator.hip_right[-2][1]) < 0.1) &
                    (abs(classificator.hip_left[-1][1] - classificator.hip_left[-2][1]) < 0.1) &
                    (classificator.heel_left[-1][1] - classificator.heel_left[-2][1] < -0.3)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.foot_left:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.ankle_left[-1][0:]
                    evaluatedJoint = "foot left"
                    pivot = "ankle left"
                    lowerbodyMov = "Etendre left"
                if ((classificator.foot_left[-1][0] - classificator.foot_left[-2][0] < -0.3) &
                    (classificator.heel_left[-1][0] - classificator.heel_left[-2][0] < -0.3) &
                    (abs(classificator.hip_right[-1][1] - classificator.hip_right[-2][1]) < 0.1) &
                    (abs(classificator.hip_left[-1][1] - classificator.hip_left[-2][1]) < 0.1) &
                    (classificator.heel_left[-1][1] - classificator.heel_left[-2][1] > 0.3)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.foot_left:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.ankle_left[-1][0:]
                    evaluatedJoint = "foot left"
                    pivot = "ankle left"
                    lowerbodyMov = "Back from Etendre left"
            #Tourner ->Not hips displacement in y; hips.x displacement any side; hips.z displacement any side
                if ((abs(classificator.hip_left[-1][0] - classificator.hip_left[-2][0]) > 1) &
                        (abs(classificator.hip_right[-1][0] - classificator.hip_right[-2][0]) > 1) &
                        (abs(classificator.hip_left[-1][2] - classificator.hip_left[-2][2]) > 1) &
                        (abs(classificator.hip_right[-1][2] - classificator.hip_right[-2][2]) > 1) &
                        (abs(classificator.foot_right[-1][0] - classificator.foot_right[-2][0]) > 0.5) &
                        (abs(classificator.foot_left[-1][0] - classificator.foot_left[-2][0]) > 0.5) &
                        (abs(classificator.hip_left[-1][1] - classificator.hip_left[-2][1]) < 0.05)
                ):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.hip_right:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.knee_right[-1][0:]
                    evaluatedJoint = "hip right"
                    pivot = "knee right"
                    lowerbodyMov = "Tourner"
            #Sauter -> Jumping; Negative in foot.y; Negative displacement in hips.y; Not displacement in hips.x any side; Negative displacement in heel.y
                if ((classificator.foot_left[-1][1] - classificator.foot_left[-2][1] < -1) &
                    (classificator.foot_right[-1][1] - classificator.foot_right[-2][1] < -1) &
                    (classificator.hip_left[-1][1] - classificator.hip_left[-2][1] < -0.8) &
                    (classificator.hip_right[-1][1] - classificator.hip_right[-2][1] < -0.8) &
                    (classificator.heel_left[-1][1] - classificator.heel_left[-2][1] < -1) &
                    (classificator.heel_right[-1][1] - classificator.heel_right[-2][1] < -1) &
                    (abs(classificator.hip_right[-1][0] - classificator.hip_right[-2][0]) < 0.05)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.hip_right:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.knee_right[-1][0:]
                    evaluatedJoint = "hip right"
                    pivot = "knee right"
                    lowerbodyMov = "Jumping"
            # Elancer -> Jumping with direction; Positive Negative in foot.y; Negative displacement in hips.y; Displacement in hips.x any side
                if ((classificator.foot_left[-1][1] - classificator.foot_left[-2][1] < -0.8) &
                    (classificator.foot_right[-1][1] - classificator.foot_right[-2][1] < -0.8) &
                    (classificator.hip_left[-1][1] - classificator.hip_left[-2][1] < -0.8) &
                    (classificator.hip_right[-1][1] - classificator.hip_right[-2][1] < -0.8) &
                    (classificator.heel_left[-1][1] - classificator.heel_left[-2][1] < -0.5) &
                    (classificator.heel_right[-1][1] - classificator.heel_right[-2][1] < -0.5) &
                    (abs(classificator.hip_right[-1][0] - classificator.hip_right[-2][0]) > 0.5)):
                    classificator.listPosXLower.clear()
                    classificator.listPosYLower.clear()
                    classificator.listPosZLower.clear()
                    for item in classificator.hip_right:
                        classificator.listPosXLower.append(item[0])
                        classificator.listPosYLower.append(item[1])
                        classificator.listPosZLower.append(item[2])
                    xP, yP, zP = classificator.knee_right[-1][0:]
                    evaluatedJoint = "hip right"
                    pivot = "knee right"
                    lowerbodyMov = "Elancer"
                cvzone.putTextRect(img, lowerbodyMov, (800,300))


            #x1, y1, z1 = classificator.wrist_right[-1]  # lmList[15][1:]
            #xP, yP, zP = lmList[11][1:]
            # x3, y3, z3 = lmList[23][1:]
            # Calculate the Angle
            # angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
            # math.atan2(y1 - y2, x1 - x2))
            # if angle < 0:
            # angle += 360
            # print(angle)

            #classificator.listPosXUpper.append(x1)
            #classificator.listPosYUpper.append(y1 * (-1))
            #classificator.listPosZUpper.append(z1)

            #if (classificator.listPosYUpper[-1] - classificator.listPosYUpper[-2] > 1):
            #    cvzone.putTextRect(img, "Ascending arms", (width + 150, height - 150))  # 100,400
            #if (classificator.listPosYUpper[-1] - classificator.listPosYUpper[-2] < -1):
            #    cvzone.putTextRect(img, "Descending arms", (width + 150, height - 150))  # 100,400
                if (classificator.listPosXLower.__len__() > 2):
                    print("xi: ",classificator.listPosXLower[-1], " x-1:",classificator.listPosXLower[-2])
                    # Get displacement in each plane
                    difX = classificator.listPosXLower[-1] - classificator.listPosXLower[-2]
                    difY = (classificator.listPosYLower[-1] - classificator.listPosYLower[-2])*(-1)
                    difZ = classificator.listPosZLower[-1] - classificator.listPosZLower[-2]
                    # Calculate total displacement
                    dif = math.sqrt((difX) ** 2 + (difY) ** 2 + (difZ) ** 2)
                    # Calculate coefficients to transform px to cm
                    A, B, C = classificator.coff
                    # Calculate real distance (cm)
                    distanceCM = A * dif ** 2 + B * dif + C
                    classificator.listDespCM.append(distanceCM)
                    # Calculate difference in displacement
                    desp = classificator.listDespCM[-1] - classificator.listDespCM[-2]

                    # print(dif)
                    # Calculate velocity in time (sec)
                    vel = abs(desp / timer)
                    classificator.listVel.append(vel)
                    # Calculate acceleration from difference in velocity
                    acc = round(((classificator.listVel[-1] - classificator.listVel[-2]) / timer), 2)
                    classificator.listAcc.append(acc)
                    classificator.listAcc[:] = [x / 10 for x in classificator.listAcc]
                    # accFilter = lowpass_filter(listAcc, 0.1, 1, order=2)
                    print("speed(tf):", round(classificator.listVel[-1], 3), "speed(ti):", round(classificator.listVel[-2], 3),
                          "acc:", classificator.listAcc[-1])

                    # Calculate jerkiness from difference in acceleration
                    jerk = ((classificator.listAcc[-1] - classificator.listAcc[-2]) / timer)
                    classificator.listJerk.append(jerk)
                    # jerkFilter = lowpass_filter(listJerk, 0.1, 1, order=4)#lfilter(b,a,listJerk)
                    # Calculate contraction index:
                    # ratio between 2 joints that surround another (middle) joint
                    ci = (math.sqrt(classificator.listPosXLower[-1] ** 2 + classificator.listPosYLower[-1]  ** 2 + classificator.listPosZLower[-1] ** 2)) / (math.sqrt(xP ** 2 + yP ** 2 + zP ** 2))

                    # Calculate the weight of the movement (Kinetic Energy)
                    weight = vel ** 2
                    print("jerk:", round(jerk, 3), "ci:", round(ci, 3), "weight:", weight)

                    if time.time() - classificator.running_time >= 0.1:
                        velprint = classificator.listVel[-1]  # vel
                        accprint = classificator.listAcc[-1]
                        printjerk = classificator.listJerk[-1]
                        weightprint = weight
                        classificator.running_time = time.time()

                    cvzone.putTextRect(img, "speed:" f'{int(velprint)} cm/s', (800, 350))  # 100,400
                    cvzone.putTextRect(img, "acc:" f'{round(accprint, 1)} cm/s^2',(800, 400))
                    cvzone.putTextRect(img, "jerk:" f'{round(abs(printjerk), 2)} cm/s^3', (800, 450))
                    cvzone.putTextRect(img, "CI:" f'{round(ci, 2)}', (800, 500))
                    cvzone.putTextRect(img, "weight:" f'{round(weightprint / 1000, 2)}', (800, 550))
                    cvzone.putTextRect(img, "joint:" f'{evaluatedJoint}', (800, 600))
                    cvzone.putTextRect(img, "pivot:" f'{pivot}', (800, 650))
                    classificator.start_time = actualtime
                    writetext = lowerbodyMov+","+ f'{velprint}'+","+f'{round(accprint)}'+","+f'{round(abs(printjerk), 2)}'+","+ f'{round(ci, 2)}'+","+f'{round(weightprint / 1000, 2)}'+","+f'{evaluatedJoint}'+","+f'{pivot}'+"\n"
                    f.write(writetext)

        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
