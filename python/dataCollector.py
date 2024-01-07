import cv2
import numpy as np
from iBinCom import iBinCom
import time
import os

width = 640
height = 480

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

iBin = iBinCom("COM3", 115200)
ret = iBin.open()

last = np.zeros((height, height, 3), np.uint8)
weight = 0
lastClass = 9999

try:
    labels = np.loadtxt('dataset/labels.txt', delimiter=",", dtype=np.int32)
    currIndex = np.max(labels[:,0])+1
    labelsFile = open('dataset/labels.txt', 'a')
except:
    os.mkdir('dataset')
    labelsFile = open('dataset/labels.txt', 'a')
    currIndex = 0

time.sleep(1)
while True:

    iBin.setLight(not iBin.getLid())

    ret, frame = cap.read()
    frame = frame[:,int(width/2-height/2):int(width/2+height/2)]
    #diff = cv2.absdiff(frame, reference)
    #diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #diff = cv2.GaussianBlur(diff, (5,5), 0)
    #_, diff = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #diff = cv2.dilate(diff, kernel, iterations=10)
    #diff = cv2.erode(diff, kernel, iterations=11)
    #crop = np.zeros((height, height, 3), np.uint8)
    #crop[diff>0] = frame[diff>0]

    cv2.imshow('Live view', frame)
    cv2.putText(last, "Last class: " + str(lastClass), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(last, "Last weight: " + str(weight), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(last, "Plastic:      1", (10, height-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(last, "Cardboard:  2", (10, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(last, "Metal:       3", (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(last, "Glass:       4", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Last', last)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('1'):
        last = frame
        cv2.imwrite('dataset/'+str(currIndex)+'.png', last)
        weight = iBin.getWeight()
        lastClass = 1
        labelsFile.write(str(currIndex)+','+str(weight)+',1\n')
        currIndex += 1
    elif key == ord('2'):
        last = frame
        cv2.imwrite('dataset/'+str(currIndex)+'.png', last)
        weight = iBin.getWeight()
        lastClass = 2
        labelsFile.write(str(currIndex)+','+str(weight)+',2\n')
        currIndex += 1
    elif key == ord('3'):
        last = frame
        cv2.imwrite('dataset/'+str(currIndex)+'.png', last)
        weight = iBin.getWeight()
        lastClass = 3
        labelsFile.write(str(currIndex)+','+str(weight)+',3\n')
        currIndex += 1
    elif key == ord('4'):
        last = frame
        cv2.imwrite('dataset/'+str(currIndex)+'.png', last)
        weight = iBin.getWeight()
        lastClass = 4
        labelsFile.write(str(currIndex)+','+str(weight)+',4\n')
        currIndex += 1

iBin.close()
cap.release()
labelsFile.close()
cv2.destroyAllWindows()
