from iBinNet import Net
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

weight = 0

time.sleep(1)

classes = ('plastic', 'cardboard', 'metal', 'glass')

if __name__ == '__main__':



    net = Net('dataset/labels.txt', 'dataset', 0.2, device = 'cpu')



    net.load('iBin_net_73.pth')
    #net.train(100)
    net.validate()


    while True:
        frame = iBin.getFrame()
        shownFrame = frame
        if not iBin.getLid():
            iBin.setLight(True)
            
            weight = iBin.getWeight()
            res = net.infer(frame , weight)
            cv2.putText(shownFrame, "Class: " + str(classes[res]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(shownFrame, "Weight: " + str(weight), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            #time.sleep(2)
        else:
            iBin.setLight(False)
            cv2.putText(shownFrame, "Lid open", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Live view', shownFrame)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        """elif key == ord('1'):
            #print("infer",frame.shape, type(frame), frame.dtype)
            weight = iBin.getWeight()
            res = net.infer(frame , weight)
            print("res print",res)"""

    iBin.close()
    cap.release()
    cv2.destroyAllWindows()
    


