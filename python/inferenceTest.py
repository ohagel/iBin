from iBinNet import Net
import cv2
import numpy as np
from iBinCom import iBinCom
import time
import os

iBin = iBinCom("COM5", 115200, 0)
ret = iBin.open()

classes = ('plastic', 'cardboard', 'metal', 'glass')

time.sleep(1)

if __name__ == '__main__':



    net = Net('dataset/labels.txt', 'dataset', 0.2)



    net.load('iBin_net.pth')
    #net.train(2)
    #net.validate()

    while True:
        frame = iBin.getFrame()
        shownFrame = frame
        if not iBin.getLid():
            iBin.setLight(True)
            
            weight = iBin.getWeight()
            res = net.infer(frame , weight)
            cv2.putText(shownFrame, "Class: " + str(classes[res]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(shownFrame, "Weight: " + str(weight), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            iBin.setLight(False)
            cv2.putText(shownFrame, "Lid open", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Live view', shownFrame)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break

    iBin.close()
    iBin.cap.release()
    cv2.destroyAllWindows()