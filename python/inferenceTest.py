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

if __name__ == '__main__':



    net = Net('dataset/labels.txt', 'dataset', 0.2)



    net.load('iBin_net.pth')
    #net.train(10)
    net.validate()


    while True:
        iBin.setLight(not iBin.getLid())
        frame = iBin.getFrame()
        cv2.imshow('Live view', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('1'):
            weight = iBin.getWeight()
            res = net.infer(frame, weight)
            print(res)

    iBin.close()
    cap.release()
    cv2.destroyAllWindows()
    


