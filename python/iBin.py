from iBinNet2 import Net
import cv2
from iBinCom import iBinCom
import time

if __name__ == '__main__':
    #init iBin communication
    iBin = iBinCom(port="COM3", baudrate=115200, capDevice=1)
    ret = iBin.open()
    time.sleep(1) #small delay to let the microcontroller boot

    #defining classes
    classes = ('plastic', 'cardboard', 'metal', 'glass') 
    #init neural network 
    net = Net(device='cuda:0')
    #Load trained model
    net.load('iBin_net2_16-01-2024_19;40.pth')

    #main inference loop
    startTime = 0
    while True:
        endTime = time.time()
        frame = iBin.getFrame()
        shownFrame = frame 
        if not iBin.getLid():
            iBin.setLight(255)
            weight = iBin.getWeight()
            res = net.infer(frame , weight)
            cv2.putText(shownFrame, "Class: " + str(classes[res]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(shownFrame, "Weight: " + str(weight), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(shownFrame, "FPS: " + "{:.2f}".format(1/(endTime-startTime)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            iBin.setLight(0)
            cv2.putText(shownFrame, "Lid open", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(shownFrame, "FPS: " + "{:.2f}".format(1/(endTime-startTime)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Live view', shownFrame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        startTime = endTime

    iBin.close()
    iBin.cap.release()
    cv2.destroyAllWindows()