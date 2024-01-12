from iBinNet import Net
import cv2
from iBinCom import iBinCom
import time

if __name__ == '__main__':
    #init iBin communication
    iBin = iBinCom(port="COM5", baudrate=115200, capDevice=0)
    ret = iBin.open()
    time.sleep(1) #small delay to let the microcontroller boot

    #defining classes
    classes = ('plastic', 'cardboard', 'metal', 'glass') 
    #init neural network 
    net = Net(device='cuda:0')
    #Load trained model
    net.load('iBin_net_12-01-2024_12;16.pth')

    lidOpenTime = time.time()

    #main inference loop
    while True:
        frame = iBin.getFrame()
        shownFrame = frame 
        print("open time", lidOpenTime, "time", time.time(), "diff", time.time() - lidOpenTime)
        if not iBin.getLid():
            iBin.setLight(True)
            if time.time() - lidOpenTime > 0.5: 
                weight = iBin.getWeight()
                res = net.infer(frame , weight)
                cv2.putText(shownFrame, "Class: " + str(classes[res]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(shownFrame, "Weight: " + str(weight), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            lidOpenTime = time.time()
            iBin.setLight(False)
            cv2.putText(shownFrame, "Lid open", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Live view', shownFrame)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break

    iBin.close()
    iBin.cap.release()
    cv2.destroyAllWindows()