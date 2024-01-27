from iBinNet2 import Net
import cv2
from iBinCom import iBinCom
import time
import numpy as np

if __name__ == '__main__':
    #init iBin communication
    iBin = iBinCom(port="COM3", baudrate=115200, capDevice=0)
    ret = iBin.open()
    time.sleep(1) #small delay to let the microcontroller boot

    #defining classes
    classes = ('Plastic', 'Cardboard', 'Metal', 'Glass') 
    #init neural network 
    net = Net(device='cuda:0')
    #Load trained model
    net.load('iBin_net2_23-01-2024_13;49.pth')

    #main inference loop
    startTime = 0
    lidTime = 0
    while True:
        endTime = time.time()
        frame = iBin.getFrame()
        shownFrame = frame 
        textFrame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        if not iBin.getLid():
            iBin.setLight(255)
            if time.time() - lidTime > 0.5: #delay inference to let the camera adjust to the light
                weight = iBin.getWeight()
                res = net.infer(frame , weight)
                cv2.putText(textFrame, "Class: " + str(classes[res]), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)#, cv2.LINE_AA)
                cv2.putText(textFrame, "Weight: " + str(weight), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)#, cv2.LINE_AA)
            cv2.putText(textFrame, "FPS: " + "{:.2f}".format(1/(endTime-startTime)), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)#, cv2.LINE_AA)
        else:
            lidTime = time.time()
            iBin.setLight(0)
            cv2.putText(textFrame, "Lid open", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)#, cv2.LINE_AA)
            cv2.putText(textFrame, "FPS: " + "{:.2f}".format(1/(endTime-startTime)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)#, cv2.LINE_AA)
        shownFrame[textFrame[:,:,0] > 0] = 255 - shownFrame[textFrame[:,:,0] > 0]
        cv2.imshow('Live view', shownFrame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        startTime = endTime

    iBin.close()
    iBin.cap.release()
    cv2.destroyAllWindows()