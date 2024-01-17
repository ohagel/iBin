from iBinCom import iBinCom
import time

iBin = iBinCom(port="COM3", baudrate=115200, capDevice=0)
ret = iBin.open()

time.sleep(1)

while True:
    iBin.setLight(not iBin.getLid())
    print(not iBin.getLid())