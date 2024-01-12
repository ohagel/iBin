from iBinCom import iBinCom

iBin = iBinCom(port="COM5", baudrate=115200, capDevice=0)
ret = iBin.open()

while True:
    print(iBin.getLid())