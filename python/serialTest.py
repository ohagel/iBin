from iBinCom import iBinCom

com = iBinCom("COM3", 115200)
ret = com.open()
print(ret)
com.test()
com.close()
