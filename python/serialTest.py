from iBinCom import iBinCom
import time

com = iBinCom("COM3", 115200)
ret = com.open()
if ret:
    print("open success")
    com.setLight(True)
    time.sleep(1)
    com.setLight(False)
    print(com.getLid())
com.close()

#val = 11111112
#print(val)
#data = val.to_bytes(4, byteorder='big')
#print(data[0], data[1], data[2], data[3])
#val2 = int.from_bytes(data, byteorder='big')
#print(val2)