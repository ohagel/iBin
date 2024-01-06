import serial

class iBinCom:
    def __init__(self, port, baudrate):
        self.opened = False
        self.port = port
        self.baudrate = baudrate

    def open(self):
        if self.opened:
            return True
        try:
            self.ser = serial.Serial(self.port, self.baudrate)
            self.opened = True
            return True
        except:
            return False

    def close(self):
        if self.opened:
            self.ser.close()

    def sendPacket(self, type, addr, data):
        if self.opened:
            data = data.to_bytes(4, byteorder='big')
            bytePacket = bytearray([type,addr,data[0],data[1],data[2],data[3]])
            print(bytePacket)
            self.ser.write(bytePacket)

    def recivePacket(self):
        if self.opened:
            recvData = self.ser.read(6)
            print(recvData)
            return int.from_bytes(recvData[2:], byteorder='big')

    def setLight(self, state):
        if self.opened:
            if state:
                self.sendPacket(1, 2, 255)
            else:
                self.sendPacket(1, 2, 0)

    def getWeight(self):
        if self.opened:
            pass
        
    def getLid(self):
        if self.opened:
            self.sendPacket(0, 0, 0)
            return self.recivePacket()


    def test(self):
        print("test")