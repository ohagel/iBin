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
            return True
        except:
            return False

    def close(self):
        if self.opened:
            self.ser.close()

    def setLight(self, light):
        if self.opened:
            pass

    def getWeight(self):
        if self.opened:
            pass
        
    def getLid(self):
        if self.opened:
            pass

    def test(self):
        print("test")