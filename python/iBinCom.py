import serial
import time
import os
import cv2
import numpy as np

class iBinCom:
    def __init__(self, port, baudrate, capDevice=1):

        self.opened = False
        self.port = port
        self.baudrate = baudrate
        self.width = 640
        self.height = 480
        self.cap = cv2.VideoCapture(capDevice, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.lid = 1

        self.weight = 0

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
            #print(bytePacket)
            self.ser.write(bytePacket)

    def recivePacket(self):
        if self.opened:
            recvData = self.ser.read(6)
            #print(recvData)
            return int.from_bytes(recvData[2:], byteorder='big', signed=True)

    def setLight(self, state):
        if self.opened:
            #if state:
            #    self.sendPacket(1, 2, 255)
            #else:
            #    self.sendPacket(1, 2, 0)
            self.sendPacket(1, 2, state)

    def getWeight(self):
        if self.opened:
            self.sendPacket(0, 1, 0)
            return self.recivePacket()
        
    def getLid(self):
        if self.opened:
            self.sendPacket(0, 0, 0)
            return False if self.recivePacket() < 1500 else True


    def test(self):
        print("test")

    def getFrame(self):
        ret, frame = self.cap.read()
        frame = frame[:,int(self.width/2-self.height/2):int(self.width/2+self.height/2)]
        return frame
