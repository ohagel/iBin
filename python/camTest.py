import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()
    cv2.imshow('Live view', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break