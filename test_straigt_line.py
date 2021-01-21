from bev import setupBEV
import tensorflow as tf
import numpy as np
import time
import cv2

perspective_transformer,matrix = setupBEV()
cap = cv2.VideoCapture(6)
cap.set(3, 1920)
cap.set(4, 1080)
inference_size = (1024,512)
while True:
        ret, frame = cap.read()
        if ret == True:
            t0 = time.time()
            frame = cv2.resize(frame,inference_size)
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_180)
            # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2.imshow('raw',frame)
            warped_image = cv2.warpPerspective(frame,matrix,(1024,512))
            cv2.imshow('warped',warped_image)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
        	cap.release()
        	cv2.destroyAllWindows()
        	break	