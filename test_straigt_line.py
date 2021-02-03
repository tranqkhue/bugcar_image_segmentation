from bev import bev_transform_tools
import tensorflow as tf
import numpy as np
import time
import cv2

transformer = bev_transform_tools.fromJSON("calibration_data.json")
matrix = transformer._intrinsic_matrix
cap = cv2.VideoCapture(6)
cap.set(3, 1280)
cap.set(4, 720)
inference_size = (1024, 512)
while True:
    ret, frame = cap.read()
    if ret == True:
        t0 = time.time()
        frame = cv2.resize(frame, inference_size)
        # frame = cv2.rotate(frame, cv2.cv2.ROTATE_180)
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow('raw', frame)
        '''
        matrix = np.array([[ 1.48279301e+01, -2.73925685e+02,  2.54431148e+04],\
                           [ 3.46265488e+01, -2.32236650e+02,  1.60349804e+04],\
                           [ 4.74466674e-02, -3.95920274e-01,  1.00000000e+00]])
        '''
        derotation = transformer.dero

        detranslation = transformer.detran

        M = transformer.M
        matrix = np.matmul(derotation, M)
        # matrix = np.matmul(detranslation, matrix)
        warped_image = cv2.warpPerspective(frame, matrix, (1024, 1024))
        m_only = cv2.warpPerspective(frame, M, (1024, 512))
        cv2.imshow('warped', warped_image)
        cv2.imshow("only M", m_only)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
