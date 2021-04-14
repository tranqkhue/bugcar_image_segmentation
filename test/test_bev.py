from calibration import WARPED_IMG_SHAPE
from bev import bev_transform_tools
import numpy as np
import time
import cv2


def test_bev():
    transformer = bev_transform_tools.fromJSON("calibration_data_30fps.json")
    matrix = transformer._bev_matrix
    cap = cv2.VideoCapture(6)
    assert cap.isOpened()
    cap.set(3, 1280)
    cap.set(4, 720)
    inference_size = (1024, 512)
    while True:
        ret, frame = cap.read()
        if ret == True:
            t0 = time.time()
            frame = cv2.resize(frame, inference_size)
          ##cv2.imshow('raw', frame)
            matrix = transformer._bev_matrix
            warped_image = cv2.warpPerspective(frame, matrix, WARPED_IMG_SHAPE)
          ##cv2.imshow('warped', warped_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    test_bev()
