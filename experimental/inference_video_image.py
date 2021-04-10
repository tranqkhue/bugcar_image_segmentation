#!/home/tranquockhue/anaconda3/envs/cvdl/bin/python
from utils import contour_noise_removal, create_skeleton, enet_preprocessing, testDevice
from models import ENET
from bev import bev_transform_tools
import time
import cv2
import numpy as np
import tensorflow as tf
import logging
import os
import gc
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.Logger('lmao')

# ================================================================================

# ---------------------------------------------------------------------------------

# ================================================================================
# Initialize

# ---------------------------------------------------------------------------------
# @profile
def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("GPU",gpus)
    if gpus:
        try:
            #tf.config.experimental.allo
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1000,)])
        except RuntimeError as e:
            print(e)
    model = ENET('./pretrained_models/enet.pb')
    perspective_transformer = bev_transform_tools.fromJSON(
        'calibration_data_30fps.json')


    #
    # for debugging with video:
    # cap = cv2.VideoCapture("test1.webm")
    frame = cv2.imread("img/3.png")
    t0 = time.time()
    ret = True
    #ret, frame = cap.read()
    if (ret == True):
        for i in range(10000):
            # Prepocessing input
            batch_frame = enet_preprocessing(frame)
            # Run inference and process the results
            t1 = time.time()
            inference_result = model.predict(batch_frame)[0]
            inference_fps = 1 / (time.time() - t1)
            print(inference_fps)
            # result_by_class = np.argmax(inference_result, axis=0)
            # segmap = np.bitwise_or(result_by_class == 0, result_by_class == 1)\
                # .astype(np.uint8)
            # Remove road branches (or noise) that are not connected to main branches
            # Main road branches go from the bottom part of the RGB map
            # (should be) right front of the vehicle        # Need to resize to be the same with the image size in calibration process

            # edges = cv2.Laplacian(occ_grid.astype(np.int16),
            #                       ddepth=cv2.CV_16S, ksize=3)
            # edges = ((edges + 404)/1212*255).astype(np.uint8)
            # #cv2.imshow("lap", edges)
            # print(np.histogram(edges))
            # buh_edges = cv2.subtract(edges, buh_mask)
            # buh_edges[buh_edges < 250] = 0
            # buh_edges = cv2.dilate(buh_edges, np.ones((2, 2)))
            # buh_edges = buh_edges.astype(np.bool)

main()