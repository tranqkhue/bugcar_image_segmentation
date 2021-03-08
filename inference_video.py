import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import cv2
import rospy
import time

from calibration import INPUT_SHAPE
from bev import bev_transform_tools
import occgrid_to_ros
from utils import contour_noise_removal, enet_preprocessing
from models import ENET
logger = logging.Logger('lmao')

#================================================================================

#---------------------------------------------------------------------------------

#================================================================================
# Initialize
publisher = occgrid_to_ros.init_node()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], \
         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=600)])
    except RuntimeError as e:
        print(e)
# model = tf.keras.models.load_model('model.hdf5')
model = ENET('pretrained_models/enet.pb')
perspective_transformer = bev_transform_tools.fromJSON(
    'calibration_data_30fps.json')
matrix = perspective_transformer._bev_matrix
#print("Check model input:  ",model.inputs)
cap = cv2.VideoCapture("test1.webm")
width = INPUT_SHAPE[0]
height = INPUT_SHAPE[1]
cap.set(3, width)
cap.set(4, height)

#---------------------------------------------------------------------------------

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if (ret == True):
        # Prepocessing input
        batch_frame = enet_preprocessing(frame)
        time_get_frame_ros = rospy.Time.now()

        # Run inference and process the results
        t1 = time.time()
        inference_result = model.predict(batch_frame)[0]
        inference_fps = 1 / (time.time() - t1)
        result_by_class = np.argmax(inference_result, axis=0)
        segmap    = np.bitwise_or(result_by_class==0, result_by_class==1)\
                .astype(np.uint8)

        # Remove road branches (or noise) that are not connected to main branches
        # Main road branches go from the bottom part of the RGB map
        # (should be) right front of the vehicle
        contour_noise_removed = contour_noise_removal(segmap)
        # Visualize the segmap by masking the RGB frame
        # (out_height, out_width) = contour_noise_removed.shape
        # resized_frame = cv2.resize(frame, (out_width, out_height))
        # segmap_viz    = cv2.bitwise_and(resized_frame, resized_frame, \
        #         mask=contour_noise_removed)
        # enlarged_viz = cv2.resize(segmap_viz, (0, 0), fx=3, fy=3)
        # cv2.imshow('segmap_cnt_noise_removal', cv2.cvtColor(enlarged_viz, \
        #              cv2.COLOR_RGB2BGR))

        # Visualize the BEV by masking and warp the RGB frame
        # Resize the segmap to scale with the calibration matrix
        # resized_segmap_viz = cv2.resize(segmap_viz, (1024, 512))
        # warped_perspective_viz = cv2.warpPerspective(resized_segmap_viz, \
        #             matrix,(1024,512))
        # cv2.imshow('warped_perspective', cv2.cvtColor(warped_perspective_viz, \
        #              cv2.COLOR_RGB2BGR))

        # Publish to Occupancy Grid
        # Need to resize to be the same with the image size in calibration process
        # print(np.histogram(contour_noise_removed))
        resized_segmap = cv2.resize(contour_noise_removed, INPUT_SHAPE)
        occ_grid = perspective_transformer.create_occupancy_grid(
            resized_segmap, perspective_transformer._bev_matrix,
            perspective_transformer.width, perspective_transformer.height,
            perspective_transformer.map_size,
            perspective_transformer.map_resolution,
            perspective_transformer.cm_per_px)
        msg    = occgrid_to_ros.og_msg(occ_grid,\
                 perspective_transformer.map_resolution,\
                 perspective_transformer.map_size,time_get_frame_ros)
        publisher.publish(msg)
        cv2.imshow("frame", frame)
        cv2.imshow("occgrid", occ_grid)
        print('Inference FPS:  ',  format(inference_fps, '.2f'), ' | ',\
           'Total loop FPS:  ', format(1/(time.time()-t0), '.2f'))
    if (cv2.waitKey(1) & 0xFF == ord('q')) | (ret == False):
        cap.release()
        cv2.destroyAllWindows()
        break
