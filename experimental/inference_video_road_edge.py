from utils import contour_noise_removal, create_skeleton, enet_preprocessing, testDevice
from models import ENET
import occgrid_to_ros
from bev import bev_transform_tools
from calibration import INPUT_SHAPE
import pyrealsense2 as rs
import time
import rospy
import cv2
import numpy as np
import tensorflow as tf
import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.Logger('lmao')

# ================================================================================

# ---------------------------------------------------------------------------------

# ================================================================================
# Initialize
publisher = occgrid_to_ros.init_node()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=600)])
    except RuntimeError as e:
        print(e)
model = ENET('pretrained_models/enet.pb')
perspective_transformer = bev_transform_tools.fromJSON(
    'calibration_data_60fps.json')
INPUT_SHAPE = INPUT_SHAPE["60fps"]
width = INPUT_SHAPE[0]
height = INPUT_SHAPE[1]
# skel_img = cv2.imread("img/segmap_boundaries_in_occ_grid.jpg",
#                       cv2.IMREAD_GRAYSCALE)
skel_img = create_skeleton(perspective_transformer, INPUT_SHAPE)
buh_mask = cv2.imread("img/buh_mask.png", cv2.IMREAD_GRAYSCALE)


# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_device("841612070098")
# config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
# profile = pipeline.start(config)
#
# for debugging with video:
cap = cv2.VideoCapture("test1.webm")

# ---------------------------------------------------------------------------------

while True:
    t0 = time.time()
    # ret = True
    # pipeline_frames = pipeline.wait_for_frames()
    # pipeline_rgb_frame = pipeline_frames.get_color_frame()
    # frame = np.asanyarray(pipeline_rgb_frame.get_data())
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
        segmap = np.bitwise_or(result_by_class == 0, result_by_class == 1)\
            .astype(np.uint8)

        # Remove road branches (or noise) that are not connected to main branches
        # Main road branches go from the bottom part of the RGB map
        # (should be) right front of the vehicle
        contour_noise_removed = contour_noise_removal(segmap)

        # Need to resize to be the same with the image size in calibration process

        resized_segmap = cv2.resize(contour_noise_removed, INPUT_SHAPE)
        occ_grid = perspective_transformer.create_occupancy_grid(
            resized_segmap, perspective_transformer._bev_matrix,
            perspective_transformer.width, perspective_transformer.height,
            perspective_transformer.map_size,
            perspective_transformer.map_resolution,
            perspective_transformer.cm_per_px)
        # edges = cv2.Laplacian(occ_grid.astype(np.int16),
        #                       ddepth=cv2.CV_16S, ksize=3)
        # edges = ((edges + 404)/1212*255).astype(np.uint8)
        # #cv2.imshow("lap", edges)
        # print(np.histogram(edges))
        edges = cv2.Canny(occ_grid.astype(np.uint8), 50, 150, apertureSize=3)
        skel = cv2.dilate(skel_img, np.ones((2, 2)))
        edges = edges - skel
      ##cv2.imshow("Edges", edges)
      ##cv2.imshow("skel", skel)
        edges[edges < 250] = 0
        edges = edges.astype(np.bool)
        # buh_edges = cv2.subtract(edges, buh_mask)
        # buh_edges[buh_edges < 250] = 0
        # buh_edges = cv2.dilate(buh_edges, np.ones((2, 2)))
        # buh_edges = buh_edges.astype(np.bool)

        occ_grid[occ_grid == 100] = -1
        occ_grid[edges] = 100
        msg = occgrid_to_ros.og_msg(occ_grid,
                                    perspective_transformer.map_resolution,
                                    perspective_transformer.map_size, time_get_frame_ros)
        publisher.publish(msg)
      ##cv2.imshow("occgrid", occ_grid)
      ##cv2.imshow("frame", cv2.resize(frame, (512, 512)))

        print('Inference FPS:  ',  format(inference_fps, '.2f'), ' | ',
              'Total loop FPS:  ', format(1/(time.time()-t0), '.2f'))
    if (cv2.waitKey(1) & 0xFF == ord('q')) | (ret == False):
        # pipeline.stop()
        cv2.destroyAllWindows()
        break
