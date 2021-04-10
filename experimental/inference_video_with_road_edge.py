import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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
from utils import contour_noise_removal, enet_preprocessing, find_intersection_line
logging.disable(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MINIMUM_ROAD_WIDTH = 1.8  # unit: metres
logger = logging.getLogger('bruh')
logger.setLevel(logging.DEBUG)
# ================================================================================

# ---------------------------------------------------------------------------------
gm = GaussianMixture(n_components=2)


# ================================================================================
# Initialize
publisher = occgrid_to_ros.init_node()
wait_time = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=200)])
    except RuntimeError as e:
        logger.error(e)
model = tf.keras.models.load_model('model.h5')
perspective_transformer = bev_transform_tools.fromJSON('calibration_data.json')
matrix = perspective_transformer._bev_matrix
#print("Check model input:  ",model.inputs)
cap = cv2.VideoCapture("test02.webm")

free_occ_grid = np.ones(shape=(INPUT_SHAPE[1], INPUT_SHAPE[0]))
free_occ_grid = perspective_transformer.create_occupancy_grid(
    free_occ_grid, perspective_transformer._bev_matrix,
    perspective_transformer.width, perspective_transformer.height,
    perspective_transformer.map_size, perspective_transformer.map_resolution,
    perspective_transformer.cm_per_px)

edges_occ_grid = cv2.Canny(free_occ_grid.astype(np.uint8),
                           50,
                           150,
                           apertureSize=3)
width_grid, height_grid = free_occ_grid.shape
# print(width_grid, height_grid)
bottom_edge_occ_grid = np.array([[0, height_grid], [width_grid, height_grid]])
# ---------------------------------------------------------------------------------

while True:

    ret, frame = cap.read()
    # ret = True
    if (ret == True):
        # Prepocessing input
        t0 = time.time()
        a = cv2.warpPerspective(
            frame, perspective_transformer._bev_matrix,
            (perspective_transformer.width, perspective_transformer.height))
        # #cv2.imshow("warped image", a)
        time_get_frame_ros = rospy.Time.now()
        batch_frame = enet_preprocessing(frame)

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
        t_before_contour = time.time()
        contour_noise_removed = contour_noise_removal(segmap)

        # Need to resize to be the same with the image size in calibration process
        resized_segmap = cv2.resize(contour_noise_removed, INPUT_SHAPE)
        occ_grid = perspective_transformer.create_occupancy_grid(
            resized_segmap, perspective_transformer._bev_matrix,
            perspective_transformer.width, perspective_transformer.height,
            perspective_transformer.map_size,
            perspective_transformer.map_resolution,
            perspective_transformer.cm_per_px)
        t_after_occ_grid = time.time()
        try:
            edges_detection = cv2.Canny(np.copy(contour_noise_removed) * 255,
                                        50,
                                        150,
                                        apertureSize=3)
            kernel = np.ones((3, 3))
            hough_out = cv2.cvtColor(edges_detection, cv2.COLOR_GRAY2BGR)
            dilated_edges = cv2.dilate(edges_detection,
                                       kernel=kernel,
                                       iterations=1)
            # edge_detection_occgrid = np.copy(occ_grid)
            # edge_detection_occgrid[edge_detection_occgrid == -1] = 50
            # edge_detection_occgrid = np.uint8(edge_detection_occgrid)
            # edge_detection_occgrid = cv2.Canny(edge_detection_occgrid,
            #                                    50,
            #                                    150,
            #                                    apertureSize=3)
            # thresh, edge_detection_occgrid = cv2.threshold(
            #     cv2.subtract(edge_detection_occgrid, edges_occ_grid), 230, 255,
            #     cv2.THRESH_BINARY)
            # kernel = np.ones((3, 3))
            # hough_out = cv2.cvtColor(edge_detection_occgrid,
            #                          cv2.COLOR_GRAY2BGR)
            # dilated_edges = cv2.dilate(edge_detection_occgrid,
            #                            kernel=kernel,
            #                            iterations=1)
            lines = cv2.HoughLines(dilated_edges, 1, np.pi / 180, 80)
            num_lines = len(lines)
            print("number of lines ", num_lines)

            normalized_lines = np.copy(lines[:, 0, :])
            normalized_lines[:, 0] /= np.linalg.norm(dilated_edges.shape)
            normalized_lines[:, 1] /= np.pi
            plt.xlabel("rho")
            plt.ylabel("theta")
            for i in range(num_lines):
                plt.scatter(normalized_lines[i, 0],
                            normalized_lines[i, 1],
                            c=(1 - i / num_lines, 0, 0 + i / num_lines, 1.0))
            i = 0
            for line in lines:
                for rho, theta in line:
                    x = np.cos(theta)
                    y = np.sin(theta)
                    x0 = x * rho
                    y0 = y * rho
                    x1 = int(x0 + 1000 * (-y))
                    y1 = int(y0 + 1000 * x)
                    x2 = int(x0 - 1000 * (-y))
                    y2 = int(y0 - 1000 * x)
                    cv2.line(hough_out, (x1, y1), (x2, y2),
                             (0 + i * 20, 0, 255 - i * 20), 1)
                    i += 1
        except TypeError as e:
            logger.error(e)
        finally:
            time_fill_poly = time.time()
            msg = occgrid_to_ros.og_msg(occ_grid,
                                        perspective_transformer.map_resolution,
                                        perspective_transformer.map_size,
                                        time_get_frame_ros)
            time_publish = time.time()

            publisher.publish(msg)
          ##cv2.imshow("edge segmap", cv2.resize(edges_detection, (512, 256)))
          ##cv2.imshow("edge", cv2.resize(dilated_edges, (512, 256)))
          ##cv2.imshow("edges occ grid", edges_occ_grid)
          ##cv2.imshow("occgrid", occ_grid)
          ##cv2.imshow("hough out", hough_out)
            plt.show()

            # #cv2.imshow('segmap', contour_noise_removed * 200)
            # #cv2.imshow('be4 contour', segmap * 200)

    key_pressed = cv2.waitKey(wait_time) & 0xFF
    if key_pressed == ord('q') or ret == False:
        # cap.release()
        cv2.destroyAllWindows()
        break
    elif key_pressed == ord('s'):
        cv2.waitKey(0)
    elif key_pressed == ord('g'):
        wait_time = 150 - wait_time
