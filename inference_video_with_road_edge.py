import logging, os
from utils import find_intersection_line
logging.disable(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import cv2
import rospy
import time
import pyrealsense2 as rs
from calibration import INPUT_SHAPE
from bev import bev_transform_tools
import occgrid_to_ros
from sklearn.mixture import GaussianMixture

MINIMUM_ROAD_WIDTH = 1.8  # unit: metres
logger = logging.getLogger('bruh')
# logging.Formatter(' %(levelname) -  %(message) ')
logger.setLevel(logging.DEBUG)
import matplotlib.pyplot as plt
#================================================================================

#---------------------------------------------------------------------------------
gm = GaussianMixture(n_components=2)


def enet_preprocessing(bgr_frame):
    IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGE_STD = np.array([0.229, 0.224, 0.225])
    input_size = (512, 256)

    resized = cv2.resize(bgr_frame, input_size)

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #rotate = cv2.rotate(rgb, cv2.cv2.ROTATE_180)
    cv2.imshow('input', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # Normalize, some statistics and stack into a batch for interference
    normalized = rgb / 256.0
    subtracted = np.subtract(normalized, IMAGE_MEAN)
    divided = np.divide(subtracted, IMAGE_STD)
    swap_axesed = np.swapaxes(divided, 1, 2)
    swap_axesed = np.swapaxes(swap_axesed, 0, 1)
    batch = np.array([swap_axesed])

    return batch


#---------------------------------------------------------------------------------


def contour_noise_removal(segmap):
    #Close small gaps by a kernel with shape proporition to the image size
    h_segmap, w_segmap = segmap.shape
    min_length = min(h_segmap, w_segmap)
    kernel = np.ones((int(min_length / 50), int(min_length / 50)), np.uint8)
    closed = cv2.morphologyEx(segmap, cv2.MORPH_CLOSE, kernel)

    #Find contours of segmap
    cnts, hie = cv2.findContours(closed, 1, 2)
    cnts = list(filter(lambda cnt: cnt.shape[0] > 2, cnts))

    # Create a rectangular mask in lower part of the frame
    # If a contour intersect with this lower part above a threshold
    # then that contour will be kept as a valid one

    LENGTH_RATIO = 0.1
    x_left = 0
    x_right = w_segmap
    y_top = int(h_segmap * (1 - LENGTH_RATIO))
    y_bot = h_segmap
    bottom_rect = np.array([(x_left,  y_top), (x_right, y_top),\
       (x_right, y_bot), (x_left,  y_bot)])
    bottom_mask = np.zeros_like(segmap)
    mask_area = (x_right - x_left) * (y_bot - y_top)
    cv2.fillPoly(bottom_mask, [bottom_rect], 1)

    # Iterate through contour[S]
    MASK_AREA_THRESH = 0.4  #The threshold of intersect over whole mask area
    main_road_cnts = []
    for cnt in cnts:
        cnt_map = np.zeros_like(segmap)
        cv2.fillPoly(cnt_map, [cnt], 1)
        masked = cv2.bitwise_and(cnt_map, bottom_mask).astype(np.uint8)
        intersected_area = np.count_nonzero(masked)
        logger.debug("Mask_area:  ", mask_area)
        logger.debug("Contour:  ", intersected_area, "  Area:  ",
                     intersected_area)
        if (intersected_area > (mask_area * MASK_AREA_THRESH)):
            main_road_cnts.append(cnt)

    contour_noise_removed = np.zeros(segmap.shape).astype(np.uint8)
    cv2.fillPoly(contour_noise_removed, main_road_cnts, 1)

    return contour_noise_removed


#================================================================================
# Initialize
publisher = occgrid_to_ros.init_node()
wait_time = 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], \
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
#---------------------------------------------------------------------------------

while True:

    ret, frame = cap.read()
    # ret = True
    if (ret == True):
        # Prepocessing input
        t0 = time.time()
        a = cv2.warpPerspective(
            frame, perspective_transformer._bev_matrix,
            (perspective_transformer.width, perspective_transformer.height))
        # cv2.imshow("warped image", a)
        time_get_frame_ros = rospy.Time.now()
        batch_frame = enet_preprocessing(frame)

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
            cv2.imshow("edge segmap", cv2.resize(edges_detection, (512, 256)))
            cv2.imshow("edge", cv2.resize(dilated_edges, (512, 256)))
            cv2.imshow("edges occ grid", edges_occ_grid)
            cv2.imshow("occgrid", occ_grid)
            cv2.imshow("hough out", hough_out)
            plt.show()

            # cv2.imshow('segmap', contour_noise_removed * 200)
            # cv2.imshow('be4 contour', segmap * 200)

    key_pressed = cv2.waitKey(wait_time) & 0xFF
    if key_pressed == ord('q') or ret == False:
        # cap.release()
        cv2.destroyAllWindows()
        break
    elif key_pressed == ord('s'):
        cv2.waitKey(0)
    elif key_pressed == ord('g'):
        wait_time = 150 - wait_time
