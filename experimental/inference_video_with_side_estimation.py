import occgrid_to_ros
from bev import bev_transform_tools
from calibration import FPS, INPUT_SHAPE, WARPED_IMG_SHAPE
import pyrealsense2 as rs
import time
import rospy
import cv2
import numpy as np
import tensorflow as tf
import logging
import os
from utils import enet_preprocessing, find_intersection_line
logging.disable(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger('bruh')
# logging.Formatter(' %(levelname) -  %(message) ')
logger.setLevel(logging.CRITICAL)
print(logger.getEffectiveLevel(), logger.isEnabledFor(logging.INFO))
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
# ================================================================================

# ---------------------------------------------------------------------------------


# ================================================================================

model = tf.keras.models.load_model('model.hdf5')
perspective_transformer = bev_transform_tools.fromJSON('calibration_data.json')
matrix = perspective_transformer._bev_matrix
#print("Check model input:  ",model.inputs)
cap = cv2.VideoCapture(0)

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
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, INPUT_SHAPE[0], INPUT_SHAPE[1],
#  rs.format.rgb8, FPS)
# pipeline.start(config)
while True:
    # pipeline_frames = pipeline.wait_for_frames()
    # pipeline_rgb_frame = pipeline_frames.get_color_frame()
    # frame = np.asanyarray(pipeline_rgb_frame.get_data())
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ret, frame = cap.read()
    # ret = True
    if (ret == True):
        # Prepocessing input
        t0 = time.time()
        a = cv2.warpPerspective(
            frame, perspective_transformer._bev_matrix,
            (perspective_transformer.width, perspective_transformer.height))
      ##cv2.imshow("warped image", a)
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

        # Publish to Occupancy Grid
        # Need to resize to be the same with the image size in calibration process
        resized_segmap = cv2.resize(contour_noise_removed, INPUT_SHAPE)
        occ_grid = perspective_transformer.create_occupancy_grid(
            resized_segmap, perspective_transformer._bev_matrix,
            perspective_transformer.width, perspective_transformer.height,
            perspective_transformer.map_size,
            perspective_transformer.map_resolution,
            perspective_transformer.cm_per_px)
        t_after_occ_grid = time.time()
        out_copy = np.copy(occ_grid)
        out_copy[out_copy == -1] = 100
        out_copy = np.uint8(out_copy)
        edges = cv2.Canny(out_copy, 50, 150, apertureSize=3)
        thresh, edges_out = cv2.threshold(cv2.subtract(edges, edges_occ_grid),
                                          230, 255, cv2.THRESH_BINARY)
        hough_out = cv2.cvtColor(edges_out, cv2.COLOR_GRAY2BGR)
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges_out = cv2.dilate(edges_out, kernel, iterations=1)
        lines = []
        try:
            lines = cv2.HoughLines(dilated_edges_out, 1, np.pi / 180, 40)
            for rho, theta in lines[0]:
                x = np.cos(theta)
                y = np.sin(theta)
                x0 = x * rho
                y0 = y * rho
                x1 = int(x0 + 1000 * (-y))
                y1 = int(y0 + 1000 * x)
                x2 = int(x0 - 1000 * (-y))
                y2 = int(y0 - 1000 * x)

                cv2.line(hough_out, (x1, y1), (x2, y2), (0, 0, 255), 1)
        except TypeError:
            pass

        # code to find the area to fill the base
        # what if there is no intersection?
        intersection = np.bitwise_and(dilated_edges_out, edges_occ_grid)
        t_intersection = time.time()
        intersection_pts = np.array(np.where(intersection >= 240)).T
        if (intersection_pts.shape[0] > 0):
            edge_grid_pts = np.array(np.where(edges_occ_grid >= 240)).T
            min_x_intersection_index = np.argmin(intersection_pts[:, 1])
            max_x_intersection_index = np.argmax(intersection_pts[:, 1])
            min_y_of_min_x_intersection, min_x_intersection = intersection_pts[
                min_x_intersection_index]
            min_y_of_max_x_intersection, max_x_intersection = intersection_pts[
                max_x_intersection_index]
            ls_intersection_pts = edge_grid_pts
            ls_intersection_pts[:, [0, 1]] = ls_intersection_pts[:, [1, 0]]
            x = ls_intersection_pts[:, 0]
            # print("x:", x)
            y = ls_intersection_pts[:, 1]
            cond1s = np.where(x < min_x_intersection, 1, 0)
            cond1 = np.where(x == min_x_intersection, 1, 0)
            # print(ls_intersection_pts[cond1.astype(np.bool).reshape(-1)])
            cond2s = np.where(x > max_x_intersection, 1, 0)
            cond2 = np.where(x == max_x_intersection, 1, 0)
            # print(ls_intersection_pts[cond2.astype(np.bool).reshape(-1)])
            cond3 = np.where(y <= min_y_of_min_x_intersection, 1, 0)
            # print(ls_intersection_pts[cond3.astype(np.bool).reshape(-1)])
            cond4 = np.where(y <= min_y_of_max_x_intersection, 1, 0)
            # print(ls_intersection_pts[cond4.astype(np.bool).reshape(-1)])
            cond1_and_3 = cv2.bitwise_and(cond1, cond3)
            cond2_and_4 = cv2.bitwise_and(cond2, cond4)
            cond_total = cv2.bitwise_or(cond1_and_3, cond2_and_4)
            cond_total = cv2.bitwise_or(cond_total, cond1s)
            cond_total = cv2.bitwise_or(cond_total, cond2s)
            lmao = ls_intersection_pts[cond_total.astype(np.bool).reshape(-1)]
            reversed_cond_total = np.where(cond_total == 1, 0,
                                           1).astype(np.bool).reshape(-1)
            ls_intersection_pts = ls_intersection_pts[reversed_cond_total]
            x = ls_intersection_pts[:, 0]
            y = -ls_intersection_pts[:, 1]
            # lexsort sorts in increasing order.
            # and the priority of sorting is from rightmost column to leftmost column.
            index = np.lexsort((y, x))
            ls_intersection_pts = ls_intersection_pts[index].tolist()
            # a = np.zeros_like(edges_occ_grid)
            # for pt in lmao:
            #     x = pt[0]
            #     y = pt[1]
            #     a[y, x] = 255
            # #cv2.imshow("a", a)

            # print("intersection of cond :", ls_intersection_pts)
        else:
            ls_intersection_pts = []
        t_after_intersection = time.time()
        # snippet to find parallel curbs
        intersection_out = np.copy(intersection)
        intersection_out = cv2.cvtColor(intersection_out, cv2.COLOR_GRAY2BGR)
        intersections_bottom = []
        rhos = []
        try:
            rho_main_line, theta = lines[0][0]
            for pts in [ls_intersection_pts[0], ls_intersection_pts[-1]]:
                logger.info("endpoint", pts)
                length = np.sqrt((pts[1])**2 + pts[0]**2)
                angle = theta - np.arctan2(pts[1], pts[0])
                rho = length * np.cos(angle)
                # print(rho, rho_main_line)
                if (abs(rho - rho_main_line) > 20):
                    rhos.append(rho)
                else:
                    try:
                        rhos.index(rho_main_line)
                    except ValueError:
                        rhos.append(rho_main_line)
            logger.debug('rhos is', rhos)
            for rho in rhos:
                x = np.cos(theta)
                y = np.sin(theta)
                x0 = x * rho
                y0 = y * rho
                x1 = int(x0 + 1000 * (-y))
                y1 = int(y0 + 1000 * x)
                x2 = int(x0 - 1000 * (-y))
                y2 = int(y0 - 1000 * x)
                line = np.array([[x1, y1], [x2, y2]])
                # print("lines is", line)
                intersection_bottom = find_intersection_line(
                    line, bottom_edge_occ_grid)
                # print("intersection_bottom", intersection_bottom)
                if intersection_bottom is None:
                    intersections_bottom = bottom_edge_occ_grid.tolist()
                else:
                    if len(rhos) == 1:
                        if theta < np.pi / 2:
                            intersections_bottom.append(
                                bottom_edge_occ_grid[1])
                            intersections_bottom.append(intersection_bottom)

                        else:
                            intersections_bottom.append(
                                bottom_edge_occ_grid[0])
                            intersections_bottom.append(intersection_bottom)

                    else:
                        intersections_bottom.append(intersection_bottom)
                cv2.line(intersection_out, (x1, y1), (x2, y2), (0, 0, 255), 1)
            t_after_find_lines = time.time()
            if len(ls_intersection_pts) != 0:
                ls_intersection_pts.append(intersections_bottom[1])
                ls_intersection_pts.append(intersections_bottom[0])
                ls_intersection_pts = np.asarray(ls_intersection_pts).astype(
                    np.int)
                logger.debug("after", ls_intersection_pts)
                cv2.fillPoly(occ_grid, [ls_intersection_pts], (0))
        except (TypeError, ValueError, IndexError) as e:
            logger.error(e)
        finally:
            time_fill_poly = time.time()
            msg = occgrid_to_ros.og_msg(occ_grid,
                                        perspective_transformer.map_resolution,
                                        perspective_transformer.map_size,
                                        time_get_frame_ros)
            time_publish = time.time()

            publisher.publish(msg)
          ##cv2.imshow("edges occ grid", edges_occ_grid)
          ##cv2.imshow("occgrid", occ_grid)
          ##cv2.imshow("edge", dilated_edges_out)
          ##cv2.imshow('hough', hough_out)
          ##cv2.imshow('intersection_out', intersection_out)
          ##cv2.imshow('segmap', contour_noise_removed * 200)
          ##cv2.imshow('be4 contour', segmap * 200)

    # print('Inference FPS:  ',  format(inference_fps, '.2f'), ' | ',\
    #    'Total loop FPS:  ', format(1/(time.time()-t0), '.2f'))

    # print("time for inference", 1 / inference_fps)
    # print("time contour+occgrid", t_after_occ_grid - t_before_contour)
    # print("time hough line", t_intersection - t_after_occ_grid)
    # print("time for finding intersection",
    #       t_after_intersection - t_intersection)
    # print("time for finding line", t_after_find_lines - t_after_intersection)
    # print("time for filling poly", time_fill_poly - t_after_find_lines)
    # print("time convert", time_publish - time_fill_poly)
    key_pressed = cv2.waitKey(wait_time) & 0xFF
    if key_pressed == ord('q') or ret == False:
        # cap.release()
        cv2.destroyAllWindows()
        break
    elif key_pressed == ord('s'):
        cv2.waitKey(0)
    elif key_pressed == ord('g'):
        wait_time = 150 - wait_time
