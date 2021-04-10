from utils import enet_preprocessing
import occgrid_to_ros
from bev import bev_transform_tools
from calibration import INPUT_SHAPE
import time
import rospy
import cv2
import numpy as np
import tensorflow as tf
import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
model = tf.keras.models.load_model('model.hdf5')
perspective_transformer = bev_transform_tools.fromJSON('calibration_data.json')
matrix = perspective_transformer._bev_matrix
#print("Check model input:  ",model.inputs)
cap = cv2.VideoCapture('test1.webm')

unknown_profile = cv2.imread('edge.png')
unknown_profile = cv2.cvtColor(unknown_profile, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------------------------------
t0 = time.time()
ret, frame = cap.read()
if (ret == True):
    # Prepocessing input
    batch_frame = enet_preprocessing(frame)
while True:
    # Run inference and process the results
    t1 = time.time()
    inference_result = model.predict(batch_frame)[0]
    inference_fps = 1 / (time.time() - t1)
    # result_by_class = np.argmax(inference_result, axis=0)
    # segmap    = np.bitwise_or(result_by_class==0, result_by_class==1)\
    # 			.astype(np.uint8)

    # # Remove road branches (or noise) that are not connected to main branches
    # # Main road branches go from the bottom part of the RGB map
    # # (should be) right front of the vehicle
    # contour_noise_removed = contour_noise_removal(segmap)
    # # Visualize the segmap by masking the RGB frame
    # # (out_height, out_width) = contour_noise_removed.shape
    # # resized_frame = cv2.resize(frame, (out_width, out_height))
    # # segmap_viz    = cv2.bitwise_and(resized_frame, resized_frame, \
    # #         mask=contour_noise_removed)
    # # enlarged_viz = cv2.resize(segmap_viz, (0, 0), fx=3, fy=3)
    # # #cv2.imshow('segmap_cnt_noise_removal', cv2.cvtColor(enlarged_viz, \
    # #              cv2.COLOR_RGB2BGR))

    # # Visualize the BEV by masking and warp the RGB frame
    # # Resize the segmap to scale with the calibration matrix
    # # resized_segmap_viz = cv2.resize(segmap_viz, (1024, 512))
    # # warped_perspective_viz = cv2.warpPerspective(resized_segmap_viz, \
    # #             matrix,(1024,512))
    # # #cv2.imshow('warped_perspective', cv2.cvtColor(warped_perspective_viz, \
    # #              cv2.COLOR_RGB2BGR))

    # # Publish to Occupancy Grid
    # # Need to resize to be the same with the image size in calibration process
    # # print(np.histogram(contour_noise_removed))
    # resized_segmap = cv2.resize(contour_noise_removed, INPUT_SHAPE)
    # occ_grid = perspective_transformer.create_occupancy_grid(
    # 	resized_segmap, perspective_transformer._bev_matrix,
    # 	perspective_transformer.width, perspective_transformer.height,
    # 	perspective_transformer.map_size,
    # 	perspective_transformer.map_resolution,
    # 	perspective_transformer.cm_per_px)
    # msg    = occgrid_to_ros.og_msg(occ_grid,\
    # 		 perspective_transformer.map_resolution,\
    # 		 perspective_transformer.map_size)
    # publisher.publish(msg)

    # occ_grid_cropped = occ_grid[int(occ_grid.shape[0]/4): int(occ_grid.shape[0]*3/4),
    # 							0                       : int(occ_grid.shape[1]/2)]
    # unknown_profile_cropped  = unknown_profile[int(unknown_profile.shape[0]/4):int(unknown_profile.shape[0]*3/4),
    # 								           0                              :int(unknown_profile.shape[1]/2)]
    # out_copy = np.copy(occ_grid_cropped)
    # out_copy = cv2.flip(out_copy, 0)
    # out_copy[out_copy == -1] = 100
    # out_copy = np.uint8(out_copy)
    # edges = cv2.Canny(out_copy, 50,150,apertureSize = 3)
    # edges_out = cv2.subtract(edges, unknown_profile_cropped)
    # hough_out = cv2.cvtColor(edges_out, cv2.COLOR_GRAY2BGR)

    # try:
    # 	lines = cv2.HoughLines(edges_out,1,np.pi/180, 10)
    # 	for rho,theta in lines[0]:
    # 		a = np.cos(theta)
    # 		b = np.sin(theta)
    # 		x0 = a*rho
    # 		y0 = b*rho
    # 		x1 = int(x0 + 1000*(-b))
    # 		y1 = int(y0 + 1000*(a))
    # 		x2 = int(x0 - 1000*(-b))
    # 		y2 = int(y0 - 1000*(a))

    # 		cv2.line(hough_out,(x1,y1),(x2,y2),(0,0,255),2)
    # except TypeError:
    # 	pass

    # kernel = np.ones((5,5),np.uint8)
    # dilated_edges_out = cv2.dilate(edges_out,kernel,iterations = 1)
    # intersection      = np.bitwise_and(dilated_edges_out, unknown_profile_cropped)
    # intersection_pts  = np.array(np.where(intersection == 255))
    # intersection_pts  = np.swapaxes(intersection_pts, 0, 1)

    # intersection_out = np.copy(intersection)
    # intersection_out = cv2.cvtColor(intersection_out, cv2.COLOR_GRAY2BGR)

    # try:
    # 	lines = cv2.HoughLines(edges_out,1,np.pi/180, 10)
    # 	for _, theta in lines[0]:
    # 		for pts in intersection_pts:
    # 			rho = pts[1]*np.cos(theta) + pts[0]*np.sin(theta)
    # 			#print(rho)
    # 			#print(theta)
    # 			a = np.cos(theta)
    # 			b = np.sin(theta)
    # 			x0 = a*rho
    # 			y0 = b*rho
    # 			x1 = int(x0 + 1000*(-b))
    # 			y1 = int(y0 + 1000*(a))
    # 			x2 = int(x0 - 1000*(-b))
    # 			y2 = int(y0 - 1000*(a))

    # 			cv2.line(intersection_out,(x1,y1),(x2,y2),(0,0,255),2)
    # except:
    # 	pass

    # #cv2.imshow('intersection', intersection)
    # #cv2.imshow('hough', hough_out)
    # #cv2.imshow('edges', edges_out)
    # #cv2.imshow('intersection_out', intersection_out)

    print('Inference FPS:  ',  format(inference_fps, '.2f'), ' | ',
          'Total loop FPS:  ', format(1/(time.time()-t0), '.2f'))

    # if (cv2.waitKey(1) & 0xFF == ord('q')) | (ret == False):
    # 	cap.release()
    # 	cv2.destroyAllWindows()
    # 	break
