import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import tensorflow as tf
import numpy as np
import cv2
import rospy
import time

from bev import bev_transform_tools 
import occgrid_to_ros

#================================================================================

def setupBEV():
	img_shape 		= (512,1024)
	top_left_tile 	= np.array([426,370]) 
	bot_left_tile 	= np.array([406,409])
	top_right_tile 	= np.array([574,371])
	bot_right_tile 	= np.array([592,410])
	tile_vertices   = np.stack((top_left_tile, bot_left_tile, \
								top_right_tile,bot_right_tile), axis = 0)
	dist2target = (0,270)
	tile_length = 60
	perspective_transformer = bev_transform_tools(img_shape, dist2target, \
												  tile_length)
	matrix = perspective_transformer.calculate_transform_matrix(tile_vertices)
	perspective_transformer.create_occ_grid_param()
	
	return perspective_transformer,matrix

#---------------------------------------------------------------------------------

def enet_preprocessing(bgr_frame):
	IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
	IMAGE_STD  = np.array([0.229, 0.224, 0.225])
	input_size = (512,256)

	resized =  cv2.resize(bgr_frame, input_size)
	rgb 	=  cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
	#rotate = cv2.rotate(rgb, cv2.cv2.ROTATE_180)
	cv2.imshow('input', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

	# Normalize, some statistics and stack into a batch for interference
	normalized  = rgb/256.0
	subtracted  = np.subtract(normalized, IMAGE_MEAN)
	divided	    = np.divide(subtracted, IMAGE_STD)
	swap_axesed = np.swapaxes(divided, 1, 2)
	swap_axesed = np.swapaxes(swap_axesed, 0, 1)
	batch 		= np.array([swap_axesed])

	return batch

#---------------------------------------------------------------------------------

def contour_noise_removal(segmap):
	#Close small gaps by a kernel with shape proporition to the image size
	h_segmap, w_segmap = segmap.shape
	min_length 		   = min(h_segmap, w_segmap)
	kernel      = np.ones((int(min_length/50),int(min_length/50)), np.uint8)
	closed      = cv2.morphologyEx(segmap, cv2.MORPH_CLOSE, kernel)

	#Find contours of segmap
	cnts, hie   = cv2.findContours(closed, 1, 2)
	cnts = list(filter(lambda cnt: cnt.shape[0] > 2, cnts))
	
	# Create a rectangular mask in lower part of the frame
	# If a contour intersect with this lower part above a threshold
	# then that contour will be kept as a valid one

	LENGTH_RATIO	= 0.1
	x_left  = 0
	x_right = w_segmap
	y_top 	= int(h_segmap* LENGTH_RATIO)
	y_bot 	= h_segmap
	bottom_rect = np.array([(x_left,  y_top), (x_right, y_top),\
							(x_right, y_bot), (x_left,  y_bot)])
	bottom_mask = np.zeros_like(segmap)
	mask_area 	= (x_right - x_left) * (y_bot - y_top)
	cv2.fillPoly(bottom_mask, [bottom_rect], 1)

	# Iterate through contour[S]
	MASK_AREA_THRESH = 0.1 #The threshold of intersect over whole mask area
	main_road_cnts = []
	for cnt in cnts:
		cnt_map = np.zeros_like(segmap)
		cv2.fillPoly(cnt_map, [cnt], 1)
		masked   = cv2.bitwise_and(cnt_map, bottom_mask).astype(np.uint8)
		intersected_area = np.count_nonzero(masked)
		#print("Mask_area:  ", mask_area)
		#print("Contour:  ", intersected_area, "  Area:  ", intersected_area)
		if (intersected_area > (mask_area*MASK_AREA_THRESH)):
			main_road_cnts.append(cnt)

	contour_noise_removed = np.zeros(segmap.shape).astype(np.uint8)
	cv2.fillPoly(contour_noise_removed, main_road_cnts, 255)

	return contour_noise_removed

#================================================================================
# Initialize
publisher = occgrid_to_ros.init_node()

gpus = tf.config.experimental.list_physical_devices('GPU')
#print('Check available GPUS:  ', gpus)
if gpus:
	try:
		tf.config.experimental.set_memory_growth(gpus[0], True)
		tf.config.experimental.set_virtual_device_configuration(gpus[0], \
			[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=600)])
	except RuntimeError as e:
		print(e)
model = tf.keras.models.load_model('model.hdf5')

perspective_transformer,matrix = setupBEV()
#print("Check model input:  ",model.inputs)
cap = cv2.VideoCapture('test.webm')
cap.set(3, 1280)
cap.set(4, 720)

#---------------------------------------------------------------------------------

while True:
	t0 = time.time()
	ret, frame = cap.read()
	if (ret == True):
		# Prepocessing input
		batch_frame = enet_preprocessing(frame)

		# Run inference and process the results
		t1 = time.time()
		inference_result = model.predict(batch_frame)[0]
		inference_fps = 1/(time.time() - t1)
		result_by_class = np.argmax(inference_result, axis = 0)
		segmap 			= np.bitwise_or(result_by_class==0, result_by_class==1)\
										.astype(np.uint8)

		# Remove road branches (or noise) that are not connected to main branches
		# Main road branches go from the bottom part of the RGB map
		# (should be) right front of the vehicle
		conour_noise_removed = contour_noise_removal(segmap)

		# Visualize the segmap by masking the RGB frame
		(out_height, out_width) = conour_noise_removed.shape
		resized_frame = cv2.resize(frame, (out_width, out_height))
		segmap_viz 	  = cv2.bitwise_and(resized_frame, resized_frame, \
										mask=conour_noise_removed)
		enlarged_viz  = cv2.resize(segmap_viz, (0,0), fx=3, fy=3)
		cv2.imshow('segmap_cnt_noise_removal', cv2.cvtColor(enlarged_viz, \
															cv2.COLOR_RGB2BGR))

		# Visualize the BEV by masking and warp the RGB frame
		# Resize the segmap to scale with the calibration matrix
		resized_segmap_viz	   = cv2.resize(segmap_viz,(1024,512)) 
		warped_perspective_viz = cv2.warpPerspective(resized_segmap_viz, \
													 matrix,(1024,512))
		cv2.imshow('wapred_perspective', cv2.cvtColor(warped_perspective_viz, \
													  cv2.COLOR_RGB2BGR))
		
		# Publish to Occupancy Grid
		# Need to resize to be the same with the image size in calibration process
		resized_segmap = cv2.resize(conour_noise_removed,(1024,512))
		occ_grid 	= perspective_transformer.create_occupancy_grid(resized_segmap)
		msg  		= occgrid_to_ros.og_msg(occ_grid,\
											perspective_transformer.map_resolution,\
											perspective_transformer.map_size)
		publisher.publish(msg)

	print('Inference FPS:  ',  format(inference_fps, '.2f'), ' | ',\
		  'Total loop FPS:  ', format(1/(time.time()-t0), '.2f'))

	if (cv2.waitKey(25) & 0xFF ==ord('q')) | (ret == False):
		cap.release()
		cv2.destroyAllWindows()
		break	