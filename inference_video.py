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

#---------------------------------------------------------------------------------

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

#def contour_noise_removal(segmap):
#	h_segmap, w_segmap = 

#---------------------------------------------------------------------------------
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

IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD  = np.array([0.229, 0.224, 0.225])
input_size = (256,128)

while True:
	ret, frame = cap.read()
	if (ret == True):
		# Prepocessing input
		frame = cv2.resize(frame,input_size)
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		#frame = cv2.rotate(frame, cv2.cv2.ROTATE_180)
		cv2.imshow('input',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

		# Normalize and stack into a batch for interference
		img = frame/256.0
		img = np.subtract(img,IMAGE_MEAN)
		img = np.divide(img,IMAGE_STD)
		img = np.swapaxes(img, 1, 2)
		img = np.swapaxes(img, 0, 1)
		img = np.array([img])

		# Run inference and process the results
		t0 = time.time()
		inference_result = model.predict(img)[0]
		print('Inference FPS:  ', 1/(time.time() - t0))
		result_by_class = np.argmax(inference_result, axis = 0)
		segmap 			= np.bitwise_or(result_by_class==0, result_by_class==1)\
										.astype(np.uint8)

		# Visualize the segmap by masking the RGB frame
		(out_height, out_width) = segmap.shape
		resized_frame = cv2.resize(frame, (out_width, out_height))
		segmap_viz 	  = cv2.bitwise_and(resized_frame, resized_frame, \
									    mask=segmap)
		enlarged 	  = cv2.resize(segmap_viz, (0,0), fx=3, fy=3)
		cv2.imshow('segmap',cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR))

		# Visualize the BEV by masking and warp the RGB frame
		# Resize the segmap to scale with the calibration matrix
		resized_segmap_viz	   = cv2.resize(segmap_viz,(1024,512)) 
		warped_perspective_viz = cv2.warpPerspective(resized_segmap_viz, \
													 matrix,(1024,512))
		cv2.imshow('wapred_perspective', cv2.cvtColor(warped_perspective_viz, \
													  cv2.COLOR_RGB2BGR))
		
		# Publish to Occupancy Grid
		resized_segmap = cv2.resize(segmap,(1024,512))
		occ_grid = perspective_transformer.create_occupancy_grid(resized_segmap)
		publisher.publish(occgrid_to_ros.og_msg(occ_grid,\
						  perspective_transformer.map_resolution,\
				 		  perspective_transformer.map_size))

	if cv2.waitKey(25) & 0xFF ==ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break	