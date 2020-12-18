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
	img_shape = (512,1024)
	top_left_tile = np.array([426,370]) 
	bot_left_tile = np.array([406,409])
	top_right_tile = np.array([574,371])
	bot_right_tile = np.array([592,410])
	tile = np.stack((top_left_tile,bot_left_tile,top_right_tile,bot_right_tile),\
				    axis = 0)
	target = (0,270)
	tile_length = 60
	perspective_transformer = bev_transform_tools(img_shape,target,tile_length)
	matrix = perspective_transformer.calculate_transform_matrix(tile)
	perspective_transformer.create_occ_grid_param()
	return perspective_transformer,matrix

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
cap = cv2.VideoCapture(0)
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
		output = model.predict(img)[0]
		print('Inference FPS:  ', 1/(time.time() - t0))
		output = np.argmax(output, axis = 0)
		mask_viz = np.bitwise_or(output==0, output==1).astype(np.uint8)

		# Visualize the segmap
		(out_height, out_width) = mask_viz.shape
		resized  	 = cv2.resize(frame, (out_width, out_height))
		output_viz 	 = cv2.bitwise_and(resized, resized, mask=mask_viz)
		output = cv2.resize(output_viz, (0,0), fx=3, fy=3)
		cv2.imshow('segmap',cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

		# Visualize the BEV
		viz_resized  = cv2.resize(output_viz,(1024,512))
		warped_image = cv2.warpPerspective(viz_resized, matrix,(1024,512))
		cv2.imshow('wapred_perspective', cv2.cvtColor(warped_image, \
													  cv2.COLOR_RGB2BGR))
		
		#Publish to Occupancy Grid
		mask_viz = cv2.resize(mask_viz,(1024,512))
		occ_grid = perspective_transformer.create_occupancy_grid(mask_viz)
		publisher.publish(occgrid_to_ros.og_msg(occ_grid,\
						  perspective_transformer.map_resolution,\
				 		  perspective_transformer.map_size))

	if cv2.waitKey(25) & 0xFF ==ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break	