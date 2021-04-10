from bev import bev_transform_tools
from calibration import INPUT_SHAPE
import time
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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=800)])
    except RuntimeError as e:
        print(e)
model = tf.keras.models.load_model('model.hdf5')
perspective_transformer = bev_transform_tools.fromJSON('calibration_data.json')
matrix = perspective_transformer._bev_matrix
#print("Check model input:  ",model.inputs)
cap = cv2.VideoCapture('test1.webm')
#cap.set(3, 1280)
#cap.set(4, 720)

# ---------------------------------------------------------------------------------
colormap = np.zeros((19), dtype=np.uint8)
# Road
colormap[0] = 0
colormap[1] = 0
# Statics
colormap[2] = 0
colormap[3] = 0
colormap[4] = 0
colormap[5] = 0
colormap[6] = 0
colormap[7] = 0
colormap[8] = 0
colormap[9] = 0
# Sky
colormap[10] = 126
# Dynamics
colormap[11] = 0
colormap[12] = 0
colormap[13] = 0
colormap[14] = 0
colormap[15] = 0
colormap[16] = 0
colormap[17] = 0
colormap[18] = 0

dynamics = np.zeros((19), dtype=np.uint8)
dynamics[11] = 127
dynamics[12] = 127
dynamics[13] = 127
dynamics[14] = 127
dynamics[15] = 127
dynamics[16] = 127
dynamics[17] = 127
dynamics[18] = 127

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if (ret == True):
        # Prepocessing input
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
        contour_noise_removed = contour_noise_removal(segmap)
        # Visualize the segmap by masking the RGB frame
        (out_height, out_width) = contour_noise_removed.shape
        resized_frame = cv2.resize(frame, (out_width, out_height))
        segmap_viz = cv2.bitwise_and(resized_frame, resized_frame,
                                     mask=contour_noise_removed)
        enlarged_viz = cv2.resize(segmap_viz, (0, 0), fx=3, fy=3)

        colored = colormap[result_by_class]
        colored = colored + contour_noise_removed * 255

        dynamics_seg = dynamics[result_by_class]
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(dynamics_seg, kernel, iterations=1)
        colored = colored + dilation
        edges = cv2.Canny(colored, 255 * 3.5, 255 * 4.5)

      ##cv2.imshow('colored', colored)
      ##cv2.imshow('edges', edges)
      ##cv2.imshow('segmap_cnt_noise_removal', enlarged_viz)

        # Visualize the BEV by masking and warp the RGB frame
        # Resize the segmap to scale with the calibration matrix
        resized_segmap_viz = cv2.resize(segmap_viz, (1280, 720))
        warped_perspective_viz = cv2.warpPerspective(cv2.resize(edges, (1280, 720)),
                                                     matrix, (768, 2048))
        _, out = cv2.threshold(warped_perspective_viz, 220, 255,
                               cv2.THRESH_BINARY)
      ##cv2.imshow('warped_perspective', cv2.resize(out, (0, 0),
                                                    fx=0.3,
                                                    fy=0.3))

    if (cv2.waitKey(1) & 0xFF == ord('q')) | (ret == False):
        cap.release()
        cv2.destroyAllWindows()
        break
