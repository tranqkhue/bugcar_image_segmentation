#!/home/tranquockhue/anaconda3/envs/tf2.2/bin/python
from utils import contour_noise_removal, enet_preprocessing
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
# def depth_frame_2_occ_grid(depth_frame):
#     points = pc.calculate(depth_frame)
#     v = points.get_vertices()
#     # xyz # vertices are in metres unit
# 	global vtx
#     vtx = np.asanyarray(v).view(np.float32).reshape(-1, 3)
#     # y_only is a (2, n) dimesion array, not (h,w) dimension array
# ------------------------------------------------------------------------------


# ================================================================================
# Initialize

# ---------------------------------------------------------------------------------
def main():
    global INPUT_SHAPE
    publisher = occgrid_to_ros.init_node(disable_signals=True)
    rate = rospy.Rate(15)
    # =========== Section for reading ROS param, after initializing the node============================
    node_name = rospy.get_name()
    params_dict = {"width": 0, "height": 0, "cell_size": 0, "serial_no": ""}
    for param in params_dict:
        try:
            value = rospy.get_param(node_name+"/"+param)
            print(value)
        except KeyError:  # rospy cannot find the desired parameters
            raise KeyError("you lack a parameter: " + param)
    params_dict[param] = value
    og_width = rospy.get_param(params_dict["width"])
    og_height = rospy.get_param(params_dict["height"])
    cell_size = rospy.get_param(params_dict["cell_size"])
    camera_serial_num = rospy.get_param(params_dict["serial_no"])

    # ==================================================================================================

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=800)])
        except RuntimeError as e:
            print(e)
    model = ENET('./pretrained_models/enet.pb')
    perspective_transformer = bev_transform_tools.fromJSON(
        'calibration_data_30fps.json')
    INPUT_SHAPE = INPUT_SHAPE["30fps"]
    width = INPUT_SHAPE[0]
    height = INPUT_SHAPE[1]

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(camera_serial_num)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)
    # for debugging with video:
    # cap = cv2.VideoCapture("img/test1.webm")

    # pc = rs.pointcloud()
    # ------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------
    # Realsense-based post-processing
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 3)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    # temporal = rs.temporal_filter()
    # ------------------------------------------------------------------------------
    while True:
        t0 = time.time()
        ret = True
        pipeline_frames = pipeline.wait_for_frames()
        pipeline_rgb_frame = pipeline_frames.get_color_frame()
        rgb_frame = np.asanyarray(pipeline_rgb_frame.get_data())

        # ret, rgb_frame = cap.read()
        if (ret == True):
            # Prepocessing input
            # enet preprocessing bottleneck, jump from 30 to 80 % cpu
            batch_frame = enet_preprocessing(rgb_frame)
            time_get_frame_ros = rospy.Time.now()
            # # rospy doesn't create bottleneck, it seems
            # # Run inference and process the results
            inference_result = model.predict(batch_frame)[0]
            result_by_class = np.argmax(inference_result, axis=0)
            segmap = np.bitwise_or(result_by_class == 0, result_by_class == 1)\
                .astype(np.uint8)
            # Remove road branches (or noise) that are not connected to main branches
            # Main road branches go from the bottom part of the RGB map
            # (should be) right front of the vehicle
            contour_noise_removed = contour_noise_removal(segmap)  # 5% cpu

            #contour_noise_removed = segmap
            # # Need to resize to be the same with the image size in calibration process

            resized_segmap = cv2.resize(contour_noise_removed, INPUT_SHAPE)
            occ_grid = perspective_transformer.create_occupancy_grid(
                resized_segmap, perspective_transformer._bev_matrix,
                perspective_transformer.width, perspective_transformer.height,
                og_width, og_height,
                cell_size,
                perspective_transformer.cm_per_px)
            msg = occgrid_to_ros.og_msg(occ_grid, cell_size,
                                        og_width, og_height,
                                        time_get_frame_ros)
            publisher.publish(msg)
            rate.sleep()
        if (ret == False):
            break
    pipeline.stop()


if __name__ == "__main__":
    main()
