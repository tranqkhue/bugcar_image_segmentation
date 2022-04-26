#!/home/tranquockhue/anaconda3/envs/tf2.2/bin/python
#!/home/thang/anaconda3/envs/tf2env/bin/python
from cameraType.realsense import RealsenseCamera
from image_processing_utils import contour_noise_removal
from models import ENET
import occgrid_to_ros
from bev import bev_transform_tools
from nav_msgs.msg import OccupancyGrid
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
    rospy.init_node("image_segmentation", anonymous=True,
                    disable_signals=True)
    rate = rospy.Rate(15)
    # =========== Section for reading ROS param, after initializing the node============================
    node_name, node_namespace = rospy.get_name(), rospy.get_namespace()
    print(node_name, node_namespace)
    params_dict = {"width": 0, "height": 0,
                   "cell_size": 0, "topic_name":"", "pose":[]}
    for param in params_dict:
        try:
            value = rospy.get_param(node_name+"/"+param)
            print("value", value)
            params_dict[param] = value
        except KeyError:  # rospy cannot find the desired parameters
            raise KeyError("you lack a parameter:" + param)
    params_dict[param] = value
    og_width_in_m = params_dict["width"]
    og_height_in_m = params_dict["height"]
    cell_size_in_m = params_dict["cell_size"]
    map_topic = params_dict["topic_name"]
    camera_pose_wrt_to_baselink = params_dict["pose"]
    publisher = rospy.Publisher(
        node_namespace+map_topic, OccupancyGrid, queue_size=5, latch=False)
    # ==================================================================================================

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=800)])
        except RuntimeError as e:
            print(e)
    # have an interface for model here
    model = ENET('./pretrained_models/enet.pb')

# 
    # read from json or yaml in ros?
    perspective_transformer = bev_transform_tools.fromJSON(
        'calibration_data.json')


    

    # ------------------------------------------------------------------------------
    while True:
        t0 = time.time()
        calibrated_input_shape = (perspective_transformer.input_width,perspective_transformer.input_height)
        frameReleaser = RealsenseCamera("69420",calibrated_input_shape,30)
        bgr_frame = frameReleaser.get_bgr_frame()
        if (bgr_frame is not None):
            # Prepocessing input
            # enet preprocessing bottleneck, jump from 30 to 80 % cpu
            time_get_frame_ros = rospy.Time.now()
            batch_frame = model.preprocess(bgr_frame)
            inference_result = model.predict(batch_frame)[0]
            # # rospy doesn't create bottleneck, it seems
            # # Run inference and process the results
            result_by_class = np.argmax(inference_result, axis=0)
            segmap = np.bitwise_or(result_by_class == 0, result_by_class == 1)\
                .astype(np.uint8)
            # Remove road branches (or noise) that are not connected to main branches
            # Main road branches go from the bottom part of the RGB map
            # (should be) right front of the vehicle
            contour_noise_removed = contour_noise_removal(segmap)  # 5% cpu


            # # Need to resize to the original image size in calibration process
            # or else it will throw an error
            resized_segmap = cv2.resize(contour_noise_removed, (perspective_transformer.input_width,perspective_transformer.input_height))
            occ_grid = perspective_transformer.create_occupancy_grid(
                resized_segmap,
                og_width_in_m, og_height_in_m,
                cell_size_in_m)

            msg = occgrid_to_ros.convert_to_occupancy_grid_msg(occ_grid, cell_size_in_m,
                                        og_width_in_m, og_height_in_m,time_get_frame_ros,camera_pose_wrt_to_baselink)
            publisher.publish(msg)
            rate.sleep()


if __name__ == "__main__":
    main()
