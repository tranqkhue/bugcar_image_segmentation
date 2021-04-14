import numpy as np
import rospy
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
import cv2
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------


def og_msg(occ_grid, map_resolution, map_width, map_height, time_stamp):
    MAP_RESOLUTION = map_resolution  # Unit: Meter
    MAP_WIDTH = map_width  # Unit: Meter, Shape: Square with center "base_link"
    MAP_HEIGHT = map_height
    map_img = cv2.flip(occ_grid, 0)
    map_img = cv2.rotate(map_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # map_img = cv2.flip(map_img, 1)
    occupancy_grid = map_img.flatten()
    occupancy_grid = occupancy_grid.tolist()

    map_msg = OccupancyGrid()

    map_msg.header = Header()
    map_msg.header.frame_id = "base_link"
    map_msg.header.stamp = time_stamp

    map_msg.info = MapMetaData()
    map_msg.info.height = int(MAP_HEIGHT / MAP_RESOLUTION)  # Unit: Pixel

    map_msg.info.width = int(MAP_WIDTH / MAP_RESOLUTION)  # Unit: Pixel
    map_msg.info.resolution = MAP_RESOLUTION

    map_msg.info.origin = Pose()
    map_msg.info.origin.position = Point()
    map_msg.info.origin.position.x = 0  # Unit: Meter
    map_msg.info.origin.position.y = -MAP_HEIGHT / 2  # Unit: Meter
    map_msg.info.origin.position.z = 0
    map_msg.info.origin.orientation = Quaternion()
    map_msg.info.origin.orientation.x = 0
    map_msg.info.origin.orientation.y = 0
    map_msg.info.origin.orientation.z = 0
    map_msg.info.origin.orientation.w = 1
    map_msg.data.extend(occupancy_grid)
    map_msg.info.map_load_time = rospy.Time.now()
    return map_msg
