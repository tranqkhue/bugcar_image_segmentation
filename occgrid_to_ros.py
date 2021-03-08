import numpy as np

import rospy
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
import cv2
#---------------------------------------------------------------------------------


def init_node():
    rospy.init_node("image_segmentation", anonymous=True)
    map_topic = "map/image_segmentation"
    OG_publisher = rospy.Publisher(map_topic,
                                   OccupancyGrid,
                                   queue_size=5,
                                   latch=False)
    return OG_publisher


#---------------------------------------------------------------------------------


def og_msg(occ_grid, map_resolution, map_size, time_stamp):
    MAP_RESOLUTION = map_resolution  #Unit: Meter
    MAP_SIZE = map_size  #Unit: Meter, Shape: Square with center "base_link"

    map_img = cv2.flip(occ_grid, 0)
    occupancy_grid = map_img.flatten()
    occupancy_grid = occupancy_grid.tolist()

    map_msg = OccupancyGrid()

    map_msg.header = Header()
    map_msg.header.frame_id = "base_link"
    map_msg.header.stamp = time_stamp

    map_msg.info = MapMetaData()
    map_msg.info.height = int(MAP_SIZE / MAP_RESOLUTION)  #Unit: Pixel

    map_msg.info.width = int(MAP_SIZE / MAP_RESOLUTION)  #Unit: Pixel
    map_msg.info.resolution = MAP_RESOLUTION

    map_msg.info.origin = Pose()
    map_msg.info.origin.position = Point()
    map_msg.info.origin.position.x = -MAP_SIZE / 2  #Unit: Meter
    map_msg.info.origin.position.y = 0  #Unit: Meter
    map_msg.info.origin.position.z = 0
    map_msg.info.origin.orientation = Quaternion()
    map_msg.info.origin.orientation.x = 0
    map_msg.info.origin.orientation.y = 0
    map_msg.info.origin.orientation.z = 0
    map_msg.info.origin.orientation.w = 1
    map_msg.data.extend(occupancy_grid)
    map_msg.info.map_load_time = rospy.Time.now()
    return map_msg
