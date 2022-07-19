import numpy as np
import rospy
from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
import cv2
from scipy.spatial.transform import Rotation as R


def convert_to_occupancy_grid_msg(occ_grid, map_resolution, map_width, map_height, time_stamp,frame_id,pose):
    MAP_RESOLUTION = map_resolution  # Unit: Meter
    MAP_WIDTH = map_width  # Unit: Meter, Shape: Rectangle with center "base_link"
    MAP_HEIGHT = map_height
    map_img = occ_grid
    map_img = cv2.flip(occ_grid, 0)
    # cv2.imshow("map",map_img)
    # we dont need rotation here?, just specify the Pose, if i understand correctly, in occupancy grid the x points upward and y goes right
    map_img = cv2.rotate(map_img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    # make a 90 deg counter clockwise yaw rotation
    
    occupancy_grid = map_img.flatten()
    occupancy_grid = occupancy_grid.tolist()

    rotation_from_bev_frame_to_desired_frame = R.from_euler("xyz",pose[3:])
    r = rotation_from_bev_frame_to_desired_frame.as_quat()
    r_mat = rotation_from_bev_frame_to_desired_frame.as_matrix()
    first_cell_location_in_bev_frame = np.array([0,-MAP_WIDTH/2,0])+pose[:3]
    first_cell_location_in_desired_frame = np.matmul(r_mat,first_cell_location_in_bev_frame)
    map_msg = OccupancyGrid()

    map_msg.header = Header()
    map_msg.header.frame_id = frame_id
    map_msg.header.stamp = time_stamp

    map_msg.info = MapMetaData()
    map_msg.info.height = int(MAP_WIDTH / MAP_RESOLUTION)  # Unit: Pixel

    map_msg.info.width = int(MAP_HEIGHT / MAP_RESOLUTION)  # Unit: Pixel
    map_msg.info.resolution = MAP_RESOLUTION
    # Quat_from_cv2img_to_baselink =  Quat_from_cameralink_to_baselink*Quat_from_cv2img_to_cameralink
    # the location of (0,0) cell w.r.t to the origin of the current frame
    map_msg.info.origin = Pose()
    map_msg.info.origin.position = Point()
    map_msg.info.origin.position.x = first_cell_location_in_desired_frame[0]  # Unit: Meter
    map_msg.info.origin.position.y = first_cell_location_in_desired_frame[1]# Unit: Meter
    map_msg.info.origin.position.z = first_cell_location_in_desired_frame[2]
    map_msg.info.origin.orientation = Quaternion()
    # map_msg.info.origin.orientation.x = Quat_from_cv2img_to_cameralink[0]
    # map_msg.info.origin.orientation.y = Quat_from_cv2img_to_cameralink[1]
    # map_msg.info.origin.orientation.z = Quat_from_cv2img_to_cameralink[2]
    # map_msg.info.origin.orientation.w = Quat_from_cv2img_to_cameralink[3]    
    map_msg.info.origin.orientation.x = r[0]
    map_msg.info.origin.orientation.y = r[1]
    map_msg.info.origin.orientation.z = r[2]
    map_msg.info.origin.orientation.w = r[3]
    map_msg.data.extend(occupancy_grid)
    map_msg.info.map_load_time = rospy.Time.now()
    return map_msg