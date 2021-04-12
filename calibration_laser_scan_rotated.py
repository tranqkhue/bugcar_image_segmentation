#!/home/tranquockhue/anaconda3/envs/tf2.2/bin/python
from occgrid_to_ros import og_msg, OccupancyGrid
import cv2
import numpy as np
import pyrealsense2 as rs
import rospy


# ================================================================================


def convert2pts(depth_frame):
    points = pc.calculate(depth_frame)
    v = points.get_vertices()
    vtx = np.asanyarray(v).view(np.float32).reshape(-1, 3)
    return vtx
# ---------------------------------------------------------------------------------


def transform(pts, M):
    pts = pts.T
    transformed_pts = M[:, [0]]*pts[[0]]+M[:, [1]]*pts[[1]]+M[:, [2]]*pts[[2]]
    return transformed_pts.T


def pts2og(pts, angle_step=np.pi/180/4, field_of_view=86.0/180.0*np.pi):
    grid_size = int(MAP_SIZE/MAP_RESOLUTION)
    og = np.zeros((grid_size, grid_size))-1

    # define the minimum angle and max angle the camera can cover
    angle_min = np.pi/2 - field_of_view/2
    angle_max = np.pi/2 + field_of_view/2  # - 3/180*np.pi
    max_distance = 4  # points that are above this distance will be discarded from og
    angle_bin = np.arange(start=angle_min, stop=angle_max, step=angle_step)
    bin_num = angle_bin.shape[0]
    x_max = max_distance*np.cos(angle_bin)
    y_max = max_distance*np.sin(angle_bin)
    placeholder_pts = np.stack([x_max, y_max], axis=1)

    y = pts[:, 1]
    pts = pts[y <= max_distance]
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    angle_is_valid = np.logical_and(
        theta >= angle_min, theta <= angle_max)
    pts = pts[angle_is_valid]
    theta = theta[angle_is_valid]
    binned_theta = np.round(
        (theta-angle_min)/angle_step).astype(np.int)
    dist = np.linalg.norm(pts, axis=1)
    sorted_by_theta_and_dist_index = np.lexsort((dist, binned_theta))
    angle, index_new = np.unique(
        binned_theta[sorted_by_theta_and_dist_index], return_index=True)
    sorted_by_theta_and_dist_index = sorted_by_theta_and_dist_index[index_new]
    sorted_by_angle_occupied_pts = pts[sorted_by_theta_and_dist_index]
    new_pts = sorted_by_angle_occupied_pts

    # add default value to rightmost and leftmost missing angles that are not provided by laser detector.
    if len(angle) > 0:
        # print(angle[[0, -1]], bin_num)
        if angle[0] > 0:
            new_pts = np.concatenate([placeholder_pts[0:angle[0]+1], new_pts])
        if angle[-1] < bin_num-1:
            new_pts = np.concatenate(
                [new_pts, placeholder_pts[angle[-1]:bin_num]], axis=0)
    else:
        new_pts = placeholder_pts

    new_pts = np.append(
        new_pts, np.array([[0, 0]]), axis=0)/MAP_RESOLUTION
    origin = np.array([grid_size/2, 0])
    new_pts += origin
    new_pts = new_pts.astype(np.int32)
    og = cv2.fillPoly(og, [new_pts], 0)
    for pt in (sorted_by_angle_occupied_pts/MAP_RESOLUTION + origin).astype(np.int):
        cv2.circle(og, (pt[0], pt[1]), radius=1, color=100, thickness=-1)
    og = cv2.dilate(og, np.ones((3, 3)))
    return og
# ================================================================================


def signed_distance_to_plane(point, coeffs, bias):
    # dont use np.dot here or cpu usage will skyrocket.
    # https://www.pugetsystems.com/labs/hpc/How-To-Use-MKL-with-AMD-Ryzen-and-Threadripper-CPU-s-Effectively-for-Python-Numpy-And-Other-Applications-1637/
    a = (point[:, 0]*coeffs[0]+point[:, 1]*coeffs[1] +
         point[:, 2]*coeffs[2]+bias)/np.linalg.norm(coeffs)
    return a


# =================================================================================
if __name__ == "__main__":
    node = rospy.init_node("laser_high_cam", disable_signals=True)
    pub = rospy.Publisher("/depth_front", OccupancyGrid, queue_size=3)
    rate = rospy.Rate(20)
    M = np.load("rotmat_cam2bev.npy").astype(np.float32)
    z_axis_bev_frame = np.array([0, 0, 1])
    that_z_axis_but_in_cam_frame = np.dot(np.linalg.inv(M), z_axis_bev_frame)
    coeffs = that_z_axis_but_in_cam_frame
    bias = 0
    MAP_SIZE = 5
    MAP_RESOLUTION = 0.02

    # ===================== INITIALIZING CAMERA=================================
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device("841612070098")
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    pc = rs.pointcloud()
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 3)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    temporal = rs.temporal_filter()
    # ==========================================================================
    try:
        while True:
            t0 = rospy.Time.now()
            pipeline_frames = pipeline.wait_for_frames()
            pipeline_depth_frame = pipeline_frames.get_depth_frame()
            pts = convert2pts(pipeline_depth_frame)
            pts = pts[signed_distance_to_plane(
                pts, coeffs=coeffs, bias=bias) > -0.2]
            pts = pts[signed_distance_to_plane(pts, coeffs, bias) < 0.05]
            rotated_pts = transform(pts, M)  # 15% cpu
            # rotated_pts = rotated_pts[np.abs(rotated_pts[:, 2]) < 0.01]
            occ_grid = pts2og(rotated_pts[:, [0, 1]])
            occ_grid = cv2.flip(occ_grid, 0).astype(np.int8)
            msg = og_msg(occ_grid, MAP_RESOLUTION, MAP_SIZE, t0)
            pub.publish(msg)
    finally:
        pipeline.stop()
