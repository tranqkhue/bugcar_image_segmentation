import sys
from time import time
import cv2
import numpy as np
from bev import bev_transform_tools
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from utils import order_points, clahe
import argparse

INPUT_SHAPE = (1920, 1080)
FPS = 30
MAP_SIZE = 10
CELL_SIZE = 0.1


def calibrate_with_median(corners_list):
    global is_calibrating, axis
    is_calibrating = False
    corners_list = np.asarray(corners_list)
    print(corners_list.shape)
    fig, ax = plt.subplots(2, 2)
    refined_corners = np.zeros(shape=(4, 2))
    for i in range(4):
        x = corners_list[:, i, 0]
        y = corners_list[:, i, 1]
        refined_corners[i] = np.array([np.mean(x), np.mean(y)])
        ax[i // 2, i % 2].hist2d(x, y)
    plt.show()
    plt.close(fig=fig)
    refined_corners = order_points(refined_corners, axis)
    return refined_corners


def find_distance_from_fiducial_to_camera_in_bev_frame(tvec, yaw_bev2fid,
                                                       rot_mat_cam2fid):
    # yaw_mat_cam2fid or yaw_mat_bev2fid, they are all the same, since only yaw is taken into account
    # and bev frame really doesn't have any kind of rotation beside yaw.
    # however , the ROTATION matrix of cam2fid and bev2fid are not identical.
    yaw_mat_bev2fid = np.array(
        [[np.cos(yaw_bev2fid), -np.sin(yaw_bev2fid), 0],
         [np.sin(yaw_bev2fid), np.cos(yaw_bev2fid), 0],\
         [0, 0, 1]])
    yaw_mat_fid2bev = np.linalg.inv(yaw_mat_bev2fid)
    rot_mat_cam2bev = np.matmul(yaw_mat_fid2bev, rot_mat_cam2fid)
    tvec_bev_frame = np.matmul(rot_mat_cam2bev, tvec)
    # print(tvec_fiducial_frame)
    # there are 3 kind of distances with respect to 3 axes x,y,z:
    # z is how far you are from the target
    # x is how left you are from the target. >0 means your camera is to the left of the target
    # y is how low your are from the target. >0 means that the target is above the baseline of your camera
    distance_x = tvec_bev_frame[0]
    distance_y = tvec_bev_frame[1]
    return distance_x, distance_y


if __name__ == "__main__":
    ## add argument
    #================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--marker_length",
                        help="the size of the aruco marker (in metres)",
                        type=float)
    parser.add_argument("-c",
                        "--cm_per_px",
                        help="how many cms does 1 px in bev image represent",
                        type=int)
    parser.add_argument("-s",
                        "--warped_shape",
                        help="the size of the bev image",
                        type=int,
                        nargs=2)
    args = parser.parse_args()
    # input shape for bev, also for realsense ioreader to function correctly
    # accepted params are (640,480),(1280,720),(1920,1680)

    if not args.warped_shape:
        WARPED_IMG_SHAPE = (768, 1024)
    else:
        WARPED_IMG_SHAPE = args.warped_shape
    if not args.marker_length:
        MARKER_LENGTH = 0.5
    else:
        MARKER_LENGTH = args.marker_length
    if not args.cm_per_px:
        CM_PER_PX = 2
    else:
        CM_PER_PX = args.cm_per_px

    print(args)
    # print(CM_PER_PX, WARPED_IMG_SHAPE, MARKER_LENGTH)
    #================================================================

    is_calibrating = False
    frame = []
    ordered_corners_list = []
    refined_corners = np.zeros(shape=(4, 2))
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    axis = np.zeros(shape=[2, 2])
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, INPUT_SHAPE[0], INPUT_SHAPE[1],
                         rs.format.rgb8, FPS)
    pipeline.start(config)
    try:
        while True:
            #find camera matrix and distortion coefficient
            #================================================
            t0 = time()
            pipeline_frames = pipeline.wait_for_frames()
            pipeline_rgb_frame = pipeline_frames.get_color_frame()
            frame = np.asanyarray(pipeline_rgb_frame.get_data())
            print("fps:", 1 / (time() - t0))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fx = pipeline_rgb_frame.profile.as_video_stream_profile(
            ).intrinsics.fx
            fy = pipeline_rgb_frame.profile.as_video_stream_profile(
            ).intrinsics.fy
            ppx = pipeline_rgb_frame.profile.as_video_stream_profile(
            ).intrinsics.ppx
            ppy = pipeline_rgb_frame.profile.as_video_stream_profile(
            ).intrinsics.ppy
            K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
            distortion_coeffs = np.array(
                pipeline_rgb_frame.profile.as_video_stream_profile(
                ).intrinsics.coeffs)
            #================================================

            #detect corners and relative pose of fiducial to the camera.
            #================================================
            (corners, ids,
             rejected) = cv2.aruco.detectMarkers(clahe(frame),
                                                 aruco_dict,
                                                 parameters=aruco_params)
            # https://docs.opencv.org/master/d9/d6a/group__aruco.html#gafce26321f39d331bc12032a72b90eda6
            result = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                markerLength=MARKER_LENGTH,
                distCoeffs=distortion_coeffs,
                cameraMatrix=K)
            #================================================

            # Find the yaw of camera with respect to the aruco's coordinate frame
            #================================================
            rvec = result[0]
            tvec = result[1]
            if rvec is not None:
                # rvec is the orientation of the aruco marker relative to the camera.
                # Sometimes, there will be 2 possible rotation vectors for 1 marker.
                # It is not clear as to what is causing this behaviour, further research needed!
                for rotation in rvec:
                    # print(rotation)
                    rotation_mat_fid2cam, _ = cv2.Rodrigues(rotation)
                    rotation_mat_cam2fid = np.linalg.inv(rotation_mat_fid2cam)
                    # the formula can be found here
                    #http://planning.cs.uiuc.edu/node102.html#eqn:yprmat
                    yaw_cam2fid = np.arctan2(rotation_mat_cam2fid[1, 0],
                                             rotation_mat_cam2fid[0, 0])
                    # print("yaw_cam2fid", yaw_cam2fid, file=sys.stderr)
                    unit_vector_along_x_axis_in_fiducial_frame = np.array(
                        [0.3, 0, 0])
                    origin_fiducial = np.array([0, 0, 0])
                    axis_in_fiducial = np.stack([
                        origin_fiducial,
                        unit_vector_along_x_axis_in_fiducial_frame
                    ],
                                                axis=0)
                    # this vector indicate that the point is 1m,wrt x-axis, away from the fiducial center.
                    axis, _ = cv2.projectPoints(objectPoints=axis_in_fiducial,
                                                rvec=rotation,
                                                tvec=np.array(tvec)[0][0],
                                                cameraMatrix=K,
                                                distCoeffs=distortion_coeffs)
                    axis = np.reshape(axis, (2, 2))
                    cv2.aruco.drawAxis(frame, K, distortion_coeffs, rotation,
                                       np.array(tvec)[0][0], 0.1)
                    # aruco:
                    # blue  represent z axis
                    # red represent x axis
                    # green represent y axis
                    # camera axes:
                    # y-axis heads downward
                    # x-axis heads right
                    # z-axis heads forward
                    # to sum up, camera axes follows NED format.
            #================================================

            # set the logic for calibrating aruco's marked corners, which basically can jump anywhere(based on observation) within the 10x10 square
            # in the vicinity of the true corner.
            # To tackle this, for each calibration process during which the marker must be fixed,
            # 100 samples of the corners' position are collected and passed through a median filter.
            # The result will be the conclusive corners of the marker.
            #================================================
            if len(corners) > 0:
                for point in corners[0][0]:
                    cv2.circle(frame, (point[0], point[1]), 0, (0, 255, 0), -1)
                if is_calibrating == True:
                    if len(ordered_corners_list) < 100:
                        # these are the point before being resized
                        resized_corner = np.round(corners[0][0])
                        # ordered_points = order_points(resized_corner, axis=axis)
                        ordered_corners_list.append(resized_corner)
                    else:
                        refined_corners = calibrate_with_median(
                            ordered_corners_list)
            #================================================

            #processing translation vector information
            # translation vector represent the position of fiducial relative to the camera frame, more info below
            #================================================
            if tvec is not None:
                tvec = np.array(tvec)[0][0]
                distance_x, distance_y = find_distance_from_fiducial_to_camera_in_bev_frame(
                    tvec, yaw_cam2fid, rotation_mat_cam2fid)
                # there are 3 kind of distances with respect to 3 axes x,y,z:
                # z is how far you are from the target
                # x is how left you are from the target. >0 means your camera is to the left of the target
                # y is how low your are from the target. >0 means that the target is above the baseline of your camera
                # https://answers.opencv.org/question/197197/what-does-rvec-and-tvec-of-aruco-markers-correspond-to/
            #================================================

            # special keys to interact with the program
            #================================================
            cv2.imshow('image', frame)
            key = cv2.waitKey(25)
            if (key & 0xFF == ord("c")):
                is_calibrating = True
                print("start calibrating,hold the aruco marker still")
            elif (key & 0xFF == ord("q")):
                cv2.destroyAllWindows()
                break
            elif (key & 0xFF == ord("s")):
                print("yaw is", yaw_cam2fid)
                bev_tool = bev_transform_tools(
                    WARPED_IMG_SHAPE, (distance_x * 100, distance_y * 100),
                    MARKER_LENGTH * 100, CM_PER_PX, yaw_cam2fid)
                print("refined corners", refined_corners)
                print("with axis", axis)
                bev_tool.calculate_transform_matrix(
                    refined_corners, bev_tool.dist2target, bev_tool.cm_per_px,
                    bev_tool.width, bev_tool.height, bev_tool.tile_length,
                    bev_tool.yaw)
                bev_tool.create_occ_grid_param(MAP_SIZE, CELL_SIZE)
                bev_tool.save_to_JSON("calibration_data.json")
                mat = bev_tool._bev_matrix
                warped = cv2.warpPerspective(frame, mat, WARPED_IMG_SHAPE)
                cv2.imshow("warped", warped)
    #===============================================
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()