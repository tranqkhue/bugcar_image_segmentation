import cv2
import numpy as np
from bev import bev_transform_tools
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from utils import order_points, clahe

is_calibrating = False
frame = []
ordered_corners_list = []
refined_corners = np.zeros(shape=(4, 2))
distance_z = -1
distance_x = -1
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()
REALSENSE_RESOLUTION = (1280, 720)
FPS = 15
IMG_SHAPE = (1280, 720)
resize_ratio = (IMG_SHAPE[0] / REALSENSE_RESOLUTION[0],
                IMG_SHAPE[1] / REALSENSE_RESOLUTION[1])

MARKER_LENGTH = 0.268
CM_PER_PX = 2


def calibrate_with_median(corners_list):
    global is_calibrating, refined_corners
    is_calibrating = False
    corners_list = np.asarray(corners_list)
    fig, ax = plt.subplots(2, 2)
    for i in range(4):
        x = corners_list[:, i, 0]
        y = corners_list[:, i, 1]
        refined_corners[i] = np.array([np.mean(x), np.mean(y)])
        ax[i // 2, i % 2].hist2d(x, y)
    plt.show()
    plt.close(fig=fig)
    refined_corners = order_points(refined_corners)
    print(refined_corners)


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


def main():
    global is_calibrating
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, REALSENSE_RESOLUTION[0],
                         REALSENSE_RESOLUTION[1], rs.format.rgb8, FPS)
    pipeline.start(config)
    try:
        while True:
            #find camera matrix and distortion coefficient
            #================================================
            pipeline_frames = pipeline.wait_for_frames()
            pipeline_rgb_frame = pipeline_frames.get_color_frame()

            frame = np.asanyarray(pipeline_rgb_frame.get_data())
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
            result = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                markerLength=MARKER_LENGTH,
                distCoeffs=distortion_coeffs,
                cameraMatrix=K)
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
                        resized_corner = np.round(resize_ratio * corners[0][0])
                        ordered_points = order_points(resized_corner)
                        ordered_corners_list.append(ordered_points)
                    else:
                        calibrate_with_median(ordered_corners_list)
            #================================================

            # Find the yaw of camera with respect to the aruco's coordinate frame
            #================================================
            rvec = result[0]
            tvec = result[1]
            if rvec is not None:

                # rvec is the orientation of the camera relative to the aruco marker.
                # Sometimes, there will be 2 possible rotation vectors for 1 marker.
                # It is not clear as to what is causing this behaviour, further research needed!
                for rotation in rvec:
                    rotation_mat_fid2cam, _ = cv2.Rodrigues(rotation)
                    rotation_mat_cam2fid = np.linalg.inv(rotation_mat_fid2cam)
                    # the formula can be found here
                    #http://planning.cs.uiuc.edu/node102.html#eqn:yprmat
                    yaw_cam2fid = np.arctan2(rotation_mat_cam2fid[1, 0],
                                             rotation_mat_cam2fid[0, 0])

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

            #processing translation vector information
            # translation vector represent the position of fiducial relative to the camera frame, more info below
            #================================================
            if tvec is not None:
                tvec = np.array(tvec)[0][0]
                # print("tvec raw", tvec)
                # print(yaw)
                distance_x, distance_y = find_distance_from_fiducial_to_camera_in_bev_frame(
                    tvec, yaw_cam2fid, rotation_mat_cam2fid)

                # print(tvec_fiducial_frame)
                # there are 3 kind of distances with respect to 3 axes x,y,z:
                # z is how far you are from the target
                # x is how left you are from the target. >0 means your camera is to the left of the target
                # y is how low your are from the target. >0 means that the target is above the baseline of your camera
                # https://answers.opencv.org/question/197197/what-does-rvec-and-tvec-of-aruco-markers-correspond-to/
            #================================================

            # special keys to interact with the program
            #================================================
            frame = cv2.resize(frame, IMG_SHAPE)
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
                if yaw_cam2fid >= -np.pi / 4 and yaw_cam2fid < np.pi / 4:
                    yaw_cam2fid = -yaw_cam2fid
                elif yaw_cam2fid >= np.pi / 4 and yaw_cam2fid < 3 * np.pi / 4:
                    yaw_cam2fid = np.pi / 2 - yaw_cam2fid
                elif yaw_cam2fid >= 3 * np.pi / 4 and yaw_cam2fid < 5 * np.pi / 4:
                    yaw_cam2fid = np.pi - yaw_cam2fid
                else:
                    yaw_cam2fid = 3 * np.pi / 2 - yaw_cam2fid
                print("yaw is", yaw_cam2fid)
                bev_tool = bev_transform_tools(
                    IMG_SHAPE, (distance_x * 100, distance_y * 100),
                    MARKER_LENGTH * 100, CM_PER_PX, yaw_cam2fid)
                bev_tool.calculate_transform_matrix(refined_corners)
                bev_tool.create_occ_grid_param(10, 0.1)
                bev_tool.save_to_JSON("calibration_data.json")
                mat = bev_tool._bev_matrix
                warped = cv2.warpPerspective(frame, mat, IMG_SHAPE)
                cv2.imshow("warped", warped)


#================================================

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

main()
