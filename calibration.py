import cv2
import csv
import rospy
import numpy as np
from fiducial_msgs.msg import FiducialTransformArray
from bev import bev_transform_tools
import matplotlib.pyplot as plt
import asyncio
import pyrealsense2 as rs

is_calibrating = False
frame = []
ordered_corners_list = []
refined_corners = []
distance_z = -1
distance_x = -1
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()
REALSENSE_RESOLUTION = (1280, 720)
FPS = 15
IMG_SHAPE = (1024, 512)
resize_ratio = (IMG_SHAPE[0] / REALSENSE_RESOLUTION[0],
                IMG_SHAPE[1] / REALSENSE_RESOLUTION[1])

MARKER_LENGTH = 0.557


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def calibrate_with_median(corners_list):
    global is_calibrating, refined_corners
    is_calibrating = False
    corners_list = np.asarray(corners_list)
    fig, ax = plt.subplots(2, 2)
    for i in range(4):
        x = corners_list[:, i, 0]
        y = corners_list[:, i, 1]
        refined_corners.append(np.array([np.mean(x), np.mean(y)]))
        ax[i // 2, i % 2].hist2d(x, y)
    plt.show()
    plt.close(fig=fig)
    refined_corners = order_points(refined_corners)
    print(refined_corners)


def order_points(points):
    sort_by_height = lambda e: e[1]
    points.sort(key=sort_by_height)
    top_points = points[:2]
    bot_points = points[2:]
    if top_points[0][0] > top_points[1][0]:
        top_left_point = top_points[1]
        top_right_point = top_points[0]
    else:
        top_left_point = top_points[0]
        top_right_point = top_points[1]
    if bot_points[0][0] > bot_points[1][0]:
        bot_left_point = bot_points[1]
        bot_right_point = bot_points[0]
    else:
        bot_left_point = bot_points[0]
        bot_right_point = bot_points[1]

    return np.array(
        [top_left_point, bot_left_point, top_right_point, bot_right_point])


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
            rgb_intrin = pipeline_rgb_frame.profile.as_video_stream_profile(
            ).intrinsics

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
                        ordered_points = order_points(resized_corner.tolist())
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
                    rotation_matrix_cam_ref, _ = cv2.Rodrigues(rotation)
                    rotation_matrix_fiducial_ref = np.linalg.inv(
                        rotation_matrix_cam_ref)
                    # the formula can be found here
                    #http://planning.cs.uiuc.edu/node102.html#eqn:yprmat
                    yaw = np.arctan2(rotation_matrix_fiducial_ref[1, 0],
                                     rotation_matrix_fiducial_ref[0, 0])

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
            # translation vector represent the position of camera relative to the fiducial marker, more info below
            #================================================
            if tvec is not None:
                tvec = np.array(tvec)[0][0]
                # print("tvec raw", tvec)
                # print(yaw)
                yaw_matrix_fiducial_ref = np.array(
                    [[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw), np.cos(yaw), 0],\
                     [0, 0, 1]])
                yaw_matrix_bev_ref = np.linalg.inv(yaw_matrix_fiducial_ref)
                rotation_matrix_fiducial_ref_yaw_compensated = np.matmul(
                    yaw_matrix_bev_ref, rotation_matrix_fiducial_ref)
                tvec_fiducial_frame = np.matmul(
                    rotation_matrix_fiducial_ref_yaw_compensated, tvec)

                # print(tvec_fiducial_frame)
                # there are 3 kind of distances with respect to 3 axes x,y,z:
                # z is how far you are from the target
                # x is how left you are from the target. >0 means your camera is to the left of the target
                # y is how low your are from the target. >0 means that the target is above the baseline of your camera

                distance_x = tvec_fiducial_frame[0]
                distance_y = tvec_fiducial_frame[1]
                # https://answers.opencv.org/question/197197/what-does-rvec-and-tvec-of-aruco-markers-correspond-to/
            #================================================

            # special keys to interact with the program
            #================================================
            frame = cv2.resize(frame, (1024, 512))
            cv2.imshow('image', frame)

            key = cv2.waitKey(25)
            if (key & 0xFF == ord("c")):
                is_calibrating = True
                print("start calibrating,hold the aruco marker still")
            elif (key & 0xFF == ord("q")):
                cv2.destroyAllWindows()
                break
            elif (key & 0xFF == ord("s")):
                bev_tool = bev_transform_tools(
                    IMG_SHAPE, (distance_x * 100, distance_y * 100),
                    MARKER_LENGTH * 100, 1)

                # reverse = np.sign(top_right_y - top_left_y)
                print("yaw is", yaw)
                if yaw >= -np.pi / 4 and yaw < np.pi / 4:
                    yaw = -yaw
                elif yaw >= np.pi / 4 and yaw < 3 * np.pi / 4:
                    yaw = np.pi / 2 - yaw
                elif yaw >= 3 * np.pi / 4 and yaw < 5 * np.pi / 4:
                    yaw = np.pi - yaw
                else:
                    yaw = 3 * np.pi / 2 - yaw
                yaw = -yaw

                bev_tool.calculate_transform_matrix(refined_corners, yaw)
                bev_tool.create_occ_grid_param(10, 0.1)
                bev_tool.save_to_JSON("calibration_data.json")
                mat = bev_tool._intrinsic_matrix
                warped = cv2.warpPerspective(frame, mat, (1024, 512))
                # cv2.imshow("only_M", warped_M)
                cv2.imshow("warped", warped)
                # c = cv2.waitKey(1) % 0x100
                # if (c == 27):
                #     break


#================================================

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

main()
