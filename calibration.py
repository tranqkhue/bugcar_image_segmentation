import cv2
import csv
import rospy
import numpy as np
from fiducial_msgs.msg import FiducialTransformArray
from bev import bev_transform_tools

# def test_device(source):
#     cap = cv2.VideoCapture(source)
#     if cap is None or not cap.isOpened():
#         print("Warning: unable to open video source ", source)
#         return False, cap
#     return True, cap

# isValid = False
# available_cam = []
# for i in range(8):
#     isValid, cap_test = test_device(i)
#     if isValid:
#         available_cam.append(cap_test)
# cap = available_cam[2]


def sort_by_height(haha):
    return haha[1]


def order_point(points):
    sort_by_height = lambda e: e[1]
    points.sort(key=sort_by_height)
    print(points)
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


# camera_matrix = np.array([626.587239 ,0.0 ,318.410833 ,0.0 ,627.261398 ,245.357087 ,0.0 ,0.0, 1.0])
# camera_matrix = np.reshape(camera_matrix,(3,3))
# distortion_coeffs = np.array([0.186030, -0.475013, 0.000964 ,-0.006895 ,0.0])
# this is the matrix we get from camera_calibration

camera_matrix = np.array([
    615.1647338867188, 0.0, 319.7691345214844, 0.0, 615.38037109375,
    242.72280883789062, 0.0, 0.0, 1.0
])
camera_matrix = np.reshape(camera_matrix, (3, 3))
distortion_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
# this is the matrix we get from camera_info

frame = []
square_markers = []
distance_z = -1
distance_x = -1


def addPoint(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if (len(square_markers) == 4):
            print("already have 4 points")
            return
        square_markers.append((x, y))


cv2.namedWindow('image')
cv2.setMouseCallback("image", addPoint)
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()
IMG_SHAPE = (1024, 512)
MARKER_LENGTH = 0.269

import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
pipeline.start(config)
try:
    while True:
        pipeline_frames = pipeline.wait_for_frames()
        pipeline_rgb_frame = pipeline_frames.get_color_frame()
        rgb_intrin = pipeline_rgb_frame.profile.as_video_stream_profile(
        ).intrinsics

        frame = np.asanyarray(pipeline_rgb_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fx = pipeline_rgb_frame.profile.as_video_stream_profile().intrinsics.fx
        fy = pipeline_rgb_frame.profile.as_video_stream_profile().intrinsics.fy
        ppx = pipeline_rgb_frame.profile.as_video_stream_profile(
        ).intrinsics.ppx
        ppy = pipeline_rgb_frame.profile.as_video_stream_profile(
        ).intrinsics.ppy
        K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        distortion_coeffs = np.array(
            pipeline_rgb_frame.profile.as_video_stream_profile(
            ).intrinsics.coeffs)

        (corners, ids,
         rejected) = cv2.aruco.detectMarkers(frame,
                                             aruco_dict,
                                             parameters=aruco_params)
        result = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            markerLength=MARKER_LENGTH,
            distCoeffs=distortion_coeffs,
            cameraMatrix=K)
        rvec = result[0]
        tvec = result[1]
        if rvec is not None:
            for point in corners[0][0]:
                cv2.circle(frame, (point[0], point[1]), 1, (0, 255, 0), -1)

            tvec = np.array(tvec)[0][0]
            # there are 3 kind of distance x,y,z:
            # z is how far you are from the target
            # x is how 'left'ish you are from the target. >0 means your camera is to the left of the target
            # y is how 'down'ish your are from the target. >0 means that the target is above the baseline of your camera
            distance_z = tvec[2]
            distance_x = tvec[0]
            distance_y = tvec[1]
            yaw = 0
            # rvec is the orientation of the camera relative to the aruco marker.
            # Sometimes, there will be 2 possible rotation vectors for 1 marker.
            # It is not clear as to what is causing this behaviour, further research needed!

            for rotation in rvec:
                # print(rotation)
                rotation_matrix, _ = cv2.Rodrigues(rotation)
                rotation_matrix = np.linalg.inv(rotation_matrix)
                # print(rotation_matrix)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                # pitch = np.arctan2(
                #     -rotation_matrix[2, 0],
                #     np.sqrt(rotation_matrix[2, 1]**2 +
                #             rotation_matrix[2, 2]**2))
                # roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                print(np.degrees(yaw))
                cv2.aruco.drawAxis(frame, K, distortion_coeffs, rotation, tvec,
                                   0.1)
                # aruco:
                # blue  represent z axis
                # red represent x axis
                # green represent y axis

                # camera axes:
                # y-axis heads downward
                # x-axis heads right
                # z-axis heads forward
                # to sum up, camera axes follows NED format.

        frame = cv2.resize(frame, (1024, 512), interpolation=cv2.INTER_AREA)

        for point in square_markers:
            cv2.circle(frame, (point[0], point[1]), 1, (255, 0, 0), -1)
        cv2.imshow('image', frame)
        key = cv2.waitKey(25)
        if (key & 0xFF == 8):  # this is backspace button
            square_markers.pop(-1)
        elif (key & 0xFF == ord("q")):
            cv2.destroyAllWindows()
            break
        elif (key & 0xFF == ord("s")):
            if len(square_markers) < 4:
                print("cant save, not enough point")
                continue
            bev_tool = bev_transform_tools(
                IMG_SHAPE, (distance_x * 100, distance_z * 100),
                MARKER_LENGTH * 100)
            ordered_tile_coords = order_point(square_markers)
            print(ordered_tile_coords)
            bev_tool.calculate_transform_matrix(ordered_tile_coords, yaw)
            print(bev_tool.dero, bev_tool.detran, bev_tool.M)
            bev_tool.create_occ_grid_param(10, 0.1)
            bev_tool.save_to_JSON("calibration_data.json")
            c = cv2.waitKey(1) % 0x100
            if (c == 27):
                break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()