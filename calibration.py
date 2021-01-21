import cv2
import csv 
import rospy
import numpy as np
from fiducial_msgs.msg import FiducialTransformArray
def test_device(source):
    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        print("Warning: unable to open video source ",source)
        return False,cap
    return True,cap
isValid = False
available_cam = []
for i in range(8):
    isValid,cap_test = test_device(i)
    if isValid:
        available_cam.append(cap_test)
cap = available_cam[2]
print(cap.isOpened())


# camera_matrix = np.array([626.587239 ,0.0 ,318.410833 ,0.0 ,627.261398 ,245.357087 ,0.0 ,0.0, 1.0])
# camera_matrix = np.reshape(camera_matrix,(3,3))
# distortion_coeffs = np.array([0.186030, -0.475013, 0.000964 ,-0.006895 ,0.0])
#this is the matrix we get from camera_calibration

camera_matrix = np.array([615.1647338867188, 0.0, 319.7691345214844, 0.0, 615.38037109375, 242.72280883789062, 0.0, 0.0, 1.0])
camera_matrix = np.reshape(camera_matrix,(3,3))
distortion_coeffs = np.array([0.0,0.0,0.0,0.0,0.0])
# this is the matrix we get from camera_info

frame = []
square_markers=[]
distance_z = -1
distance_x = -1
def addPoint(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if(len(square_markers)==4):
            print("already have 4 points")
            return
        square_markers.append((x,y))


# rospy.Subscriber("/fiducial_transform",FiducialTransformArray,getDistance)
cv2.namedWindow('image')
cv2.setMouseCallback("image",addPoint)
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters_create()
with open("calibration_data.txt",mode="w+") as csv_file:
    point_writer = csv.writer(csv_file,delimiter=',')
    while not rospy.is_shutdown():
        ret,frame = cap.read()
        if not ret :
            continue 
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        for point in square_markers:
            cv2.circle(frame,(point[0],point[1]),2,(255,0,0),-1)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict,parameters=aruco_params)
        cv2.aruco.drawDetectedMarkers(frame,corners)
        result = cv2.aruco.estimatePoseSingleMarkers(corners,markerLength=0.269,distCoeffs=distortion_coeffs,cameraMatrix=camera_matrix)
        rvec = result[0]
        tvec = result[1]
        if rvec is not None:
            rvec = np.array(rvec)
            # print(rvec)
            tvec = np.array(tvec)[0][0]
            distance_z = tvec[2]
            distance_x = tvec[0]
            distance_y = tvec[1]
            print("distance x is:",distance_x)
            print("distance y is:",distance_y)
            print("distance z is:",distance_z)
            cv2.aruco.drawAxis(frame,camera_matrix,distortion_coeffs,rvec,tvec,0.1)
        for points in corners:
            for point in points[0]:
                point = (int(point[0]),int(point[1]))
                cv2.circle(frame,point,2,(0,255,0),-1)
        cv2.imshow("image",frame)
        key = cv2.waitKey(25)
        if ( key & 0xFF == 8):
            square_markers.remove()
        elif (key & 0xFF == ord("q")):
            cv2.destroyAllWindows()
            break
        elif(key & 0xFF == ord("s")):
            if len(square_markers) < 4:
                print("cant save, not enough point")
                continue
            write_list = [coordinate for coordinate in square_markers]
            write_list.append(distance_z)
            write_list.append(distance_x)
            print(write_list)
            point_writer.writerow(write_list)
            csv_file.flush()
