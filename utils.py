import numpy as np
import cv2


def order_points(points, x_axis):
    #axis is a 2x2 array containing 2 coordinates,1st one for the center and 2nd for a point on the chosen axis
    print(x_axis)
    center = x_axis[0]
    translated_points = points - center
    x_axis -= center
    # this is the rotation from the axis to the x axis in the img
    rotation = -np.arctan2(x_axis[1, 1], x_axis[1, 0])
    rot_mat = np.array([[np.cos(rotation), -np.sin(rotation)],
                        [np.sin(rotation), np.cos(rotation)]])
    rotated_points = np.transpose(
        np.matmul(rot_mat, np.transpose(translated_points)))
    #the idea here is , we rotate the x_axis in fiducial to match the x_axis in camera, which will give us a rot matrix
    # apply that rot matrix to 4 corners of fiducial.
    # and we'll check if each points has pos or neg y_coordinate
    # if pos: the point is on the left side of the x_axis,
    # if neg: the point is on the right side of the x_axis
    # after that, sort each of the left and right array by x_coordinate.
    axis_left_side = []
    axis_right_side = []
    sort_by_x = lambda x: x["point"][0]
    order_point = []
    for i, point in enumerate(rotated_points):
        if point[1] < 0:
            axis_right_side.append({"order": i, "point": point})
        else:
            axis_left_side.append({"order": i, "point": point})
    axis_left_side.sort(key=sort_by_x)
    axis_right_side.sort(key=sort_by_x)
    for point in axis_left_side:
        order_point.append(point["order"])
    for point in axis_right_side:
        order_point.append(point["order"])
    sorted_points = points[order_point]
    return sorted_points


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