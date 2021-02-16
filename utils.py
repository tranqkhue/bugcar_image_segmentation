import numpy as np
import cv2


def order_points(points):
    points = points.tolist()
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