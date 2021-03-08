import numpy as np
import cv2
import tensorflow as tf


def contour_noise_removal(segmap):
    #Close small gaps by a kernel with shape proporition to the image size
    h_segmap, w_segmap = segmap.shape
    min_length = min(h_segmap, w_segmap)
    kernel = np.ones((int(min_length / 50), int(min_length / 50)), np.uint8)
    closed = cv2.morphologyEx(segmap, cv2.MORPH_CLOSE, kernel)

    #Find contours of segmap
    cnts, hie = cv2.findContours(closed, 1, 2)
    cnts = list(filter(lambda cnt: cnt.shape[0] > 2, cnts))

    # Create a rectangular mask in lower part of the frame
    # If a contour intersect with this lower part above a threshold
    # then that contour will be kept as a valid one

    LENGTH_RATIO = 0.1
    x_left = 0
    x_right = w_segmap
    y_top = int(h_segmap * (1 - LENGTH_RATIO))
    y_bot = h_segmap
    bottom_rect = np.array([(x_left,  y_top), (x_right, y_top),\
       (x_right, y_bot), (x_left,  y_bot)])
    bottom_mask = np.zeros_like(segmap)
    mask_area = (x_right - x_left) * (y_bot - y_top)
    cv2.fillPoly(bottom_mask, [bottom_rect], 1)

    # Iterate through contour[S]
    MASK_AREA_THRESH = 0.4  #The threshold of intersect over whole mask area
    main_road_cnts = []
    for cnt in cnts:
        cnt_map = np.zeros_like(segmap)
        cv2.fillPoly(cnt_map, [cnt], 1)
        masked = cv2.bitwise_and(cnt_map, bottom_mask).astype(np.uint8)
        intersected_area = np.count_nonzero(masked)
        if (intersected_area > (mask_area * MASK_AREA_THRESH)):
            main_road_cnts.append(cnt)

    contour_noise_removed = np.zeros(segmap.shape).astype(np.uint8)
    cv2.fillPoly(contour_noise_removed, main_road_cnts, 1)

    return contour_noise_removed


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


def find_intersection_line(line1, line2):
    # line: 2x2 array representing 2 point on the line (x1,y1),(x2,y2).
    (x1, y1) = line1[0]
    (x2, y2) = line1[1]
    if x2 - x1 == 0:
        b1 = 0
        a1 = 1
        c1 = x1
    else:
        b1 = -1
        a1 = (y2 - y1) / (x2 - x1)
        c1 = (x1 * y2 - x2 * y1) / (x2 - x1)

    (x3, y3) = line2[0]
    (x4, y4) = line2[1]
    if x4 == x3:
        b2 = 0
        a2 = 1
        c2 = x3
    else:
        b2 = -1
        a2 = (y4 - y3) / (x4 - x3)
        c2 = (x3 * y4 - x4 * y3) / (x4 - x3)
    if a1 == a2:
        return
    coeff = np.array([[a1, b1], [a2, b2]])
    res = np.array([c1, c2])
    intersection = np.linalg.solve(coeff, res)
    return intersection


def enet_preprocessing(bgr_frame):
    IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGE_STD = np.array([0.229, 0.224, 0.225])
    input_size = (512, 256)

    resized = cv2.resize(bgr_frame, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #rotate = cv2.rotate(rgb, cv2.cv2.ROTATE_180)
    # Normalize, some statistics and stack into a batch for interference
    normalized = rgb / 256.0
    subtracted = np.subtract(normalized, IMAGE_MEAN)
    divided = np.divide(subtracted, IMAGE_STD)
    swap_axesed = np.swapaxes(divided, 1, 2)
    swap_axesed = np.swapaxes(swap_axesed, 0, 1)
    batch = np.array([swap_axesed])

    return batch


def freeze_session(session,
                   keep_var_names=None,
                   output_names=None,
                   clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    # with tf.compat.v1.get_default_graph() as graph:
    #     graph.
    print(output_names)
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.compat.v1.global_variables()).difference(
                keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph