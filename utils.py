import numpy as np
import cv2
from numpy.core.fromnumeric import swapaxes
import tensorflow as tf





def order_points_counter_clockwise(points, x_axis):
    # axis is a 2x2 array containing 2 coordinates,1st one for the center and 2nd for a point on the chosen axis
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
    # the idea here is , we rotate the x_axis in fiducial to match the x_axis in camera, which will give us a rot matrix
    # apply that rot matrix to 4 corners of fiducial.
    # and we'll check if each points has pos or neg y_coordinate
    # if pos: the point is on the left side of the x_axis,
    # if neg: the point is on the right side of the x_axis
    # after that, sort each of the left and right array by x_coordinate.
    axis_left_side = []
    axis_right_side = []
    def sort_by_x(x): return x["point"][0]
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


def testDevice():
    for source in range(10):
        cap = cv2.VideoCapture(source)
        if cap is None or not cap.isOpened():
            print('Warning: unable to open video source: ', source)
