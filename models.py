import tensorflow as tf
import numpy as np
import cv2


class ENET:
    INPUT_TENSOR_NAME = "input0:0"
    OUTPUT_TENSOR_NAME = "CATkrIDy/concat:0"

    def __init__(self, GRAPH_PB_PATH):
        self.sess = tf.compat.v1.Session()
        with tf.compat.v1.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
            string = f.read()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(string)
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

    def predict(self, image):
        segmap = self.sess.run(self.OUTPUT_TENSOR_NAME,
                               feed_dict={self.INPUT_TENSOR_NAME: image})
        return segmap


class DeepLabV3:
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'import/ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'import/SemanticPredictions:0'
    INPUT_SIZE = 1024
    FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

    def __init__(self, GRAPH_PB_PATH):
        self.sess = tf.compat.v1.Session()
        with tf.compat.v1.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")
        for node in graph_def.node:
            print(node.name)

    def predict(self, img):
        print(img.shape)
        batch, depth, height, width = img.shape
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        print(target_size)
        resized_image = img  #cv2.resize(img, target_size)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        return batch_seg_map
