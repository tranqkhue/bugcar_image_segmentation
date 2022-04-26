import cv2
import tensorflow as tf
import numpy as np
from abc import ABC

from bugcar_image_segmentation.image_processing_utils import contour_noise_removal

class InferenceModel(ABC):
    def predict(self,preprocessed_image):
        pass
    @classmethod
    def preprocess(rgb_image):
        pass
class ENET(InferenceModel):
    INPUT_TENSOR_NAME = "input0:0"
    OUTPUT_TENSOR_NAME = "CATkrIDy/concat:0"
    IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGE_STD = np.array([0.229, 0.224, 0.225])
    input_size = (512, 256)

    def __init__(self, GRAPH_PB_PATH):
        self.sess = tf.compat.v1.Session()
        with tf.compat.v1.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
            string = f.read()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(string)
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            self.test = None
    # @profile
    def predict(self, preprocessed_imgs):
        segmap = self.sess.run(self.OUTPUT_TENSOR_NAME,
                               feed_dict={self.INPUT_TENSOR_NAME: preprocessed_imgs})
        # predict shape will have size (batch_size,num_of_class,h,w)
        # print("predict shape",segmap.shape)
        # result_by_class = np.argmax(segmap, axis=0)
        result_by_class = tf.math.argmax(segmap, axis=1) # TF accelerates this step. Don't know why
        # print("result_by_class shape",result_by_class.shape)

        # segmap_by_class = np.bitwise_or(result_by_class == 0, result_by_class == 1).astype(np.uint8)
        segmap_by_class = tf.bitwise.bitwise_or(tf.cast(result_by_class == 0, tf.uint8), \
                                                tf.cast(result_by_class == 1, tf.uint8))
        np_segmap_by_class = segmap_by_class.numpy()
        # Remove road branches (or noise) that are not connected to main branches
        # Main road branches go from the bottom part of the RGB map
        # (should be) right front of the vehicle
        # contour_noise_removed = contour_noise_removal(segmap)  # 5% cpu
        return np_segmap_by_class 

    @classmethod
    def preprocess(cls,bgr_frame):
        """ preprocess a bgr image to fit in enet model """
        resized = cv2.resize(bgr_frame, cls.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize, some statistics and stack into a batch for interference
        normalized = (rgb / 256.0 - cls.IMAGE_MEAN)/cls.IMAGE_STD
        swap_axesed = np.moveaxis(normalized, -1, 0)
        # print(swap_axesed.shape)
        batch = np.expand_dims(swap_axesed, 0)
        return batch


class DeepLabV3(InferenceModel):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'import/ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'import/SemanticPredictions:0'
    INPUT_SIZE = 1024
    FROZEN_GRAPH_NAME = 'deeplab.pb'

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
        resized_image = img  # cv2.resize(img, target_size)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        return batch_seg_map
    @classmethod
    def preprocess(cls,bgr_frame):
        """ preprocess a bgr image to fit in enet model """
        resized = cv2.resize(bgr_frame, cls.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize, some statistics and stack into a batch for interference
        normalized = (rgb / 256.0 - cls.IMAGE_MEAN)/cls.IMAGE_STD
        swap_axesed = np.moveaxis(normalized, -1, 0)
        # print(swap_axesed.shape)
        batch = np.expand_dims(swap_axesed, 0)
        return batch