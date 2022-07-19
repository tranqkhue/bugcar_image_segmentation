import cv2
import tensorflow as tf
import numpy as np
from abc import ABC

from .image_processing_utils import contour_noise_removal

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
    INPUT_WIDTH, INPUT_HEIGHT = (512, 256)

    def __init__(self, GRAPH_PB_PATH=None):
        self.sess = tf.compat.v1.Session()
        if GRAPH_PB_PATH == None:
            GRAPH_PB_PATH = "./pretrained_models/enet.pb"
        with tf.compat.v1.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
            string = f.read()
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(string)
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            self.test = None
    # def __init__(self):
    #     self.sess = tf.compat.v1.Session()
    #     with tf.compat.v1.gfile.GFile("./pretrained_models/enet.pb", 'rb') as f:
    #         string = f.read()
    #         graph_def = tf.compat.v1.GraphDef()
    #         graph_def.ParseFromString(string)
    #         self.sess.graph.as_default()
    #         tf.import_graph_def(graph_def, name='')
    #         self.test = None
    # @profile
    def predict(self, preprocessed_imgs):
        segmap = self.sess.run(self.OUTPUT_TENSOR_NAME,
                               feed_dict={self.INPUT_TENSOR_NAME: preprocessed_imgs})
        # flat_objects = [2,9] #only pavement and vegetation
        # we will denote : 
        # 1 road
        # 0 flat non-road object which incclude only pavement and vegetation in this case
        # 2 non flat objects, which is the rest


        # predict shape will have size (batch_size,num_of_class,h,w)
        print("predict shape",segmap.shape)
        # result_by_class = np.argmax(segmap, axis=0)
        result_by_class = tf.math.argmax(segmap, axis=1) # TF accelerates this step. Don't know why
        segmap_by_class = tf.ones(result_by_class.shape)*2
        segmap_by_class =  tf.where(tf.logical_or(result_by_class==2,result_by_class==9),0,segmap_by_class)
        segmap_by_class =  tf.where(tf.logical_or(result_by_class==0,result_by_class==1),1,segmap_by_class)
        # for i in range(15):
        #     hay = np.where(result_by_class.numpy().astype(np.float32)[0]==i,1.0,0.0)
        #     hay = np.stack([hay,hay,hay],axis=2)
        #     print(hay.shape)
        #     cv2.imshow("asd"+str(i),hay)
        # x = cv2.applyColorMap(result_by_class.numpy().astype(np.uint8)[0]*15, cv2.COLORMAP_JET)
        # cv2.imshow("x",x)
        # cv2.waitKey(1)
        np_segmap_by_class = segmap_by_class.numpy().astype(np.uint8)
        # print(np_segmap_by_class)
        return np_segmap_by_class 
    def predict_binary(self, preprocessed_imgs):
        segmap = self.sess.run(self.OUTPUT_TENSOR_NAME,
                               feed_dict={self.INPUT_TENSOR_NAME: preprocessed_imgs})
        # we will denote : 
        # 1 road
        # 0 non-road object 
        # predict shape will have size (batch_size,num_of_class,h,w)
        # result_by_class = np.argmax(segmap, axis=0)
        result_by_class = tf.math.argmax(segmap, axis=1) # TF accelerates this step. Don't know why  
        segmap_by_class = tf.bitwise.bitwise_or(tf.cast(result_by_class == 0, tf.uint8), \
                                                tf.cast(result_by_class == 1, tf.uint8))
        np_segmap_by_class = segmap_by_class.numpy()
        return np_segmap_by_class 

    @classmethod
    def preprocess(cls, bgr_frame):
        """ preprocess a bgr image to fit in enet model """
        resized = cv2.resize(bgr_frame, (cls.INPUT_WIDTH, cls.INPUT_HEIGHT))
        # cv2.imshow("resized",resized)
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