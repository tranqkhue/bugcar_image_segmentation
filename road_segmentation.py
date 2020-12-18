#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 08:49:24 2020

@author: Tran Quoc Khue
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image
import cv2
import time

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'import/ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'import/SemanticPredictions:0'
    INPUT_SIZE = 1024
    FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
    
    def __init__(self, model_name):
        self.graph = tf.Graph()
        FROZEN_GRAPH_PATH = model_name+'/'+ self.FROZEN_GRAPH_NAME

        with gfile.FastGFile(FROZEN_GRAPH_PATH,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def)
        self.sess = tf.Session(graph=self.graph)
        
    def run(self, img):
        #image: A cv2.img object, raw input image. 
        height,width,depth = img.shape
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(img,target_size)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

model = DeepLabModel('deeplab_mnv3_large_cityscapes_trainfine')
#print(model.graph.get_operations())
img = Image.open('input/test_input.jpg')
try:
    img_resized = np.delete(np.array(img.resize((1024, 768))), 3, axis=2)
except IndexError:
    img_resized = np.array(img.resize((1024, 768)))
t0 = time.time()
result = model.run(img)
print('Model run time:  ', str(time.time()-t0))
array_result = result[1]
print(array_result.shape)
cv2.imwrite('damn_son.jpg',array_result)

for k in range(18):
    output = np.zeros(((768, 1024, 3)), dtype=np.uint8)
    for i in range(len(array_result)):
        for j in range(len(array_result[i])):
            if (array_result[i][j]==k):
                output[i][j]=img_resized[i][j]  
            else:
                output[i][j]=(0,0,0)
                
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output_path = 'output/test_segmentation_' + str(k) + '.png'
    cv2.imwrite(output_path, output)