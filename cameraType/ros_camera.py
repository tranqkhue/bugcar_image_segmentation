print(__name__)
from cmath import e
import logging
import multiprocessing as mp
import os
import threading
import time
import cv2
import numpy as np
import yaml
from cameraType.baseCamera import BaseCamera
import rospy
from sensor_msgs.msg import CompressedImage
import cv_bridge
import traceback
from stitcher import Stitcher



class ROSCamera(BaseCamera):
    # Generate the ROS camera object, but this Object assume that the program calling this object has already
    # initialized a node
    def __init__(self,topic_name,camera_info_file= None):
        self.logger = logging.getLogger("__main__")
        self.ev = threading.Event()
        self.img= None
        self.sub = rospy.Subscriber(topic_name, CompressedImage, self.getImage)
        self.bridge = cv_bridge.core.CvBridge()
        
        if(camera_info_file is not None):
            with open(camera_info_file,"r") as f: 
                obj = yaml.safe_load(f)
                self.K =  np.reshape(np.array(obj["intrinsic_matrix"]),(3,3))
                self.distortion_coeffs = obj["distortion"]
        else:
            self.K = None
            self.distortion_coeffs = None
    def get_bgr_frame(self):
        self.ev.wait()
        self.ev.clear()
        return self.img
    def get_intrinsic_matrix(self):
        return self.K
    def getImage(self,data):
        np_arr = np.fromstring(data.data, np.uint8)
        self.img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # self.logger.info("%f",time.time())
        self.ev.set()
    def get_distortion_coeff(self):
        return self.distortion_coeffs
    def stop(self):
        self.sub.unregister()


class ROSStitchCamera(BaseCamera):
    # Generate the ROS camera object, but this Object assume that the program calling this object has already
    # initialized a node
    def __init__(self,left_img_topic,right_img_topic,camera_info_file,stitcher: Stitcher=Stitcher() ):
        self.imgLeft= None
        self.imgRight = None
        self.left_new_image = False
        self.right_new_image = False
        self.stitcher = stitcher
        self.ev = threading.Event()

        # print(camera_info_file)
        with open(camera_info_file,"r") as f: 
            obj = yaml.safe_load(f)
            # print(obj)
            self.K =  np.reshape(np.array(obj["intrinsic_matrix"]),(3,3))
            self.distortion_coeffs = obj["distortion"]
        self.sub_left = rospy.Subscriber(left_img_topic, CompressedImage, self.__getImageLeft, queue_size=3)
        self.sub_right = rospy.Subscriber(right_img_topic, CompressedImage, self.__getImageRight, queue_size=3)
        self.bridge = cv_bridge.core.CvBridge()

    # stitch the left image and the right image and the return the whole frame
    def get_bgr_frame(self):
        self.ev.wait()
        self.ev.clear()
        self.left_new_image = False
        self.right_new_image = False
        h, w, _ = self.imgLeft.shape
        # print(self.imgLeft.shape)
        leftImgCropped = self.imgLeft[:, 0:int(w*5.8/10)]
        rightImgCropped  = self.imgRight[:, int(w*4/10):w]
        stitched = None
        try:
            stitched = self.stitcher.stitch([leftImgCropped, rightImgCropped])
        except TypeError as e:
            traceback.print_exc()
        return stitched[0]
    def get_stitch_image_with_matching_features(self):
            self.ev.wait()
            self.ev.clear()
        # if (self.left_new_image == True) & (self.right_new_image == True):
            self.left_new_image = False
            self.right_new_image = False
            h, w, _ = self.imgLeft.shape
            leftImgCropped = self.imgLeft[:, 0:int(w*5.675/10)]
            rightImgCropped  = self.imgRight[:, int(w*5/11):w]
            cv2.imshow("left",leftImgCropped)
            cv2.imshow("right",rightImgCropped)
            stitched = None
            
            try:
                stitched = self.stitcher.stitch([leftImgCropped, rightImgCropped])
            except TypeError as e:
                traceback.print_exc()
            if(stitched is not None):
                return stitched
            return stitched
        # return None
    def get_intrinsic_matrix(self):
        return self.K
        
    def __getImageLeft(self,data):
        # sellf.imgLeft = self.bridge.imgmsg_to_cv2(data,"bgr8")
        # self.imgLeft = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        np_arr = np.fromstring(data.data, np.uint8)
        self.imgLeft = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # print(self.imgLeft is None)
        self.left_new_image = True
        if(self.right_new_image):
            self.ev.set()
        # the callback still run normally, but the ev variable cannot be set
        # print("is called stithced left")

     
    def __getImageRight(self,data):
        # self.imgRight = self.bridge.imgmsg_to_cv2(data,"bgr8")
        # self.imgRight = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        np_arr = np.fromstring(data.data, np.uint8)
        self.imgRight = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.right_new_image = True
        if(self.left_new_image):
            self.ev.set()

    def get_distortion_coeff(self):
        distortion_coeffs = 0
        return distortion_coeffs
    def stop(self):
        self.sub_left.unregister()
        self.sub_right.unregister()
