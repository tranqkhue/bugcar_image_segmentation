import logging, os
from queue import Full
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import cv2
import rospy
import time
import multiprocessing as mp
from multiprocessing import Process, Queue
from calibration import INPUT_SHAPE
from bev import bev_transform_tools
import occgrid_to_ros
from utils import contour_noise_removal, enet_preprocessing

logger = logging.Logger('lmao')


#================================================================================
def read_data(queue):
    cap = cv2.VideoCapture(6)
    cap.set(3, 1280)
    cap.set(4, 720)
    while (True):
        ret, frame = cap.read()
        if (ret == True):
            try:
                queue.put_nowait(frame)
            except Full:
                pass


#---------------------------------------------------------------------------------
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


#================================================================================
# Initialize
if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = Queue(maxsize=1)
    read_thread = Process(target=read_data, args=(q, ))
    read_thread.start()
    publisher = occgrid_to_ros.init_node()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0], \
             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=600)])
        except RuntimeError as e:
            print(e)
    # model = tf.keras.models.load_model('model.hdf5')
    model = ENET('my_model.pb')
    perspective_transformer = bev_transform_tools.fromJSON(
        'calibration_data.json')
    matrix = perspective_transformer._bev_matrix
    #print("Check model input:  ",model.inputs)

    #---------------------------------------------------------------------------------
    try:
        while True:
            t0 = time.time()
            frame = q.get()
            # Prepocessing input
            batch_frame = enet_preprocessing(frame)
            # Run inference and process the results
            t1 = time.time()
            inference_result = model.predict(batch_frame)[0]
            inference_fps = 1 / (time.time() - t1)
            result_by_class = np.argmax(inference_result, axis=0)
            segmap    = np.bitwise_or(result_by_class==0, result_by_class==1)\
                    .astype(np.uint8)
            # Remove road branches (or noise) that are not connected to main branches
            # Main road branches go from the bottom part of the RGB map
            # (should be) right front of the vehicle
            contour_noise_removed = contour_noise_removal(segmap)
            # Need to resize to be the same with the image size in calibration process
            resized_segmap = cv2.resize(contour_noise_removed, INPUT_SHAPE)
            occ_grid = perspective_transformer.create_occupancy_grid(
                resized_segmap, perspective_transformer._bev_matrix,
                perspective_transformer.width, perspective_transformer.height,
                perspective_transformer.map_size,
                perspective_transformer.map_resolution,
                perspective_transformer.cm_per_px)
            msg    = occgrid_to_ros.og_msg(occ_grid,\
                     perspective_transformer.map_resolution,\
                     perspective_transformer.map_size,rospy.Time.now())
            publisher.publish(msg)
            cv2.imshow("frame", frame)
            cv2.imshow("occgrid", occ_grid)
            print('Inference FPS:  ',  format(inference_fps, '.2f'), ' | ',\
               'Total loop FPS:  ', format(1/(time.time()-t0), '.2f'))
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
    finally:
        cv2.destroyAllWindows()
        q.close()
        read_thread.terminate()
