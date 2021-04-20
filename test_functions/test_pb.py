import tensorflow as tf
import time
import cv2
import numpy as np
from utils import enet_preprocessing
# cam = cv2.VideoCapture(0)
GRAPH_PB_PATH = './random_dir/my_model.pb'
INPUT_TENSOR_NAME = "input0:0"
OUTPUT_TENSOR_NAME = "CATkrIDy/concat:0"

label_to_colours = {
    0: (128, 64, 128),  # road
    1: (244, 35, 232),  #sidewalk
    2: (70, 70, 70),  #pavement? 
    3: (102, 102, 156),  #wall
    4: (190, 153, 153),  #fence
    5: (153, 153, 153),  #pole
    6: (250, 170, 30),  # 'traffic
    7: (220, 220, 0),  #'traffic
    8: (107, 142, 35),  #'vegetation
    9: (152, 251, 152),  #'terrain
    10: (70, 130, 180),  #'sky
    11: (220, 20, 60),  #'person
    12: (255, 0, 0),  #'vehicle?
    13: (0, 0, 142),  #'car
    14: (0, 0, 70),  #'truck
    15: (0, 60, 100)  #'bus
}

label_to_colours_array = np.array([[128, 128, 128], [128, 0,
                                                     0], [192, 192, 128],
                                   [128, 64, 128], [60, 40,
                                                    222], [128, 128, 0],
                                   [192, 128, 128], [64, 64, 128],
                                   [64, 0, 128], [64, 64, 0], [0, 128, 192],
                                   [0, 0, 0], [128, 128, 192], [192, 64, 128],
                                   [0, 64, 192], [255, 0, 0]]).astype(np.uint8)

# with tf.Graph().as_default():
#     with tf.compat.v1.Session() as sess:
#         tf.compat.v1.keras.backend.set_session(sess)
#         model = tf.compat.v1.keras.models.load_model("model.h5")
#         print(model.outputs)
#         # Create, compile and train model...
#         frozen_graph = freeze_session(
#             tf.compat.v1.keras.backend.get_session(),
#             output_names=[out.op.name for out in model.outputs])
#         tf.compat.v1.train.write_graph(frozen_graph,
#                                        "random_dir",
#                                        "my_model.pb",
#                                        as_text=False)

cam = cv2.VideoCapture("test01.webm")


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # print(gamma_table)
    return cv2.LUT(img, gamma_table)


# cam.set(cv2.CAP_PROP_EXPOSURE, -7)
with tf.compat.v1.Session() as sess:
    print("load graph")
    with tf.compat.v1.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
        string = f.read()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(string)
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes = [n for n in graph_def.node]
        names = []
        # image = enet_preprocessing(cv2.imread('3.png'))
        for t in graph_nodes:
            names.append(t.name)
        print(names)
        while (True):
            ret, frame = cam.read()
            if (ret == True):
                # frame = gamma_trans(frame, 2.5)
              ##cv2.imshow("frame", frame)
                frame = enet_preprocessing(frame)
                t0 = time.time()
                segmap = sess.run(OUTPUT_TENSOR_NAME,
                                  feed_dict={INPUT_TENSOR_NAME: frame})[0]
                segmap_exponent = np.exp(segmap)

                road = np.divide(segmap_exponent[0],
                                 np.sum(segmap_exponent, axis=0)) * 255
                road = road.astype(np.uint8)
                print(road)
              ##cv2.imshow("road", road)
                print("FPS:", 1 / (time.time() - t0))
                segmap = np.argmax(segmap, axis=0).astype(np.uint8)
                new_image = np.zeros(shape=(segmap.shape[0], segmap.shape[1],
                                            3))

                for i in range(segmap.shape[0]):
                    for j in range(segmap.shape[1]):
                        label = segmap[i, j]
                        new_image[i, j] = np.array(label_to_colours[label])
                new_image = new_image.astype(np.uint8)
              ##cv2.imshow("lmao", new_image)

                a = cv2.waitKey(1) & 0xFF
                if a == ord('q'):
                    cv2.waitKey(0)
