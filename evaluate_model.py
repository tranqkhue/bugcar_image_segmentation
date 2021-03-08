import tensorflow as tf
import cv2
import numpy as np
import os
from models import ENET, DeepLabV3
gpus = tf.config.experimental.list_physical_devices('GPU')
print("gpu:", gpus)
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], \
         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=700)])
    except RuntimeError as e:
        print(e)
saver = tf.train.Checkpoint()
model = DeepLabV3("pretrained_models/frozen_inference_graph.pb")
# model.save_weights("model_weight.h5")
IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])

# print(model.summary())
# tf.keras.utils.plot_model(model)
# sess = tf.compat.v1.keras.backend.get_session()
# save_path = saver.save("model.ckpt")


def process_image(img):
    # convert image from bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 256))
    # cv2.imshow("img",img)
    img = img / 256.0
    img = np.subtract(img, IMAGE_MEAN)
    img = np.divide(img, IMAGE_STD)
    imgs = np.array([img])
    #change image shape from [1,256,512,3] to [1,3,256,512]
    imgs = np.swapaxes(imgs, 2, 3)
    imgs = np.swapaxes(imgs, 1, 2)

    output = model.predict(imgs)[0]
    output = np.argmax(output, axis=0)
    output = np.where(output <= 1, 1, 0).astype(np.uint8)
    return output


def calculate_accuracy(pred, mask):
    intersection = np.bitwise_and(pred, mask)
    union = np.bitwise_or(pred, mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def evaluate_image(img, mask):
    output = process_image(img)
    mask = cv2.resize(mask, (256, 128))
    iou = calculate_accuracy(output, mask)
    return iou, output


def evaluate_folder():
    DATA_SRC = "../data/Road_data/self-collected dataset"
    IMG_SRC = "../data/Road_data/self-collected dataset/img"
    MASK_SRC = "../data/Road_data/self-collected dataset/masks_machine"
    img_list = os.listdir(IMG_SRC)
    mask_list = os.listdir(MASK_SRC)
    acc_list = np.zeros(shape=(len(img_list)))
    # assume files in both folders have same order.
    for i, img_name in enumerate(img_list):
        base_name = img_name[:-4]
        full_img_path = os.path.join(IMG_SRC, img_name)
        full_mask_path = os.path.join(MASK_SRC, base_name + ".png")
        img = cv2.imread(full_img_path)
        mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        acc, pred = evaluate_image(img, mask)
        if (acc < 0.8):
            new_img = np.zeros(shape=(1024, 1024, 3))
            new_img[0:512, :, :] = cv2.resize(img, (1024, 512))
            pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR) * 255
            new_img[512:1024, :] = cv2.resize(pred, (1024, 512))
            filename = "bad_inference/acc" + str(acc) + ".jpg"
            cv2.imwrite(filename, new_img)
        print("accuracy : ", acc)
        acc_list[i] = acc

    acc = np.sum(acc_list) / len(acc_list)
    print("total accuracy is: ", acc)


evaluate_folder()
