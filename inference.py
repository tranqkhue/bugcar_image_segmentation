import tensorflow as tf 
import numpy as np 
import cv2
import time

#Limit memory usage for Dekstop GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], \
    	[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=600)])
  except RuntimeError as e:
    print(e)

#Load model
model = tf.keras.models.load_model('test.hdf5')

#Load, crop and resize
original_img = cv2.imread('test05.jpg')
#cropped_original = original_img[120:1080, 0:1920]
rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#cropped = rgb_img[120:1080, 0:1920]
#resized = cv2.resize(cropped, (512,256))
#cropped = cropped.astype('float32')
INPUT_SIZE = [256,512]
INPUT_SIZE1 = [128,256]
resized = tf.image.resize(rgb_img, INPUT_SIZE1, method='bilinear')

#Normalize the image
normalized = resized/256.0
IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD  = np.array([0.229, 0.224, 0.225])
sub = np.subtract(normalized, IMAGE_MEAN)
div = np.divide(sub, IMAGE_STD)

#Swap axes
img  = np.swapaxes(div, 1, 2)
img  = np.swapaxes(img, 0, 1)
imgs = np.array([img])

#Run inference for multiple times (for benchmarking)
for i in range(30):
	t0 = time.time()
	raw_output = model.predict(imgs)[0]
	print('FPS:  ' + str(1/(time.time() - t0)))

#Get highest probability class for each pixel
nc     = len(raw_output[:,0,0]) #Number of classes
scores = np.max(raw_output, axis=0)
output = np.argmax(raw_output, axis=0)
#scores_per_class = [np.sum(scores[output == c]) / np.sum(scores) for c in range(nc)]

#Find pixels that belong to classes (0 is road, 1 is road mark, 2 is sidewalk):
mask_viz_0 = (output==0)
mask_viz_0 = mask_viz_0.astype(np.uint8)*255
mask_viz_1 = (output==1)
mask_viz_1 = mask_viz_1.astype(np.uint8)*255
mask_viz   = cv2.bitwise_or(mask_viz_0, mask_viz_1)
#mask_viz   = cv2.resize(mask_viz, (1024, 512))

#Output image
(out_height, out_width) = mask_viz.shape
resized  = cv2.resize(rgb_img, (out_width, out_height))
output_viz = cv2.bitwise_and(resized, resized, mask=mask_viz)
cv2.imwrite('output.png', output_viz)