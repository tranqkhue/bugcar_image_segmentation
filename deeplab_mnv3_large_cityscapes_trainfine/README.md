# Bugcar's Image Segmentation TEST SCRIPTS
## Network: DeepLabv3 with MobileNetv3 backbone, trained with Cityscapes
[DeepLab Model Zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)
## Demo road segmentation with network that trained with Cityscapes dataset ONLY!!! No transfer learning has been done
Input
![alt text](https://github.com/tranqkhue/bugcar_image_segmentation/blob/master/test_input_2.jpg?raw=true)
Output segmented road
![alt text](https://github.com/tranqkhue/bugcar_image_segmentation/blob/master/output/test_segmentation_0.png?raw=true)
## Requirements
- OpenCV>=3
- Tensorflow==1.15

## Label maps
[Cityscapes Label Map](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)

**Citycapes label ID in inference is from 0 to 18 (excludes 255 and -1 ID)**
>'road'  :             0,
>'sidewalk':           1,
>'building':           2,
>'wall':               3,
>'fence':              4,
>'pole':               5, 
>'traffic light':      6, 
>'traffic sign':       7,
>'vegetation':         8,
>'terrain':            9, 
>'sky':                10, 
>'person':             11, 
>'rider':              12, 
>'car':                13, 
>'truck':              14, 
>'bus':                15, 
>'train':              16,
>'motorcycle':         17,
>'bicycle':	      18
