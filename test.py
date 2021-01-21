import cv2
import numpy as np;
while(1):
    x = np.zeros((512,512))
    a = cv2.imshow("img",x)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k) # else print its val