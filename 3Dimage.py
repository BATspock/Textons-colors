import cv2
import numpy as np 
from stl import mesh
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt 
from skimage import measure

image = cv2.imread("center_test: 1.png")
print(image.shape)
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image1.shape)
cv2.imshow("image1", image1)
image1 = cv2.bitwise_not(image1)
image1 = cv2.dilate(image1, kernel=np.ones((7,7)), iterations=1)
image1 = cv2.bitwise_not(image1)
image2 = cv2.imread("center_processing_test: 0.png",0)
cv2.imshow("image2", image2)
show = np.zeros_like(image)
show[image1 == 0] = [0,0,255]
show[image2 == 0] = [0,255,255]
show = cv2.medianBlur(show,5)

cv2.imshow("check", show)
print(show.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("final.png", show)