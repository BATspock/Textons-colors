import cv2
import sys
import numpy as np 
image = cv2.imread(sys.argv[1])
im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.blur(thresh, (3,3))
thres = cv2.dilate(thresh, np.ones((5,5)), iterations=3)
copy = np.zeros_like(thres)
copy[thresh>0]=255
or_region = cv2.bitwise_or(im, copy)
hatched_region = np.zeros_like(or_region)
threshold = np.mean(or_region)
#print(threshold)

hatched_region[or_region < 250] = 255

hatched_region = cv2.morphologyEx(hatched_region, cv2.MORPH_OPEN, np.ones((3,3)))
hatched_region = cv2.medianBlur(hatched_region, 5)

separator_lines = np.zeros_like(image)
separator_lines [copy == 255] = [0,0,255]

hatching = np.zeros_like(image)
hatching[hatched_region==255] = [0,255,255]
#hatching = cv2.morphologyEx(hatching, cv2.MORPH_CLOSE, np.ones((7,7)))
#hatching = cv2.medianBlur(hatching, 7)

final =  cv2.bitwise_or(hatching, separator_lines)
#final = cv2.blur(final, (5,5))
print(final.shape)
#import numpy as np
#from math import sqrt

#def compare(image1, image2):
#    return ((((image1-image2)**2)**(1/2)).mean())  

#print(compare(final,cv2.imread('/home/adityakishore/workspace/IIITB/Aditya/textons_colors/images_from_textons/10.jpgcenter: 0.png',0) ))

cv2.imshow('or', hatching)
cv2.imshow('separator', separator_lines)
cv2.imshow('final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
