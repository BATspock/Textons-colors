import numpy as np 
from preprocess import preprocess
from texton_color_utils import Textons
import cv2
import sys

im = cv2.imread(sys.argv[1])
#im = cv2.blur(im,(5,5))
#im = cv2.medianBlur(im, 5)
#ob = preprocess(im)
#im, _ = ob.kmeans(3)

#im = cv2.medianBlur(im,3)

tex = Textons(im, int(sys.argv[2]), int(sys.argv[3]), 1)
im = tex.textons()
centers = np.unique(im)

for _ in range(int(sys.argv[2])):
    show = np.zeros_like(im)
    show[im==centers[_]] = 255
    #show = cv2.morphologyEx(show, cv2.MORPH_OPEN, np.ones((5, 5)))
    cv2.imwrite('center: '+str(_) + '.png', cv2.bitwise_not(show))

for _ in range(int(sys.argv[2])):
    show = np.zeros_like(im)
    show[im==centers[_]] = 255
    show = cv2.morphologyEx(show, cv2.MORPH_OPEN, np.ones((5, 5)))
    cv2.imwrite('center processing: '+str(_) + '.png', cv2.bitwise_not(show))


cv2.imshow("image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()


