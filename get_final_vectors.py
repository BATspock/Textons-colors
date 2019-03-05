import cv2
import sys
import numpy as np
from sklearn.preprocessing import normalize

class createVector(object):
    """
    This class creates the final vector
    """

    def __init__(self, vector, image, color_image):
        self.v = vector
        self.im = image
        self.im_color = color_image
        #print(self.im_color[50,50,:])
    def generateVectors(self):
        """
        generate feature vector with row, col and cluster center values
        """
        hold = []
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                b, g, r = self.im_color[i, j, :]
                hold.append([b/255.0, g/255.0, r/255.0, i, j])
        hold = np.array(hold)
        self.v = normalize(self.v, axis=0)
        complete_vector = np.concatenate((self.v, hold), axis= 1)
        #print("complete vector ...")
        #print(complete_vector.shape)
        center_indices = np.full((complete_vector.shape[0], 1), -1)
        complete_vector = np.concatenate((complete_vector, center_indices), axis=1)
        print('complete vector')
        print(complete_vector.shape)
        return complete_vector